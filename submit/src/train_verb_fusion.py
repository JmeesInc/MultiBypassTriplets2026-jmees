"""Step 2b/2c: Swin3D verb model + frozen target/instrument fusion -> triplet (ivt) prediction.

- Verb branch: Swin3D-B (Kinetics-400), input = 16-frame causal clip @ stride2 ending at keyframe.
- Target branch (frozen, cached): convnext pooled feature(1024) + logits(15).
- Instrument branch (frozen, cached): cfg2 presence(12) + low-res 12-class mask(28x28) used as a
  spatial prior (mask-guided pool of the keyframe spatial slice of the Swin3D feature map).
- Fusion: concat[global, mask-guided, target_proj, presence, target_logits] -> MLP -> heads
  ivt(85, primary) + aux i(12)/v(13)/t(15). Loss = sum of BCE (pos_weight).
- Eval: videowise ivt mAP; i/v/t derived from ivt via component_maps (official style).

Usage: GPU=0 python src/train_verb_fusion.py --config config_verb.yaml
"""
import argparse
import csv
import glob
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

HERE = os.path.dirname(__file__)
REPO = os.environ.get("MB_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VID_DIR = os.environ.get("MB_VID_DIR", os.path.join(os.environ.get("MB_DATA", "data"), "videos"))
LBL_DIR = os.environ.get("MB_LBL_DIR", os.path.join(os.environ.get("MB_DATA", "data"), "label_files_challenge"))
CACHE = os.path.join(HERE, "..", "results", "frozen_cache")
sys.path.insert(0, os.path.join(REPO, "reference/starter_kit"))
from utils.triplet_mappings import triplet_maps  # noqa

MB = triplet_maps["multibypasst40"]
CM = np.array(MB["component_maps"])  # [85,6] cols ivt,i,v,t,iv,it
NI, NV, NT, NIVT = 12, 13, 15, 85
# pairwise (SPIRIT の I-T / I-V 2枝): 有効な組だけの compact クラス。it=54, iv=39。
# triplet を unary(i×v×t, 純合成は0.434に劣化)でなく pairwise(it×iv)で合成すると密に監督が付き、
# train ゼロの triplet も (i,t)/(i,v) を見ていれば scoring できる（hidden test zero-train 対策）。
IT_MAP = CM[:, 5].astype(np.int64)  # [85] triplet -> it index
IV_MAP = CM[:, 4].astype(np.int64)  # [85] triplet -> iv index
NIT, NIV = int(IT_MAP.max()) + 1, int(IV_MAP.max()) + 1  # 54, 39
# compact it/iv edge -> component indices (for GATv2 graph reasoning, B): it_idx -> (i,t), iv_idx -> (i,v)
IT_I = np.zeros(NIT, np.int64); IT_T = np.zeros(NIT, np.int64); IV_I = np.zeros(NIV, np.int64); IV_V = np.zeros(NIV, np.int64)
for _k in range(NIVT):
    IT_I[CM[_k, 5]] = CM[_k, 1]; IT_T[CM[_k, 5]] = CM[_k, 3]
    IV_I[CM[_k, 4]] = CM[_k, 1]; IV_V[CM[_k, 4]] = CM[_k, 2]
KMEAN = np.array([0.485, 0.456, 0.406], np.float32); KSTD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class EMA:
    """Exponential moving average of model parameters (regularises the overfitting fusion).
    Keeps a float32 shadow of every param; buffers are copied verbatim at apply-time.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    @torch.no_grad()
    def copy_to(self, model):
        """load EMA weights into a (separate) model instance for evaluation."""
        msd = model.state_dict()
        for n in self.shadow:
            msd[n].copy_(self.shadow[n].to(msd[n].dtype))
        model.load_state_dict(msd)


def setup_logger(outdir):
    lg = logging.getLogger("verb"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    fh = logging.FileHandler(os.path.join(outdir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def load_folds(p):
    return {r["video"]: int(r["fold"]) for r in csv.DictReader(open(p))}


def init_wandb(cfg, lg):
    """Init a W&B run (config-gated via cfg['wandb'], non-fatal)."""
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    try:
        import wandb
        rid = f"{cfg['experiment']['name']}_fold{cfg['data']['val_fold']}"
        wb = wandb.init(project=wcfg.get("project", "multibypass-triplet"),
                        name=cfg["experiment"]["name"], group=wcfg.get("group", "expA02_fusion"),
                        id=rid, resume="allow", config=cfg,
                        settings=wandb.Settings(_disable_stats=True))
        lg.info(f"wandb run: {wb.url}")
        return wb
    except Exception as e:
        lg.warning(f"wandb init failed ({e}); continuing without wandb")
        return None


def build_index(cfg, v2f, is_val):
    """[(video, keyframe_fid, cache_row, ivt(85), i(12), v(13), t(15))]."""
    vf = cfg["data"]["val_fold"]; items = []
    for lf in sorted(glob.glob(os.path.join(LBL_DIR, "*.json"))):
        vid = os.path.splitext(os.path.basename(lf))[0]
        if vid not in v2f or (v2f[vid] == vf) != is_val:
            continue
        cpath = os.path.join(CACHE, f"{vid}.npz")
        if not os.path.exists(cpath):
            continue
        fid2row = {int(f): r for r, f in enumerate(np.load(cpath)["fids"])}
        d = json.load(open(lf))
        byimg = defaultdict(list)
        for a in d["annotations"]:
            byimg[a["image_id"]].append(a)
        for im in d["images"]:
            fid = im["id"]
            if fid not in fid2row:
                continue  # only cached keyframes
            anns = byimg.get(fid, [])
            if not anns and cfg["data"].get("drop_empty", True):
                continue
            ivt = np.zeros(NIVT, np.float32); iy = np.zeros(NI, np.float32)
            vy = np.zeros(NV, np.float32); ty = np.zeros(NT, np.float32)
            ity = np.zeros(NIT, np.float32); ivy = np.zeros(NIV, np.float32)  # pairwise it/iv multihot
            for a in anns:
                ivt[a["category_id"]] = 1.0; iy[a["instrument_id"]] = 1.0
                vy[a["verb_id"]] = 1.0; ty[a["target_id"]] = 1.0
                ity[IT_MAP[a["category_id"]]] = 1.0; ivy[IV_MAP[a["category_id"]]] = 1.0
            items.append((vid, fid, fid2row[fid], ivt, iy, vy, ty, ity, ivy))
    return items


class ClipDS(Dataset):
    def __init__(self, items, cfg, train, teacher_lut=None):
        self.items = items; self.cfg = cfg; self.train = train
        self.T = cfg["data"]["clip_len"]; self.stride = cfg["data"]["clip_stride"]
        self.img = cfg["data"]["img"]
        self.teacher_lut = teacher_lut  # D: 蒸留教師 soft ivt {(vid,fid): [85]}
        self.cache = {}  # vid -> npz arrays (lazy)
        a = cfg.get("aug", {})  # augmentation (train only); defaults reproduce v1 behaviour
        self.a_hflip = a.get("hflip", 0.5)      # p(horizontal flip); flips clip AND imask together
        self.a_color = a.get("color", 0.0)      # brightness/contrast jitter magnitude (0=off)
        self.a_tjit = a.get("temporal_jitter", 0)  # +-N stride jitter (keyframe stays aligned)
        # multi-scale (dilated) temporal sampling: explicit per-frame offsets from the keyframe.
        # e.g. [-128,...,-1,1,...,128] -> 16 frames spanning ~4min, dense near kf, sparse far.
        # Overrides clip_len/stride. Valid because inference has the whole video (offline Docker).
        self.offsets = cfg["data"].get("frame_offsets", None)
        # verb-focused augs (train only; keyframe/label frame always preserved). imask co-transformed.
        self.tsj = a.get("tsample_jitter", None)   # aug1: {strides:[1,2,3], phase:true} window+phase jitter
        self.fco = a.get("frame_cutout", None)     # aug2: {p:0.5, max_drop:2} drop consecutive non-key frames
        self.rrc = a.get("rrc", None)              # aug3: {p:0.5, scale:[0.8,1.0]} clip-consistent RandomResizedCrop
        self.slc = a.get("sliding_crop", None)     # aug4: {p:0.5, zoom_max:1.15, max_shift:0.15} pseudo camera motion
        self.dyna = a.get("dyna", None)            # aug5: {mag:0.3, n_freq:3} DynaAugment (smooth temporal photometric)

    def _cache(self, vid):
        if vid not in self.cache:
            d = np.load(os.path.join(CACHE, f"{vid}.npz"))
            self.cache[vid] = {k: d[k] for k in d.files}
        return self.cache[vid]

    def __len__(self):
        return len(self.items)

    def _read(self, vid, fid):
        fp = os.path.join(VID_DIR, vid, f"{max(fid,0):06d}.jpg")
        im = cv2.imread(fp)
        if im is None:
            im = np.zeros((self.img, self.img, 3), np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (self.img, self.img)), cv2.COLOR_BGR2RGB)
        return im

    def _crop_resize(self, frame, box):  # frame [H,W,3] float, box=(x0,y0,x1,y1) in [0,1] -> resized to img
        H, W = frame.shape[:2]
        x0, y0 = max(0, int(box[0] * W)), max(0, int(box[1] * H))
        x1, y1 = min(W, int(box[2] * W)), min(H, int(box[3] * H))
        x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)
        return cv2.resize(frame[y0:y1, x0:x1], (self.img, self.img))

    def _crop_mask(self, mask, box, size=28):  # mask [12,M,M] -> crop by same [0,1] box, resize to size
        M = mask.shape[1]
        x0, y0 = max(0, int(box[0] * M)), max(0, int(box[1] * M))
        x1, y1 = min(M, int(box[2] * M)), min(M, int(box[3] * M))
        x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)
        out = np.zeros((mask.shape[0], size, size), np.float32)
        for ch in range(mask.shape[0]):
            out[ch] = cv2.resize(mask[ch, y0:y1, x0:x1], (size, size))
        return out

    def _dyna_curve(self, T, mag, nfreq):  # aug5: smooth [-mag,mag] curve over T frames (sum of sinusoids)
        t = np.linspace(0, 1, T); curve = np.zeros(T, np.float32)
        for _ in range(nfreq):
            f = random.uniform(0.5, 3.0); ph = random.uniform(0, 2 * math.pi); amp = random.uniform(0, 1)
            curve += amp * np.sin(2 * math.pi * f * t + ph)
        return curve / (np.abs(curve).max() + 1e-6) * mag

    def _sliding_boxes(self):  # aug4: crop box drifts linearly start->end across the clip (pseudo camera motion)
        z = random.uniform(1.0, self.slc.get("zoom_max", 1.15)); side = 1.0 / z
        ms = self.slc.get("max_shift", 0.15)
        cx0 = random.uniform(side / 2, 1 - side / 2); cy0 = random.uniform(side / 2, 1 - side / 2)
        ang = random.uniform(0, 2 * math.pi); d = random.uniform(0, ms)
        cx1 = min(max(cx0 + d * math.cos(ang), side / 2), 1 - side / 2)
        cy1 = min(max(cy0 + d * math.sin(ang), side / 2), 1 - side / 2)
        boxes = []
        for i in range(self.T):
            a = i / (self.T - 1); cx = cx0 * (1 - a) + cx1 * a; cy = cy0 * (1 - a) + cy1 * a
            boxes.append((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        return boxes, boxes[-1]  # per-frame boxes + keyframe(last) box for imask

    def __getitem__(self, idx):
        vid, kf, row, ivt, iy, vy, ty, ity, ivy = self.items[idx]
        T = self.T
        if self.offsets is not None:
            j = random.uniform(0.7, 1.4) if (self.train and self.a_tjit) else 1.0  # scale jitter
            fids = [kf + int(round(o * j)) for o in self.offsets]  # dilated multi-scale sampling
        elif self.train and self.tsj:  # aug1: window(stride)+phase jitter; keyframe (last) stays = kf
            stride = random.choice(self.tsj.get("strides", [1, 2, 3]))
            phase = random.uniform(-0.5, 0.5) * stride if self.tsj.get("phase", True) else 0.0
            fids = [kf - int(round(stride * (T - 1 - i) - phase)) for i in range(T - 1)] + [kf]
        else:
            stride = self.stride
            if self.train and self.a_tjit:  # temporal stride jitter; keyframe (last) stays = kf
                stride = max(1, self.stride + random.randint(-self.a_tjit, self.a_tjit))
            fids = [kf - stride * (T - 1 - i) for i in range(T)]  # causal, ends at kf
        clip = np.stack([self._read(vid, max(f, 0)) for f in fids]).astype(np.float32) / 255.0  # [T,H,W,3] in [0,1]
        # geometric aug (aug3 rrc XOR aug4 sliding crop); track keyframe box for imask co-transform
        key_box = None
        if self.train and self.rrc and random.random() < self.rrc.get("p", 0.5):  # aug3: clip-consistent RRC
            smin, smax = self.rrc.get("scale", [0.8, 1.0])
            side = math.sqrt(random.uniform(smin, smax))
            x0 = random.uniform(0, 1 - side); y0 = random.uniform(0, 1 - side)
            key_box = (x0, y0, x0 + side, y0 + side)
            clip = np.stack([self._crop_resize(clip[i], key_box) for i in range(T)])
        elif self.train and self.slc and random.random() < self.slc.get("p", 0.5):  # aug4: sliding crop
            boxes, key_box = self._sliding_boxes()
            clip = np.stack([self._crop_resize(clip[i], boxes[i]) for i in range(T)])
        if self.train and self.fco and random.random() < self.fco.get("p", 0.5):  # aug2: FrameCutOut (not last)
            n = random.randint(1, self.fco.get("max_drop", 2))
            if T - 1 - n > 0:
                st = random.randint(0, T - 1 - n); clip[st:st + n] = 0.0
        if self.train and self.dyna:  # aug5 DynaAugment: per-frame smooth brightness/contrast variation
            mag = self.dyna.get("mag", 0.3); nf = self.dyna.get("n_freq", 3)
            cb = self._dyna_curve(clip.shape[0], mag, nf); cc = self._dyna_curve(clip.shape[0], mag, nf)
            g = float(clip.mean())
            for f in range(clip.shape[0]):
                clip[f] = np.clip((clip[f] * (1 + cb[f]) - g) * (1 + cc[f]) + g, 0.0, 1.0)
        elif self.train and self.a_color > 0:  # temporal-consistent (static) photometric jitter
            br = 1.0 + random.uniform(-self.a_color, self.a_color)
            ct = 1.0 + random.uniform(-self.a_color, self.a_color)
            g = float(clip.mean())
            clip = np.clip((clip * br - g) * ct + g, 0.0, 1.0)
        clip = (clip - KMEAN) / KSTD                       # [T,H,W,3]
        do_flip = self.train and random.random() < self.a_hflip
        if do_flip:
            clip = clip[:, :, ::-1, :].copy()              # temporal-consistent hflip
        clip = torch.from_numpy(clip.transpose(3, 0, 1, 2))  # [3,T,H,W]
        c = self._cache(vid)
        imask = c["inst_mask"][row].astype(np.float32)     # [12,28,28]
        if key_box is not None:
            imask = self._crop_mask(imask, key_box)        # co-transform mask by keyframe crop box
        if do_flip:
            imask = imask[:, :, ::-1].copy()               # flip mask with clip to keep spatial alignment
        return {"clip": clip,
                "tfeat": torch.from_numpy(c["target_feat"][row].astype(np.float32)),
                "tlogit": torch.from_numpy(c["target_logits"][row].astype(np.float32)),
                "pres": torch.from_numpy(c["inst_presence"][row].astype(np.float32)),
                "imask": torch.from_numpy(imask),
                "ivt": torch.from_numpy(ivt), "i": torch.from_numpy(iy),
                "v": torch.from_numpy(vy), "t": torch.from_numpy(ty),
                "it": torch.from_numpy(ity), "iv": torch.from_numpy(ivy),
                **({"teacher": torch.from_numpy(self.teacher_lut.get((vid, kf), np.zeros(NIVT, np.float32))),
                    "has_teach": torch.tensor(1.0 if (vid, kf) in self.teacher_lut else 0.0)}
                   if self.teacher_lut is not None else {})}


class ClassQueryHead(nn.Module):
    """A(SPIRIT TUF移植): per-classクエリが Swin3D keyframe 空間トークンに cross-attention → per-class logit。

    素の Linear ヘッド(global特徴)に対し、クラスごとに関連空間領域へ注目して証拠を集約。
    query は learnable(text初期化はconfigで将来対応)。
    """

    def __init__(self, ncls, c_in=1024, d=256, nhead=4, dropout=0.1):
        super().__init__()
        self.q = nn.Embedding(ncls, d); nn.init.normal_(self.q.weight, std=0.02)
        self.kv = nn.Linear(c_in, d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 1)

    def forward(self, key_tokens):          # key_tokens [B, HW, c_in] -> [B, ncls]
        B = key_tokens.shape[0]
        q = self.q.weight.unsqueeze(0).expand(B, -1, -1)
        kv = self.kv(key_tokens)
        a, _ = self.attn(q, kv, kv)
        return self.out(self.ln(q + a)).squeeze(-1)


class TemporalAttnPool(nn.Module):
    """A(SPIRIT TCA移植): Swin3D特徴の時間次元を mean-pool で潰す代わりに、空間pool後の
    時間トークン [B,T',C] に CLS クエリで cross-attention して集約。動作の時間展開
    (staple/coagulate/dissect の多フレーム signature)を保持する。
    """

    def __init__(self, c, nhead=8, maxT=16, dropout=0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, maxT, c)); nn.init.trunc_normal_(self.pe, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, c)); nn.init.trunc_normal_(self.cls, std=0.02)
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                          # x [B,T',H',W',C] -> [B,C]
        B, T, H, W, C = x.shape
        xt = x.mean(dim=(2, 3)) + self.pe[:, :T]   # 空間pool -> 時間トークン [B,T',C] + 時間PE
        q = self.cls.expand(B, -1, -1)
        a, _ = self.attn(q, xt, xt)                # CLS が時間トークンに attention [B,1,C]
        return self.ln(a.squeeze(1) + xt.mean(1))  # + 時間平均 residual


def make_encoder(name):
    """Pluggable 3D video encoder -> (module w/ global-feature forward, out_dim). Kinetics-400 pretrained."""
    from torchvision.models import video as TV
    if name == "swin3d_global":  # 同family内のbaseline: Swin3D-Bのpooled特徴
        m = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1); m.head = nn.Identity(); return m, 1024
    if name == "mvit_v2_s":
        m = TV.mvit_v2_s(weights=TV.MViT_V2_S_Weights.KINETICS400_V1); m.head = nn.Identity(); return m, 768
    if name == "r2plus1d_18":
        m = TV.r2plus1d_18(weights=TV.R2Plus1D_18_Weights.KINETICS400_V1); m.fc = nn.Identity(); return m, 512
    if name == "s3d":
        m = TV.s3d(weights=TV.S3D_Weights.KINETICS400_V1); m.classifier = nn.Identity(); return m, 1024
    if name.startswith("hf:"):  # HuggingFace video model (VideoMAE/TimeSformer): mean-pool last_hidden_state
        from transformers import AutoModel
        return HFVideoEncoder(name[3:]), None
    raise ValueError(f"unknown backbone {name}")


class HFVideoEncoder(nn.Module):
    """HF video transformer (VideoMAE/TimeSformer) -> mean-pooled token feature [B,D]. clip [B,3,T,H,W]."""

    def __init__(self, hf_name):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(hf_name)
        self.out_dim = self.net.config.hidden_size

    def forward(self, clip):
        x = clip.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] as HF expects pixel_values
        out = self.net(pixel_values=x).last_hidden_state  # [B, tokens, D]
        return out.mean(dim=1)


class EncoderFusion(nn.Module):
    """Encoder-swap experiment: pluggable video backbone global feature + frozen target/instrument
    branches (presence/target-feat/target-logits). No mask-guided spatial pool (backbone-agnostic).
    Same downstream fusion so only the encoder differs from the Swin3D recipe."""

    def __init__(self, backbone, use_target=True, use_instrument=True, pairwise=False):
        super().__init__()
        self.enc, ed = make_encoder(backbone)
        if ed is None:
            ed = self.enc.out_dim
        self.use_target = use_target; self.use_instrument = use_instrument; self.pairwise = pairwise
        self.enc_proj = nn.Sequential(nn.Linear(ed, 512), nn.GELU())
        fus = 512
        if use_instrument:
            fus += NI  # presence (12)
        if use_target:
            self.tproj = nn.Sequential(nn.Linear(1024, 256), nn.GELU()); fus += 256 + NT
        self.trunk = nn.Sequential(nn.Linear(fus, 1024), nn.GELU(), nn.Dropout(0.3),
                                   nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3))
        self.hdim = 512
        self.head_ivt = nn.Linear(512, NIVT); self.head_i = nn.Linear(512, NI)
        self.head_v = nn.Linear(512, NV); self.head_t = nn.Linear(512, NT)
        if pairwise:
            self.head_it = nn.Linear(512, NIT); self.head_iv = nn.Linear(512, NIV)

    def forward(self, clip, tfeat, tlogit, pres, imask):
        g = self.enc_proj(self.enc(clip))
        parts = [g]
        if self.use_instrument:
            parts.append(pres)
        if self.use_target:
            parts += [self.tproj(tfeat), tlogit]
        z = self.trunk(torch.cat(parts, dim=1))
        oit = self.head_it(z) if self.pairwise else None
        oiv = self.head_iv(z) if self.pairwise else None
        return self.head_ivt(z), self.head_i(z), self.head_v(z), self.head_t(z), oit, oiv


class BipartiteGATv2(nn.Module):
    """二部グラフの GATv2 相互 message passing。A↔B（例: instrument↔target）。"""

    def __init__(self, d, nhead=4):
        super().__init__()
        self.nhead, self.dh = nhead, d // nhead
        self.wa = nn.Linear(d, d); self.wb = nn.Linear(d, d)
        self.att_a = nn.Parameter(torch.randn(nhead, 2 * self.dh) * 0.02)
        self.att_b = nn.Parameter(torch.randn(nhead, 2 * self.dh) * 0.02)

    def _attend(self, Q, K, att):                        # Q[b,nq,d] attends to K[b,nk,d] (GATv2)
        B, nq, d = Q.shape; nk = K.shape[1]; H, dh = self.nhead, self.dh
        Qh = Q.view(B, nq, H, dh); Kh = K.view(B, nk, H, dh)
        cat = torch.cat([Qh.unsqueeze(2).expand(B, nq, nk, H, dh),
                         Kh.unsqueeze(1).expand(B, nq, nk, H, dh)], dim=-1)  # [B,nq,nk,H,2dh]
        e = (torch.nn.functional.leaky_relu(cat, 0.2) * att.view(1, 1, 1, H, 2 * dh)).sum(-1)  # [B,nq,nk,H]
        alpha = e.softmax(dim=2)
        out = (alpha.unsqueeze(-1) * Kh.unsqueeze(1)).sum(2)  # [B,nq,H,dh]
        return out.reshape(B, nq, d)

    def forward(self, A, B):
        A2 = A + self._attend(self.wa(A), self.wb(B), self.att_a)
        B2 = B + self._attend(self.wb(B), self.wa(A), self.att_b)
        return A2, B2


class GATv2Pairwise(nn.Module):
    """B(SPIRIT PIC移植): z から i/v/t ノード特徴 → I-T / I-V 二部グラフ GATv2 → it/iv スコア。
    軽量 linear it/iv ヘッドを関係グラフ推論に格上げ。"""

    def __init__(self, hdim, d=128, nhead=4):
        super().__init__()
        self.d = d
        self.pi = nn.Linear(hdim, NI * d); self.pv = nn.Linear(hdim, NV * d); self.pt = nn.Linear(hdim, NT * d)
        self.gat_it = BipartiteGATv2(d, nhead); self.gat_iv = BipartiteGATv2(d, nhead)
        self.it_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.iv_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("it_i", torch.tensor(IT_I)); self.register_buffer("it_t", torch.tensor(IT_T))
        self.register_buffer("iv_i", torch.tensor(IV_I)); self.register_buffer("iv_v", torch.tensor(IV_V))

    def forward(self, z):
        B = z.shape[0]
        ni = self.pi(z).view(B, NI, self.d); nv = self.pv(z).view(B, NV, self.d); nt = self.pt(z).view(B, NT, self.d)
        ni_it, nt_it = self.gat_it(ni, nt)               # I-T グラフ message passing
        ni_iv, nv_iv = self.gat_iv(ni, nv)               # I-V グラフ
        oit = self.it_head(torch.cat([ni_it[:, self.it_i], nt_it[:, self.it_t]], -1)).squeeze(-1)  # [B,54]
        oiv = self.iv_head(torch.cat([ni_iv[:, self.iv_i], nv_iv[:, self.iv_v]], -1)).squeeze(-1)  # [B,39]
        return oit, oiv


class MaskGuidedSpatialAttn(nn.Module):
    """器具マスクを空間バイアスにした cross-attention（mask-guided pool の平均を attention に格上げ）。
    各器具クエリ q_i が keyframe 空間トークンに attention。logit に alpha*log(mask_i) を加算して
    器具iの領域を prior にしつつ周辺文脈(対象/動作)も見る。presence 重みで集約 → [B,out_dim]。
    class-query(A)が失敗したのと違い、マスクの強い空間prior で grounding される。
    """

    def __init__(self, c=1024, ni=NI, d=256, nhead=4, out_dim=1024):
        super().__init__()
        self.ni, self.d, self.nhead = ni, d, nhead
        self.q = nn.Embedding(ni, d); nn.init.normal_(self.q.weight, std=0.02)
        self.wk = nn.Linear(c, d); self.wv = nn.Linear(c, d)
        self.alpha = nn.Parameter(torch.tensor(2.0))     # マスクバイアス強度(学習可)
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, out_dim)

    def forward(self, key, imask, pres):                 # key[B,h,w,C], imask[B,ni,h,w], pres[B,ni] -> [B,out_dim]
        B, h, w, C = key.shape; HW = h * w
        kv = key.reshape(B, HW, C)
        k, v = self.wk(kv), self.wv(kv)                  # [B,HW,d]
        q = self.q.weight[None].expand(B, -1, -1)        # [B,ni,d]
        H, dh = self.nhead, self.d // self.nhead
        qh = q.reshape(B, self.ni, H, dh).transpose(1, 2)   # [B,H,ni,dh]
        kh = k.reshape(B, HW, H, dh).transpose(1, 2)
        vh = v.reshape(B, HW, H, dh).transpose(1, 2)
        logits = (qh @ kh.transpose(-1, -2)) / (dh ** 0.5)  # [B,H,ni,HW]
        bias = self.alpha * torch.log(imask.reshape(B, self.ni, HW).clamp(min=1e-4))  # [B,ni,HW]
        attn = (logits + bias[:, None]).softmax(dim=-1)
        out = (attn @ vh).transpose(1, 2).reshape(B, self.ni, self.d)  # [B,ni,d]
        out = self.ln(out)
        agg = (out * pres.unsqueeze(-1)).sum(1) / (pres.sum(1, keepdim=True) + 1e-4)  # presence重み集約 [B,d]
        return self.out(agg)                             # [B,out_dim]


class SwinFusion(nn.Module):
    def __init__(self, use_target=True, use_instrument=True, fusion="concat", pairwise=False,
                 class_query=False, temporal_attn=False, mask_attn=False, gatv2=False):
        super().__init__()
        self.use_target = use_target; self.use_instrument = use_instrument; self.fusion = fusion
        self.pairwise = pairwise; self.class_query = class_query; self.temporal_attn = temporal_attn
        self.mask_attn = mask_attn; self.gatv2 = gatv2 and pairwise
        m = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
        self.patch_embed, self.pos_drop, self.features, self.norm = m.patch_embed, m.pos_drop, m.features, m.norm
        C = 1024
        if temporal_attn:  # A: 時間 mean-pool を時間 attention 集約に置換
            self.tpool = TemporalAttnPool(C)
        if mask_attn:  # 器具マスク誘導の空間 cross-attention（mask-pool 平均の格上げ）
            self.msa = MaskGuidedSpatialAttn(c=C, out_dim=C)
        if fusion == "xattn":
            # binding transformer: tokens = [global-clip, target, per-instrument region] interact via
            # self-attention so the head sees which instrument relates to which target/action (i,v,t binding).
            D = 512
            self.g_proj = nn.Linear(C, D)
            self.inst_proj = nn.Linear(C, D)              # instrument-region (mask-pooled keyframe) token
            self.tgt_proj = nn.Linear(1024, D)
            self.tlogit_proj = nn.Linear(NT, D)
            self.pres_proj = nn.Linear(1, D)              # per-instrument presence/confidence -> added to its token
            self.inst_id_emb = nn.Embedding(NI, D)        # instrument identity (which of the 12 the token is)
            self.type_emb = nn.Parameter(torch.zeros(3, D))  # 0=global, 1=target, 2=instrument
            enc = nn.TransformerEncoderLayer(D, nhead=8, dim_feedforward=4 * D, dropout=0.3,
                                             batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(enc, num_layers=2)
            self.hdim = D
        else:
            fus = C                                    # Swin3D global (always)
            if use_instrument:
                fus += C + NI                          # mask-guided + presence
            if use_target:
                self.tproj = nn.Sequential(nn.Linear(1024, 256), nn.GELU())
                fus += 256 + NT                        # target_proj + target_logits
            self.trunk = nn.Sequential(nn.Linear(fus, 1024), nn.GELU(), nn.Dropout(0.3),
                                       nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3))
            self.hdim = 512
        self.head_ivt = nn.Linear(self.hdim, NIVT); self.head_i = nn.Linear(self.hdim, NI)
        self.head_v = nn.Linear(self.hdim, NV); self.head_t = nn.Linear(self.hdim, NT)
        if pairwise:  # SPIRIT の I-T / I-V 補助ヘッド（密な監督 + zero-train triplet 合成用）
            if self.gatv2:  # B: 関係グラフ推論版（linear it/iv を GATv2 に格上げ）
                self.gat = GATv2Pairwise(self.hdim)
            else:
                self.head_it = nn.Linear(self.hdim, NIT); self.head_iv = nn.Linear(self.hdim, NIV)
        if class_query:  # A: ivt の class-query head（keyframe空間トークンへ attention）
            self.cq_ivt = ClassQueryHead(NIVT, c_in=1024)

    def _heads(self, z):
        if self.pairwise and self.gatv2:
            oit, oiv = self.gat(z)
        elif self.pairwise:
            oit, oiv = self.head_it(z), self.head_iv(z)
        else:
            oit = oiv = None
        return self.head_ivt(z), self.head_i(z), self.head_v(z), self.head_t(z), oit, oiv

    def forward(self, clip, tfeat, tlogit, pres, imask):
        x = self.norm(self.features(self.pos_drop(self.patch_embed(clip))))  # [B,T',H',W',C]
        g = self.tpool(x) if self.temporal_attn else x.mean(dim=(1, 2, 3))   # global [B,C] (時間attn or mean)
        if self.fusion == "xattn":
            key = x[:, -1]                                                   # keyframe slice [B,h,w,C]
            B, h, w, C = key.shape
            im = torch.nn.functional.interpolate(imask, size=(h, w), mode="area")  # [B,12,h,w]
            imn = im / (im.sum(dim=(2, 3), keepdim=True) + 1e-6)            # per-instrument spatial weights
            inst = torch.einsum("bnhw,bhwc->bnc", imn, key)                 # mask-pooled region feats [B,12,C]
            it = self.inst_proj(inst) + self.inst_id_emb.weight[None] + self.pres_proj(pres.unsqueeze(-1))
            it = it + self.type_emb[2]
            gt = (self.g_proj(g) + self.type_emb[0]).unsqueeze(1)           # [B,1,D]
            tt = (self.tgt_proj(tfeat) + self.tlogit_proj(tlogit) + self.type_emb[1]).unsqueeze(1)
            tokens = torch.cat([gt, tt, it], dim=1)                         # [B, 14, D]
            z = self.encoder(tokens)[:, 0]                                  # global-token readout
            return self._heads(z)
        parts = [g]
        if self.use_instrument:
            key = x[:, -1]                                                   # keyframe slice [B,H',W',C]
            if self.mask_attn:  # マスク誘導 空間 cross-attention（平均pool の格上げ）
                im = torch.nn.functional.interpolate(imask, size=(key.shape[1], key.shape[2]), mode="area")
                m = self.msa(key, im, pres)                                  # [B,C]
            else:
                fg = imask.max(dim=1).values.unsqueeze(1)                    # [B,1,28,28] foreground
                w = torch.nn.functional.interpolate(fg, size=(key.shape[1], key.shape[2]), mode="area")[:, 0]
                w = w / (w.sum(dim=(1, 2), keepdim=True) + 1e-6)
                m = (key * w.unsqueeze(-1)).sum(dim=(1, 2))                  # mask-guided mean [B,C]
            parts += [m, pres]
        if self.use_target:
            parts += [self.tproj(tfeat), tlogit]
        z = self.trunk(torch.cat(parts, dim=1))
        oivt, oi, ov, ot, oit, oiv = self._heads(z)
        if self.class_query:  # A: keyframe 空間トークンへの class-query を ivt に加算
            key = x[:, -1]; B, h, w, C = key.shape
            oivt = oivt + self.cq_ivt(key.reshape(B, h * w, C))
        return oivt, oi, ov, ot, oit, oiv


def ivt_to_comp(ivt_scores):
    """derive i/v/t scores from ivt via component_maps max-pool. ivt_scores [N,85]."""
    N = ivt_scores.shape[0]
    i = np.zeros((N, NI)); v = np.zeros((N, NV)); t = np.zeros((N, NT))
    for k in range(NIVT):
        _, ci, cv, ct = CM[k, 0], CM[k, 1], CM[k, 2], CM[k, 3]
        i[:, ci] = np.maximum(i[:, ci], ivt_scores[:, k])
        v[:, cv] = np.maximum(v[:, cv], ivt_scores[:, k])
        t[:, ct] = np.maximum(t[:, ct], ivt_scores[:, k])
    return i, v, t


def videowise_map(scores, labels, vids, ncls):
    byv = defaultdict(list)
    for i, vd in enumerate(vids):
        byv[vd].append(i)
    ms = []
    for vd, idx in byv.items():
        y, p = labels[idx], scores[idx]
        aps = [average_precision_score(y[:, c], p[:, c]) for c in range(ncls) if y[:, c].sum() > 0]
        if aps:
            ms.append(np.mean(aps))
    return float(np.mean(ms)) if ms else 0.0


@torch.no_grad()
def evaluate(model, loader, items, device, trpos=None):
    model.eval(); S = []; SIT = []; SIV = []
    pairwise = getattr(model, "pairwise", False)
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            oivt, oi, ov, ot, oit, oiv = model(b["clip"].to(device), b["tfeat"].to(device),
                                               b["tlogit"].to(device), b["pres"].to(device), b["imask"].to(device))
        S.append(torch.sigmoid(oivt).float().cpu().numpy())
        if pairwise:
            SIT.append(torch.sigmoid(oit).float().cpu().numpy()); SIV.append(torch.sigmoid(oiv).float().cpu().numpy())
    P = np.concatenate(S)
    Yivt = np.stack([it[3] for it in items]); vids = [it[0] for it in items]
    pi, pv, pt = ivt_to_comp(P)
    Yi = np.stack([it[4] for it in items]); Yv = np.stack([it[5] for it in items]); Yt = np.stack([it[6] for it in items])
    out = {"ivt": videowise_map(P, Yivt, vids, NIVT), "i": videowise_map(pi, Yi, vids, NI),
           "v": videowise_map(pv, Yv, vids, NV), "t": videowise_map(pt, Yt, vids, NT)}
    if pairwise:  # 診断: pairwise 直接合成(it×iv) と、direct との support-gated blend の ivt mAP
        Pit, Piv = np.concatenate(SIT), np.concatenate(SIV)
        Ppw = Pit[:, IT_MAP] * Piv[:, IV_MAP]                          # composed ivt [N,85]
        out["ivt_pw"] = videowise_map(Ppw, Yivt, vids, NIVT)
        if trpos is not None:
            w = np.array([1.0 if t == 0 else 0.5 if t < 50 else 0.3 if t < 500 else 0.1 for t in trpos], np.float32)
            Pblend = (1 - w)[None] * P + w[None] * Ppw
            out["ivt_blend"] = videowise_map(Pblend, Yivt, vids, NIVT)
    return out


def pos_weight(items, key_idx, ncls, cap):
    Y = np.stack([it[key_idx] for it in items]); pos = Y.sum(0)
    return torch.tensor(np.clip(np.where(pos > 0, (len(Y) - pos) / np.maximum(pos, 1), 1), 0, cap), dtype=torch.float32)


class FocalBCE(nn.Module):
    """Multi-label focal BCE (imbalance): down-weights easy examples via (1-p_t)^gamma, keeps pos_weight.
    gamma=0 reduces to plain BCEWithLogitsLoss(pos_weight)."""
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__(); self.register_buffer("pw", pos_weight); self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pw, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)          # prob of the true class
        return (ce * (1 - p_t).clamp(min=1e-6) ** self.gamma).mean()


def make_loss(cfg, pw, device):
    """BCE (default) or focal, per config train.loss / train.focal_gamma."""
    if cfg["train"].get("loss", "bce") == "focal":
        return FocalBCE(pw, gamma=cfg["train"].get("focal_gamma", 2.0)).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config_verb.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump_oof", default="")  # アンサンブル用: best_model の val OOF スコアを npz 保存して終了
    ap.add_argument("--dump_split", default="val")  # dump 対象: val / train / all（蒸留教師soft用）
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); seed_all(cfg["experiment"]["seed"])
    global CACHE
    CACHE = cfg["data"].get("cache_dir", CACHE)
    outdir = os.path.join(HERE, "..", "results", cfg["experiment"]["name"], f"fold{cfg['data']['val_fold']}")
    os.makedirs(outdir, exist_ok=True); shutil.copy(args.config, os.path.join(outdir, "config.yaml"))
    lg = setup_logger(outdir); device = "cuda"
    wb = init_wandb(cfg, lg)
    v2f = load_folds(cfg["data"]["folds_csv"])
    tr = build_index(cfg, v2f, False); va = build_index(cfg, v2f, True)
    trpos = np.stack([it[3] for it in tr]).sum(0) if tr else np.zeros(NIVT)  # [85] train triplet support (blend gate)
    if args.limit:
        tr = tr[:args.limit]; va = va[:max(50, args.limit // 4)]
    lg.info(f"train={len(tr)} val={len(va)} val_fold={cfg['data']['val_fold']}")

    dcfg = cfg.get("distill", {})  # D: 蒸留. teacher_soft(npz) の soft ivt を train frame に付与
    teacher_lut = None
    if dcfg.get("teacher_soft"):
        td = np.load(dcfg["teacher_soft"], allow_pickle=True)
        _tsc = np.asarray(td["scores"], dtype=np.float32); _tv = td["vids"]; _tf = td["fids"]  # 実体化(NpzFile再解凍のO(n^2)回避)
        teacher_lut = {(str(v), int(f)): _tsc[i]
                       for i, (v, f) in enumerate(zip(_tv, _tf))}
        lg.info(f"distill: teacher_soft={dcfg['teacher_soft']} n={len(teacher_lut)} weight={dcfg.get('weight', 1.0)}")
    tl = DataLoader(ClipDS(tr, cfg, True, teacher_lut=teacher_lut), batch_size=cfg["train"]["bs"], shuffle=True,
                    num_workers=cfg["train"]["workers"], pin_memory=True, drop_last=True)
    vl = DataLoader(ClipDS(va, cfg, False), batch_size=cfg["train"]["bs"], shuffle=False,
                    num_workers=cfg["train"]["workers"], pin_memory=True)
    mcfg = cfg.get("model", {})
    pairwise = mcfg.get("pairwise_heads", False)  # SPIRIT I-T/I-V 補助ヘッド（config-gated, 既定OFF）
    classq = mcfg.get("class_query", False)  # A: text-conditioned class-query head (config-gated)
    tattn = mcfg.get("temporal_attn", False)  # A: 時間 attention 集約 (config-gated)
    maskattn = mcfg.get("mask_attn", False)  # マスク誘導 空間 cross-attention (config-gated)
    gv2 = mcfg.get("gatv2", False)  # B: GATv2 pairwise 関係グラフ推論 (config-gated)
    backbone = mcfg.get("backbone", "swin3d")  # 動画encoder差し替え: swin3d / mvit_v2_s / r2plus1d_18 / s3d / hf:...
    if backbone != "swin3d":
        model = EncoderFusion(backbone, use_target=mcfg.get("use_target", True),
                              use_instrument=mcfg.get("use_instrument", True), pairwise=pairwise).to(device)
        lg.info(f"model: EncoderFusion backbone={backbone} pairwise={pairwise}")
    else:
        model = SwinFusion(use_target=mcfg.get("use_target", True),
                           use_instrument=mcfg.get("use_instrument", True),
                           fusion=mcfg.get("fusion", "concat"), pairwise=pairwise, class_query=classq, temporal_attn=tattn, mask_attn=maskattn, gatv2=gv2).to(device)
    lg.info(f"model: use_target={mcfg.get('use_target', True)} use_instrument={mcfg.get('use_instrument', True)} "
            f"fusion={mcfg.get('fusion', 'concat')} pairwise_heads={pairwise}")
    if mcfg.get("swin_init"):  # load CholecT50-pretrained Swin3D backbone (verb / verb+tool)
        ck = torch.load(mcfg["swin_init"], map_location="cpu")["backbone"]
        model.patch_embed.load_state_dict(ck["patch_embed"]); model.pos_drop.load_state_dict(ck["pos_drop"])
        model.features.load_state_dict(ck["features"]); model.norm.load_state_dict(ck["norm"])
        lg.info(f"Swin3D init <- {mcfg['swin_init']} (heads={ck.get('heads', '?') if isinstance(ck, dict) else '?'})")
    if args.dump_oof:  # OOF/教師soft dump: best_model で推論 → sigmoid ivt スコアを保存
        bm = os.path.join(outdir, "best_model.pth")
        model.load_state_dict(torch.load(bm, map_location=device)["model"]); model.eval()
        items = {"val": va, "train": tr, "all": va + tr}[args.dump_split]
        dl = DataLoader(ClipDS(items, cfg, False), batch_size=cfg["train"]["bs"], shuffle=False,
                        num_workers=cfg["train"]["workers"], pin_memory=True)
        S = []
        with torch.no_grad():
            for b in dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    oivt = model(b["clip"].to(device), b["tfeat"].to(device), b["tlogit"].to(device),
                                 b["pres"].to(device), b["imask"].to(device))[0]
                S.append(torch.sigmoid(oivt).float().cpu().numpy())
        P = np.concatenate(S)
        np.savez(args.dump_oof, scores=P, labels=np.stack([it[3] for it in items]),
                 vids=np.array([it[0] for it in items]), fids=np.array([it[1] for it in items]))
        lg.info(f"dumped {args.dump_split} -> {args.dump_oof} scores={P.shape}"); return

    pw_spec = [("ivt", 3, NIVT), ("i", 4, NI), ("v", 5, NV), ("t", 6, NT)]
    if pairwise:
        pw_spec += [("it", 7, NIT), ("iv", 8, NIV)]  # items index 7=ity, 8=ivy
    pw = {k: pos_weight(tr, ix, n, cfg["train"]["pos_weight_cap"]).to(device)
          for k, ix, n in pw_spec}
    lf = {k: make_loss(cfg, pw[k], device) for k in pw}
    # C: ヘッド別損失重み（SPIRIT流 triplet重視）。config train.loss_weights で上書き、既定は全1.0
    lw = {"ivt": 1.0, "i": 1.0, "v": 1.0, "t": 1.0, "it": 1.0, "iv": 1.0}
    lw.update(cfg["train"].get("loss_weights", {}))
    lg.info(f"loss_weights={lw}")

    bb = list(self_params(model, True)); rest = list(self_params(model, False))
    groups = [{"params": bb, "lr": cfg["train"]["lr"] * cfg["train"]["backbone_lr_mult"]},
              {"params": rest, "lr": cfg["train"]["lr"]}]
    epochs = cfg["train"]["epochs"]; steps = len(tl) * epochs; warm = int(steps * cfg["train"]["warmup_ratio"])
    opt_name = cfg["train"].get("optimizer", "adamw").lower()
    sf = opt_name in ("radam_sf", "radam_schedulefree", "schedulefree")
    if sf:  # RAdamScheduleFree: built-in warmup + Polyak averaging -> no external LR schedule
        from schedulefree import RAdamScheduleFree
        opt = RAdamScheduleFree(groups, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        sched = None
        lg.info(f"optimizer=RAdamScheduleFree lr={cfg['train']['lr']} (RAdam built-in warmup, no LR schedule)")
    else:
        opt = torch.optim.AdamW(groups, weight_decay=cfg["train"]["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(warm, 1) if s < warm
                                                  else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
        lg.info(f"optimizer=AdamW lr={cfg['train']['lr']} cosine warmup_ratio={cfg['train']['warmup_ratio']}")
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, cfg["train"]["ema_decay"]) if (cfg["train"].get("ema_decay") and not sf) else None
    if ema:
        lg.info(f"EMA enabled decay={cfg['train']['ema_decay']}")
    elif cfg["train"].get("ema_decay") and sf:
        lg.info("EMA disabled: RAdamScheduleFree already Polyak-averages (redundant)")
    start, best, hist = 0, -1.0, []
    last = os.path.join(outdir, "last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start = ck["epoch"] + 1
        best = ck.get("best", -1); hist = ck.get("hist", [])
        if ema and ck.get("ema"):
            ema.load_state_dict({k: v.to(device) for k, v in ck["ema"].items()})
        lg.info(f"resumed ep{start} best={best:.4f}")

    fcm = cfg["train"].get("framecutmix")  # aug6: temporal splice of 2 clips + ratio-mixed labels
    for ep in range(start, epochs):
        model.train()
        if sf:
            opt.train()  # schedule-free: use training (interpolated) weights during optimisation
        t0 = time.time(); run = 0.0
        for bi, b in enumerate(tl):
            lkeys = ("ivt", "i", "v", "t", "it", "iv") if pairwise else ("ivt", "i", "v", "t")
            clip = b["clip"]; yv = {k: b[k] for k in lkeys}
            if fcm and random.random() < fcm.get("p", 0.5):  # replace first c frames (keep last=keyframe A)
                B_, _, T_ = clip.shape[:3]; perm = torch.randperm(B_)
                c = random.randint(1, T_ - 1); clip = clip.clone(); clip[:, :, :c] = clip[perm][:, :, :c]
                lam = (T_ - c) / T_  # fraction of frames (incl. keyframe) from the original clip
                yv = {k: lam * b[k] + (1 - lam) * b[k][perm] for k in yv}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                oivt, oi, ov, ot, oit, oiv = model(clip.to(device), b["tfeat"].to(device), b["tlogit"].to(device),
                                                   b["pres"].to(device), b["imask"].to(device))
                loss = (lw["ivt"] * lf["ivt"](oivt, yv["ivt"].to(device)) + lw["i"] * lf["i"](oi, yv["i"].to(device))
                        + lw["v"] * lf["v"](ov, yv["v"].to(device)) + lw["t"] * lf["t"](ot, yv["t"].to(device)))
                if pairwise:
                    loss = loss + lw["it"] * lf["it"](oit, yv["it"].to(device)) + lw["iv"] * lf["iv"](oiv, yv["iv"].to(device))
                if "teacher" in b and dcfg.get("weight", 0) > 0 and torch.isfinite(oivt).all():  # D: 蒸留(SPIRIT式 TS_n重み)
                    teach = b["teacher"].to(device).clamp(1e-4, 1 - 1e-4)          # [B,85] 教師soft
                    ht = b["has_teach"].to(device); gt = b["ivt"].to(device)
                    with torch.no_grad():  # w_n = TS·exp(−TG/β): 「教師自信あり&生徒ズレ」重視 + アノテ抜け弱め
                        sivt = torch.sigmoid(oivt.float()).clamp(1e-4, 1 - 1e-4)   # 生徒 prob(fp32安定化)
                        ts = -(teach * torch.log(sivt) + (1 - teach) * torch.log(1 - sivt)).mean(1)  # 教師↔生徒 不一致
                        tg = -(gt * torch.log(teach) + (1 - gt) * torch.log(1 - teach)).mean(1)       # 教師↔GT 不一致
                        wn = (ts * torch.exp(-tg / dcfg.get("beta", 1.0)) * ht).clamp(0, 3.0)
                    dw = dcfg["weight"] * min(1.0, (ep + 1) / dcfg.get("warmup_ep", 2))  # weight warmup
                    dloss = (torch.nn.functional.binary_cross_entropy_with_logits(oivt, teach, reduction="none").mean(1) * wn).sum() / (wn.sum() + 1e-6)
                    if torch.isfinite(dloss):
                        loss = loss + lw["ivt"] * dw * dloss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched:
                sched.step()
            if ema:
                ema.update(model)
            run += loss.item()
            if bi % 50 == 0:
                lr_now = sched.get_last_lr()[1] if sched else cfg["train"]["lr"]
                lg.debug(f"ep{ep} {bi}/{len(tl)} loss={loss.item():.3f} lr={lr_now:.2e}")
        if sf:
            opt.eval()  # swap to Polyak-averaged weights for evaluation + deploy checkpoint
        mp = evaluate(model, vl, va, device, trpos)
        cand_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}  # (SF: averaged; AdamW: raw)
        if sf:
            opt.train()  # restore training weights before saving last.pth (resume-correct)
        logline = (f"[ep{ep}] loss={run/len(tl):.3f} | raw ivt={mp['ivt']:.4f} "
                   f"i={mp['i']:.4f} v={mp['v']:.4f} t={mp['t']:.4f}")
        if "ivt_pw" in mp:  # pairwise 合成の効果を毎epoch可視化
            logline += f" | pw={mp['ivt_pw']:.4f} blend={mp.get('ivt_blend', float('nan')):.4f}"
        rec = {"epoch": ep, "train_loss": run / len(tl), **{f"vmAP_{k}": mp[k] for k in mp}}
        cand, cand_mp = mp["ivt"], mp
        if ema:  # evaluate the EMA weights on a throwaway copy, keep whichever (raw|ema) is better
            if backbone != "swin3d":
                ema_model = EncoderFusion(backbone, use_target=mcfg.get("use_target", True),
                                          use_instrument=mcfg.get("use_instrument", True), pairwise=pairwise).to(device)
            else:
                ema_model = SwinFusion(use_target=mcfg.get("use_target", True),
                                       use_instrument=mcfg.get("use_instrument", True),
                                       fusion=mcfg.get("fusion", "concat"), pairwise=pairwise, class_query=classq, temporal_attn=tattn, mask_attn=maskattn, gatv2=gv2).to(device)
            ema.copy_to(ema_model)
            mpe = evaluate(ema_model, vl, va, device)
            rec.update({f"ema_{k}": mpe[k] for k in mpe})
            logline += f" || ema ivt={mpe['ivt']:.4f} i={mpe['i']:.4f} v={mpe['v']:.4f} t={mpe['t']:.4f}"
            if mpe["ivt"] > cand:
                cand, cand_sd, cand_mp = mpe["ivt"], ema_model.state_dict(), mpe
            del ema_model
        hist.append(rec)
        lg.info(logline + f" ({time.time()-t0:.0f}s)")
        if wb:
            wb.log(rec, step=ep)
        json.dump(hist, open(os.path.join(outdir, "training_log.json"), "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None,
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist,
                    "ema": ema.state_dict() if ema else None}, last)
        if cand > best:
            best = cand
            torch.save({"model": cand_sd, "epoch": ep, "vmAP": cand_mp}, os.path.join(outdir, "best_model.pth"))
            lg.info(f"  new best ivt vmAP={best:.4f} -> best_model.pth")
    lg.info(f"done. best ivt vmAP={best:.4f}")
    if wb:
        wb.summary["best_ivt"] = best; wb.finish()


def self_params(model, backbone):
    """yield swin3d backbone params (backbone=True) or the rest (fusion/heads)."""
    bb_prefixes = ("patch_embed", "pos_drop", "features", "norm")
    for n, p in model.named_parameters():
        is_bb = n.split(".")[0] in bb_prefixes
        if is_bb == backbone:
            yield p


if __name__ == "__main__":
    main()
