"""SPIRIT fine-tune 学習（生frame, backbone上位block学習）。

train_spirit.py の凍結キャッシュ版に対し、raw frame を T=8 clip で読み ViT-L に通して学習。
- model: SpiritStageA_FT（DINOv3-L trunk[下位凍結] + SpiritStageA head）
- param groups: head_lr / backbone_lr（SPIRIT: 1e-4 / 2e-5）
- autocast: bf16(A100) or fp16。RandAug は当面軽量(clip一貫 hflip+color)。
- frame_stride で学習frameを間引ける（GPU1検証用に高速化）。

使い方: GPU=N python src/train_spirit_ft.py --config config_ftA_f0.yaml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import random
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

HERE = os.path.dirname(__file__)
REPO = os.environ.get("MB_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LBL_DIR = os.environ.get("MB_LBL_DIR", os.path.join(REPO, "data/multibypasst40_challenge_trainval/label_files_challenge"))
VID_DIR = os.environ.get("MB_VID_DIR", os.path.join(REPO, "data/multibypasst40_challenge_trainval/videos"))
DINOV3_REPO = os.path.join(REPO, "reference/dinov3")
VITL_WEIGHTS = os.environ.get("VITL_WEIGHTS", os.path.join(os.environ.get("WORK", "work"), "weights", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
sys.path.insert(0, os.path.join(REPO, "reference/starter_kit"))
from utils.triplet_mappings import triplet_maps  # noqa: E402
sys.path.insert(0, HERE)
from model_spirit_ft import SpiritStageA_FT, SpiritFull_FT, NI, NV, NT, NIVT  # noqa: E402

MB = triplet_maps["multibypasst40"]; CM = np.array(MB["component_maps"])
NIT, NIV = NI * NT, NI * NV  # 180, 156 dense pairwise grids (SPIRIT full)
IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def setup_logger(outdir):
    lg = logging.getLogger("spiritft"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    fh = logging.FileHandler(os.path.join(outdir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def build_index(cfg, v2f, is_val):
    vf = cfg["data"]["val_fold"]; stride = cfg["data"].get("frame_stride", 1); items = []
    cache_dir = cfg["data"].get("frozen_cache", None)  # #2/#3: 凍結cache(imask/target)注入。未指定なら従来通り
    for lf in sorted(glob.glob(os.path.join(LBL_DIR, "*.json"))):
        vid = os.path.splitext(os.path.basename(lf))[0]
        if vid not in v2f or (v2f[vid] == vf) != is_val:
            continue
        fid2row = {}
        if cache_dir:
            cpath = os.path.join(cache_dir, f"{vid}.npz")
            if os.path.exists(cpath):
                fid2row = {int(f): r for r, f in enumerate(np.load(cpath)["fids"])}
        d = json.load(open(lf)); byimg = defaultdict(list)
        for a in d["annotations"]:
            byimg[a["image_id"]].append(a)
        maxfid = max(int(im["id"]) for im in d["images"])
        for im in d["images"]:
            fid = int(im["id"])
            if (not is_val) and stride > 1 and (fid % stride):   # trainのみ間引き, valは全frame
                continue
            if cache_dir and fid not in fid2row:                 # cache注入時は cache にある frame のみ
                continue
            anns = byimg.get(fid, [])
            if not anns and cfg["data"].get("drop_empty", True):
                continue
            crow = fid2row.get(fid, -1)
            ivt = np.zeros(NIVT, np.float32); iy = np.zeros(NI, np.float32)
            vy = np.zeros(NV, np.float32); ty = np.zeros(NT, np.float32)
            ity = np.zeros(NIT, np.float32); ivy = np.zeros(NIV, np.float32)  # dense it/iv (full model)
            for a in anns:
                ivt[a["category_id"]] = 1.0; iy[a["instrument_id"]] = 1.0
                vy[a["verb_id"]] = 1.0; ty[a["target_id"]] = 1.0
                ity[a["instrument_id"] * NT + a["target_id"]] = 1.0
                ivy[a["instrument_id"] * NV + a["verb_id"]] = 1.0
            items.append((vid, fid, maxfid, ivt, iy, vy, ty, ity, ivy, crow))
    return items


class ClipDS(Dataset):
    def __init__(self, items, cfg, train, teacher_lut=None):
        self.items = items; self.T = cfg["data"]["clip_len"]; self.train = train
        self.teacher_lut = teacher_lut  # D: 蒸留教師 soft ivt {(vid,fid): [85]}
        a = cfg.get("aug", {})
        self.hflip = a.get("hflip", 0.5) if train else 0.0
        self.color = a.get("color", 0.2) if train else 0.0        # brightness/contrast 幅
        self.rrc = a.get("rrc", None) if train else None          # {p, scale:[lo,hi]} clip一貫RandomResizedCrop
        self.sat = a.get("saturation", 0.0) if train else 0.0     # 彩度/色相ジッタ幅
        self.fco = a.get("frame_cutout", None) if train else None  # 時間aug: {p, max_drop} 非keyframeを欠落
        self.dyna = a.get("dyna", None) if train else None        # 時間aug: {mag, n_freq} 滑らかper-frame photometric
        # Swin発の frame-sampling を SPIRIT へ: 指数offsets で長期文脈(~3分)を T フレームに圧縮。
        # dense causal(T秒)より広い phase 文脈を時間attention(TCA)で活かす狙い。len==clip_len。
        self.offsets = cfg["data"].get("frame_offsets", None)
        self.cache_dir = cfg["data"].get("frozen_cache", None)  # #2/#3: 凍結cache注入
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _read_raw(self, vid, fid):
        im = cv2.imread(os.path.join(VID_DIR, vid, f"{max(fid,0):06d}.jpg"))
        im = np.zeros((IMG, IMG, 3), np.uint8) if im is None else cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB)
        return im.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        vid, kf, maxfid, ivt, iy, vy, ty, ity, ivy, crow = self.items[idx]
        if self.offsets is not None:                                   # 指数サンプリング(長期文脈, keyframe=末尾)
            fids = [max(kf + off, 0) for off in self.offsets]
        else:
            fids = [max(kf - self.T + 1 + j, 0) for j in range(self.T)]  # causal dense stride1
        clip = np.stack([self._read_raw(vid, f) for f in fids])        # [T,H,W,3] in [0,1]
        # clip一貫 RandomResizedCrop（全frame同一box, 論文RandAugment相当のgeometric）
        if self.rrc is not None and random.random() < self.rrc.get("p", 0.5):
            lo, hi = self.rrc.get("scale", [0.7, 1.0]); side = math.sqrt(random.uniform(lo, hi))
            x0 = random.uniform(0, 1 - side); y0 = random.uniform(0, 1 - side)
            x1, y1 = int((x0 + side) * IMG), int((y0 + side) * IMG); x0i, y0i = int(x0 * IMG), int(y0 * IMG)
            clip = np.stack([cv2.resize(clip[i, y0i:y1, x0i:x1], (IMG, IMG)) for i in range(self.T)])
        if self.train and self.color > 0:                              # clip一貫 brightness/contrast
            br = 1 + random.uniform(-self.color, self.color); ct = 1 + random.uniform(-self.color, self.color)
            g = float(clip.mean()); clip = np.clip((clip * br - g) * ct + g, 0, 1)
        if self.train and self.sat > 0:                                # clip一貫 saturation
            sf = 1 + random.uniform(-self.sat, self.sat)
            gray = clip.mean(-1, keepdims=True); clip = np.clip(gray + (clip - gray) * sf, 0, 1)
        if self.dyna is not None:                                      # 時間aug: 滑らかper-frame photometric (DynaAugment)
            mag = self.dyna.get("mag", 0.3); nf = self.dyna.get("n_freq", 3); tt = np.linspace(0, 1, self.T)
            cb = np.zeros(self.T, np.float32); cc = np.zeros(self.T, np.float32)
            for _ in range(nf):
                cb += random.uniform(0, 1) * np.sin(2 * np.pi * random.uniform(0.5, 3) * tt + random.uniform(0, 6.28))
                cc += random.uniform(0, 1) * np.sin(2 * np.pi * random.uniform(0.5, 3) * tt + random.uniform(0, 6.28))
            cb = cb / (np.abs(cb).max() + 1e-6) * mag; cc = cc / (np.abs(cc).max() + 1e-6) * mag; g = float(clip.mean())
            for f in range(self.T):
                clip[f] = np.clip((clip[f] * (1 + cb[f]) - g) * (1 + cc[f]) + g, 0, 1)
        clip = (clip - MEAN) / STD
        if self.fco is not None and random.random() < self.fco.get("p", 0.5):  # 時間aug: 非keyframe(末尾以外)を欠落
            n = random.randint(1, self.fco.get("max_drop", 2))
            if self.T - 1 - n > 0:
                st = random.randint(0, self.T - 1 - n); clip[st:st + n] = 0.0
        if self.train and random.random() < self.hflip:
            clip = clip[:, :, ::-1, :].copy()
        clip = torch.from_numpy(clip.transpose(0, 3, 1, 2).copy())     # [T,3,H,W]
        out = {"frames": clip, "ivt": torch.from_numpy(ivt), "i": torch.from_numpy(iy),
               "v": torch.from_numpy(vy), "t": torch.from_numpy(ty),
               "it": torch.from_numpy(ity), "iv": torch.from_numpy(ivy)}
        if self.cache_dir is not None and crow >= 0:                   # #2/#3: keyframe の凍結cache注入
            if vid not in self._cache:
                self._cache[vid] = np.load(os.path.join(self.cache_dir, f"{vid}.npz"))
            c = self._cache[vid]
            out["imask"] = torch.from_numpy(c["inst_mask"][crow].astype(np.float32))       # [12,28,28] #2
            out["tfeat"] = torch.from_numpy(c["target_feat"][crow].astype(np.float32))     # [1024] #3
            out["tlogit"] = torch.from_numpy(c["target_logits"][crow].astype(np.float32))  # [15] #3
            out["pres"] = torch.from_numpy(c["inst_presence"][crow].astype(np.float32))    # [12] #3
        if self.teacher_lut is not None:  # D: 蒸留教師 soft ivt（無いframeは zeros+mask）
            out["teacher"] = torch.from_numpy(self.teacher_lut.get((vid, kf), np.zeros(NIVT, np.float32)))
            out["has_teach"] = torch.tensor(1.0 if (vid, kf) in self.teacher_lut else 0.0)
        return out


def pos_weight(items, ix, ncls, cap):
    Y = np.stack([it[ix] for it in items]); pos = Y.sum(0); neg = len(Y) - pos
    return torch.tensor(np.where(pos > 0, np.minimum(neg / np.maximum(pos, 1), cap), 1.0), dtype=torch.float32)


def ivt_to_comp(s):
    N = s.shape[0]; i = np.zeros((N, NI)); v = np.zeros((N, NV)); t = np.zeros((N, NT))
    for k in range(NIVT):
        i[:, CM[k, 1]] = np.maximum(i[:, CM[k, 1]], s[:, k])
        v[:, CM[k, 2]] = np.maximum(v[:, CM[k, 2]], s[:, k])
        t[:, CM[k, 3]] = np.maximum(t[:, CM[k, 3]], s[:, k])
    return i, v, t


def videowise_map(sc, lb, vids, nc):
    byv = defaultdict(list)
    for i, vd in enumerate(vids):
        byv[vd].append(i)
    ms = []
    for _, idx in byv.items():
        y, p = lb[idx], sc[idx]
        aps = [average_precision_score(y[:, c], p[:, c]) for c in range(nc) if y[:, c].sum() > 0]
        if aps:
            ms.append(np.mean(aps))
    return float(np.mean(ms)) if ms else 0.0


def _extra(b, device):  # #2/#3: 凍結cache 入力を batch から取り出す（無ければ None）
    return dict(imask=b["imask"].to(device) if "imask" in b else None,
                tfeat=b["tfeat"].to(device) if "tfeat" in b else None,
                pres=b["pres"].to(device) if "pres" in b else None)


@torch.no_grad()
def evaluate(model, loader, items, device, amp_dtype):
    model.eval(); S = []
    for b in loader:
        with torch.autocast("cuda", dtype=amp_dtype):
            o = model(b["frames"].to(device), **_extra(b, device))
        S.append(np.nan_to_num(torch.sigmoid(o["ivt"]).float().cpu().numpy()))
    P = np.concatenate(S); Yivt = np.stack([it[3] for it in items]); vids = [it[0] for it in items]
    pi, pv, pt = ivt_to_comp(P)
    Yi = np.stack([it[4] for it in items]); Yv = np.stack([it[5] for it in items]); Yt = np.stack([it[6] for it in items])
    return {"ivt": videowise_map(P, Yivt, vids, NIVT), "i": videowise_map(pi, Yi, vids, NI),
            "v": videowise_map(pv, Yv, vids, NV), "t": videowise_map(pt, Yt, vids, NT)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--dump_oof", default="")  # アンサンブル用: best_model の val OOF スコアを npz 保存して終了
    ap.add_argument("--dump_split", default="val")  # dump 対象: val / train / all（蒸留教師soft用）
    args = ap.parse_args(); cfg = yaml.safe_load(open(args.config))
    name = cfg["experiment"]["name"]
    outdir = os.path.join(HERE, "..", "results", name, f"fold{cfg['data']['val_fold']}")
    os.makedirs(outdir, exist_ok=True); lg = setup_logger(outdir)
    yaml.safe_dump(cfg, open(os.path.join(outdir, "config.yaml"), "w"))
    seed_all(cfg["experiment"].get("seed", 42))
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if cfg["train"].get("amp", "bf16") == "bf16" else torch.float16

    v2f = {r["video"]: int(r["fold"]) for r in csv.DictReader(open(cfg["data"]["folds_csv"]))}
    tr = build_index(cfg, v2f, False); va = build_index(cfg, v2f, True)
    lg.info(f"train={len(tr)} val={len(va)} val_fold={cfg['data']['val_fold']} amp={amp_dtype}")
    dcfg = cfg.get("distill", {})  # D: 蒸留. teacher_soft(npz) の soft ivt を train frame に付与
    teacher_lut = None
    if dcfg.get("teacher_soft"):
        _td = np.load(dcfg["teacher_soft"], allow_pickle=True)
        _tsc = np.asarray(_td["scores"], dtype=np.float32); _tv = _td["vids"]; _tf = _td["fids"]  # 実体化(NpzFile再解凍のO(n^2)回避)
        teacher_lut = {(str(v), int(f)): _tsc[i]
                       for i, (v, f) in enumerate(zip(_tv, _tf))}
        lg.info(f"distill: teacher_soft n={len(teacher_lut)} weight={dcfg.get('weight', 1.0)} beta={dcfg.get('beta', 1.0)}")
    tl = DataLoader(ClipDS(tr, cfg, True, teacher_lut=teacher_lut), batch_size=cfg["train"]["bs"], shuffle=True,
                    num_workers=cfg["train"]["workers"], pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(ClipDS(va, cfg, False), batch_size=cfg["train"].get("val_bs", cfg["train"]["bs"]), shuffle=False,
                    num_workers=cfg["train"]["workers"], pin_memory=True, persistent_workers=True)

    mtype = cfg["model"].get("type", "stageA")  # "stageA"(TUF) or "full"(TUF+PIC+TGR)
    full = mtype == "full"
    bb = torch.hub.load(DINOV3_REPO, "dinov3_vitl16", source="local", weights=VITL_WEIGHTS)
    Cls = SpiritFull_FT if full else SpiritStageA_FT
    mask_prior = cfg["model"].get("mask_prior", False)   # #2: 器具マスク空間prior token
    frozen_feat = cfg["model"].get("frozen_feat", False)  # #3: 凍結target/presence token
    extra_kw = dict(mask_prior=mask_prior, frozen_feat=frozen_feat) if full else {}
    model = Cls(bb, freeze_blocks=cfg["model"].get("freeze_blocks", 16),
                grad_ckpt=cfg["model"].get("grad_ckpt", True),
                d=cfg["model"].get("d", 128), ds=cfg["model"].get("ds", 256),
                nhead=cfg["model"].get("nhead", 4), dropout=cfg["model"].get("dropout", 0.05),
                T=cfg["data"]["clip_len"], N=64, **extra_kw).to(device)
    lg.info(f"mask_prior={mask_prior} frozen_feat={frozen_feat}")
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lg.info(f"model={mtype} trainable params: {ntr/1e6:.1f}M")

    if args.dump_oof:  # OOF/教師soft dump: best_model で推論 → sigmoid ivt スコアを保存
        bm = os.path.join(outdir, "best_model.pth")
        model.load_state_dict(torch.load(bm, map_location=device)["model"]); model.eval()
        items = {"val": va, "train": tr, "all": va + tr}[args.dump_split]
        dl = DataLoader(ClipDS(items, cfg, False), batch_size=cfg["train"].get("val_bs", cfg["train"]["bs"]),
                        shuffle=False, num_workers=cfg["train"]["workers"], pin_memory=True)
        S = []
        with torch.no_grad():
            for b in dl:
                with torch.autocast("cuda", dtype=amp_dtype):
                    o = model(b["frames"].to(device), **_extra(b, device))
                S.append(torch.sigmoid(o["ivt"]).float().cpu().numpy())
        P = np.concatenate(S)
        np.savez(args.dump_oof, scores=P, labels=np.stack([it[3] for it in items]),
                 vids=np.array([it[0] for it in items]), fids=np.array([it[1] for it in items]))
        lg.info(f"dumped {args.dump_split} -> {args.dump_oof} scores={P.shape}"); return

    pw_spec = [("ivt", 3, NIVT), ("i", 4, NI), ("v", 5, NV), ("t", 6, NT)]
    if full:
        pw_spec += [("it", 7, NIT), ("iv", 8, NIV)]
    pw = {k: pos_weight(tr, ix, n, cfg["train"]["pos_weight_cap"]).to(device) for k, ix, n in pw_spec}
    lf = {k: nn.BCEWithLogitsLoss(pos_weight=pw[k]) for k in pw}
    dflt_lw = {"ivt": 1.0, "i": 1.0, "v": 1.0, "t": 1.0}
    if full:
        dflt_lw.update({"it": 0.5, "iv": 0.5})  # pairwise 補助（SPIRIT: unary<pairwise<triplet 重み）
    lw = {**dflt_lw, **cfg["train"].get("loss_weights", {})}  # config で上書き, 欠けは既定で補完
    heads = ("ivt", "i", "v", "t", "it", "iv") if full else ("ivt", "i", "v", "t")

    epochs = cfg["train"]["epochs"]; accum = cfg["train"].get("grad_accum", 1)
    opt = torch.optim.AdamW(model.param_groups(cfg["train"]["lr"], cfg["train"]["backbone_lr"]),
                            weight_decay=cfg["train"]["weight_decay"])
    steps = (len(tl) // accum) * epochs; warm = int(steps * cfg["train"]["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(warm, 1) if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    fcm = cfg["train"].get("framecutmix")  # 時間aug FrameCutMix (batch-level)
    best = -1.0; hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); run = 0.0; opt.zero_grad(set_to_none=True)
        _t = time.time()
        for bi, b in enumerate(tl):
            if bi and bi % 100 == 0:
                dt = time.time() - _t; ips = 100 * cfg["train"]["bs"] / dt
                lg.info(f"  ep{ep} step{bi}/{len(tl)} {dt/100*1000:.0f}ms/step {ips:.1f} img/s"); _t = time.time()
            frames = b["frames"]; yb = {k: b[k] for k in heads}
            if fcm and random.random() < fcm.get("p", 0.5):  # 時間aug FrameCutMix: 先頭c枚を別sampleに差替(keyframe=末尾は保持)+ラベルmix
                Bn, T_ = frames.shape[:2]; perm = torch.randperm(Bn)
                c = random.randint(1, T_ - 1); frames = frames.clone(); frames[:, :c] = frames[perm][:, :c]
                lam = (T_ - c) / T_
                yb = {k: lam * yb[k] + (1 - lam) * yb[k][perm] for k in yb}
            with torch.autocast("cuda", dtype=amp_dtype):
                o = model(frames.to(device), **_extra(b, device))
                loss = sum(lw[k] * lf[k](o[k], yb[k].to(device)) for k in heads)
                if "teacher" in b and dcfg.get("weight", 0) > 0 and torch.isfinite(o["ivt"]).all():  # D: 蒸留(ivt)+SPIRIT式重み
                    tivt = b["teacher"].to(device).clamp(1e-4, 1 - 1e-4)      # 教師 soft ivt [B,85]
                    ht = b["has_teach"].to(device); gt = b["ivt"].to(device)
                    with torch.no_grad():
                        sivt = torch.sigmoid(o["ivt"].float()).clamp(1e-4, 1 - 1e-4)  # 生徒 prob(fp32で安定化)
                        ts = -(tivt * torch.log(sivt) + (1 - tivt) * torch.log(1 - sivt)).mean(1)  # 教師↔生徒 不一致
                        tg = -(gt * torch.log(tivt) + (1 - gt) * torch.log(1 - tivt)).mean(1)       # 教師↔GT 不一致
                        wn = (ts * torch.exp(-tg / dcfg.get("beta", 1.0)) * ht).clamp(0, 3.0)  # clip で外れ値の勾配爆発防止
                    dw = dcfg["weight"] * min(1.0, (ep + 1) / dcfg.get("warmup_ep", 2))  # weight warmup
                    dloss = (torch.nn.functional.binary_cross_entropy_with_logits(o["ivt"], tivt, reduction="none").mean(1) * wn).sum() / (wn.sum() + 1e-6)
                    if torch.isfinite(dloss):
                        loss = loss + lw["ivt"] * dw * dloss
                loss = loss / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            run += loss.item() * accum
        mp = evaluate(model, vl, va, device, amp_dtype)
        rec = {"epoch": ep, "loss": run / len(tl), **{f"vmAP_{k}": mp[k] for k in mp}}; hist.append(rec)
        lg.info(f"[ep{ep}] loss={run/len(tl):.3f} | ivt={mp['ivt']:.4f} i={mp['i']:.4f} "
                f"v={mp['v']:.4f} t={mp['t']:.4f} ({time.time()-t0:.0f}s)")
        if mp["ivt"] > best:
            best = mp["ivt"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best}, os.path.join(outdir, "best_model.pth"))
            lg.info(f"  new best ivt={best:.4f} -> best_model.pth")
        json.dump(hist, open(os.path.join(outdir, "training_log.json"), "w"), indent=2)
    lg.info(f"done. best ivt vmAP={best:.4f}")


if __name__ == "__main__":
    main()
