"""Pre-train Swin3D-B on CholecT50 clips for surgical VERB (+ optional TOOL/instrument) recognition,
to initialise the MB fusion's Swin3D verb branch (Kinetics-400 -> surgical action).

Rationale: verb is the fusion's trainable core and weakest component (v~0.53). CholecT50 shares 9/13
MB verbs and is also 1fps, so the 16-frame@stride2 clip has the SAME 30s temporal span as MB.
Verb strongly depends on which tool acts -> optionally co-train an instrument head (tool-aware features).
Both verb and instrument labels are derived from the per-frame triplet_id via label_mapping.txt.

Saves the Swin3D backbone (patch_embed/pos_drop/features/norm) so train_verb_fusion.py can load it
via model.swin_init. Heads are pretraining-only.

Usage: GPU=2 python src/pretrain_swin3d_cholec.py --config config_pretrain_verb.yaml
"""
import argparse
import glob
import json
import logging
import math
import os
import random
import shutil
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
CT50 = os.environ.get("CHOLECT50_ROOT", "data/CholecT50")
NV_CH, NI_CH = 10, 6          # CholecT50 verbs / instruments
KMEAN = np.array([0.485, 0.456, 0.406], np.float32); KSTD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def setup_logger(outdir):
    lg = logging.getLogger("pre"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    fh = logging.FileHandler(os.path.join(outdir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def build_maps():
    lm = np.loadtxt(os.path.join(CT50, "label_mapping.txt"), delimiter=",", dtype=int)  # [IVT,I,V,T,IV,IT]
    return {r[0]: r[1] for r in lm}, {r[0]: r[2] for r in lm}   # tid->instrument, tid->verb


class CholecClipDS(Dataset):
    def __init__(self, cfg, split_videos, train):
        self.T = cfg["data"]["clip_len"]; self.stride = cfg["data"]["clip_stride"]
        self.img = cfg["data"]["img"]; self.train = train
        self.fstride = cfg["data"].get("frame_stride", 3)   # subsample labeled frames to cut clip count
        self.offsets = cfg["data"].get("frame_offsets", None)  # dilated multi-scale sampling (matches fusion)
        tid2i, tid2v = build_maps()
        self.items = []   # (video, frame_id, verb_multihot, inst_multihot)
        for vp in split_videos:
            vid = os.path.basename(vp)
            lf = os.path.join(CT50, "labels", f"{vid}.json")
            if not os.path.exists(lf):
                continue
            ann = json.load(open(lf))["annotations"]
            fids = sorted(int(f) for f in ann)
            for k, fid in enumerate(fids):
                if k % self.fstride:
                    continue
                verbs, insts = set(), set()
                for r in ann[str(fid)]:
                    tid = r[0]
                    if tid in tid2v and tid >= 0:
                        verbs.add(tid2v[tid]); insts.add(tid2i[tid])
                vy = np.zeros(NV_CH, np.float32); iy = np.zeros(NI_CH, np.float32)
                for v in verbs:
                    vy[v] = 1.0
                for i in insts:
                    iy[i] = 1.0
                self.items.append((vid, fid, vy, iy))

    def __len__(self):
        return len(self.items)

    def _read(self, vid, fid):
        fp = os.path.join(CT50, "videos", vid, f"{max(fid,0):06d}.png")
        im = cv2.imread(fp)
        if im is None:
            im = np.zeros((self.img, self.img, 3), np.uint8)
        return cv2.cvtColor(cv2.resize(im, (self.img, self.img)), cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        vid, kf, vy, iy = self.items[idx]
        if self.offsets is not None:
            j = random.uniform(0.7, 1.4) if self.train else 1.0
            fids = [kf + int(round(o * j)) for o in self.offsets]  # dilated multi-scale sampling
        else:
            fids = [kf - self.stride * (self.T - 1 - i) for i in range(self.T)]  # causal, ends at kf
        clip = np.stack([self._read(vid, max(f, 0)) for f in fids]).astype(np.float32) / 255.0
        if self.train and random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()
        clip = (clip - KMEAN) / KSTD
        clip = torch.from_numpy(clip.transpose(3, 0, 1, 2))  # [3,T,H,W]
        return {"clip": clip, "verb": torch.from_numpy(vy), "inst": torch.from_numpy(iy)}


class Swin3DPretrain(nn.Module):
    def __init__(self, use_inst):
        super().__init__()
        m = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
        self.patch_embed, self.pos_drop, self.features, self.norm = m.patch_embed, m.pos_drop, m.features, m.norm
        self.head_v = nn.Linear(1024, NV_CH)
        self.use_inst = use_inst
        if use_inst:
            self.head_i = nn.Linear(1024, NI_CH)

    def backbone_state(self):
        return {"patch_embed": self.patch_embed.state_dict(), "pos_drop": self.pos_drop.state_dict(),
                "features": self.features.state_dict(), "norm": self.norm.state_dict()}

    def forward(self, clip):
        g = self.norm(self.features(self.pos_drop(self.patch_embed(clip)))).mean(dim=(1, 2, 3))
        return self.head_v(g), (self.head_i(g) if self.use_inst else None)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); Pv, Yv = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            ov, _ = model(b["clip"].to(device))
        Pv.append(torch.sigmoid(ov).float().cpu().numpy()); Yv.append(b["verb"].numpy())
    P = np.concatenate(Pv); Y = np.concatenate(Yv)
    aps = [average_precision_score(Y[:, c], P[:, c]) for c in range(NV_CH) if Y[:, c].sum() > 0]
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config_pretrain_verb.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); seed_all(cfg["experiment"]["seed"])
    outdir = os.path.join(HERE, "..", "results", cfg["experiment"]["name"])
    os.makedirs(outdir, exist_ok=True); shutil.copy(args.config, os.path.join(outdir, "config.yaml"))
    lg = setup_logger(outdir); device = "cuda"
    use_inst = "instrument" in cfg["model"].get("heads", ["verb"])

    vids = sorted(glob.glob(os.path.join(CT50, "videos", "VID*")))
    val_vids = vids[:3]; tr_vids = vids[3:]   # 3 videos held out for a verb-mAP sanity signal
    tr = CholecClipDS(cfg, tr_vids, True); va = CholecClipDS(cfg, val_vids, False)
    if args.limit:
        tr.items = tr.items[:args.limit]; va.items = va.items[:max(50, args.limit // 4)]
    lg.info(f"pretrain heads={cfg['model'].get('heads')} | train clips={len(tr)} val clips={len(va)}")

    tl = DataLoader(tr, cfg["train"]["bs"], shuffle=True, num_workers=cfg["train"]["workers"],
                    pin_memory=True, drop_last=True)
    vl = DataLoader(va, cfg["train"]["bs"], shuffle=False, num_workers=cfg["train"]["workers"], pin_memory=True)

    model = Swin3DPretrain(use_inst).to(device)
    bb = [p for n, p in model.named_parameters() if not n.startswith("head")]
    hd = [p for n, p in model.named_parameters() if n.startswith("head")]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["train"]["lr"] * cfg["train"]["backbone_lr_mult"]},
                             {"params": hd, "lr": cfg["train"]["lr"]}], weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]; steps = len(tl) * epochs; warm = int(steps * cfg["train"]["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(warm, 1) if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.cuda.amp.GradScaler()
    iw = cfg["train"].get("inst_weight", 1.0)
    bce = nn.BCEWithLogitsLoss()
    start, best, hist = 0, -1.0, []
    last = os.path.join(outdir, "last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); start = ck["epoch"] + 1
        best = ck.get("best", -1); hist = ck.get("hist", []); lg.info(f"resumed ep{start}")

    for ep in range(start, epochs):
        model.train(); t0 = time.time(); run = 0.0
        for bi, b in enumerate(tl):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                ov, oi = model(b["clip"].to(device))
                loss = bce(ov, b["verb"].to(device))
                if use_inst:
                    loss = loss + iw * bce(oi, b["inst"].to(device))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if bi % 100 == 0:
                lg.debug(f"ep{ep} {bi}/{len(tl)} loss={loss.item():.3f} lr={sched.get_last_lr()[1]:.2e}")
        vmap = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_loss": run / len(tl), "val_verb_mAP": vmap})
        lg.info(f"[ep{ep}] loss={run/len(tl):.4f} val_verb_mAP={vmap:.4f} ({time.time()-t0:.0f}s)")
        json.dump(hist, open(os.path.join(outdir, "training_log.json"), "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist}, last)
        if vmap > best:
            best = vmap
            torch.save({"backbone": model.backbone_state(), "epoch": ep, "val_verb_mAP": vmap,
                        "heads": cfg["model"].get("heads")}, os.path.join(outdir, "swin_backbone_best.pth"))
            lg.info(f"  new best verb_mAP={best:.4f} -> swin_backbone_best.pth")
    lg.info(f"done. best val_verb_mAP={best:.4f}")


if __name__ == "__main__":
    main()
