"""Step 1 MTL: MB target classification + external surgical SEGMENTATION (multi-task).

Motivation: v1 (ImageNet convnext_base) overfits (train_loss->0.0009, tmAP peaks ep2=0.682);
SurgeNet backbone-init underperformed (0.664). The real lever is external supervision.
CholecT50 and EndoVis2018 are SEGMENTATION datasets -> we add a dense seg head (per dataset,
native label space) on the SHARED backbone to shape anatomy/tissue features, while the MB
classification head (the one the fusion consumes) is trained on MB frames.

Design (deploy-compatible):
- backbone = standard timm convnext_base (num_classes=15). Its cls path == v1 exactly, so the
  saved best_model.pth is a drop-in for cache_frozen_feats.py / the Docker (backbone + 15-head).
- seg path taps the 4 stage outputs via forward hooks -> lite-FPN decoder -> per-dataset 1x1 head.
  Training-only; discarded at deploy.
- Round-robin over 3 sources (MB cls / cholec seg / endovis seg); loss routed by source.
- Eval = MB val videowise tmAP (cls head), identical protocol to train_target.py.

Usage: GPU=1 python src/train_target_mtl.py --config config_target_mtl.yaml
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

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(__file__)
REPO = os.environ.get("MB_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "reference/starter_kit"))
from utils.triplet_mappings import triplet_maps  # noqa

NT = 15                       # MB targets
CHOLEC_SEG_C = 16             # 15 cholec targets + background(15)
ENDOVIS_SEG_C = 12            # native EndoVis2018 classes (incl background-tissue=0)
DSAD_SEG_C = 16               # DSAD seg lives directly in MB-target space (15 MB + background=15)
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
CHOLEC_ROOT = os.path.join(REPO, "ex_data/cholec_triplet_seg")
ENDOVIS_ROOT = os.environ.get("ENDOVIS2018_ROOT", "data/EndoVis2018/train")
DSAD_ROOT = os.environ.get("DSAD_ROOT", "data/dsad/multilabel")
CH_T = triplet_maps["cholect50"]["t"]; CH_NAME2ID = {v: int(k) for k, v in CH_T.items()}

# DSAD (Dresden) laparoscopic organ seg -> MB target id. Domain-closest to MB (laparoscopic abdomen);
# directly covers MB's weak classes colon/liver/spleen and the 0-sample spleen. Painted in this order
# (later overwrites on overlap; small_intestine/colon last as they interleave with larger organs).
DSAD_ANATOMY2MB = {"stomach": 1, "liver": 4, "spleen": 11, "colon": 7, "small_intestine": 0}

# mapped-cls co-training: external seg class id -> MB target id (only confidently-shared anatomy).
# Injects real positives/negatives into the MB cls head for MB's weak classes.
CHOLEC2MB = {8: 4, 10: 3, 6: 13, 9: 12}   # liver->liver, omentum->omentum, fluid->fluid, adhesion->adhesion
ENDOVIS2MB = {6: 2, 8: 5, 10: 0}          # thread->thread, suturing-needle->needle, intestine->small_bowel


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def setup_logger(outdir):
    lg = logging.getLogger("mtl"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    fh = logging.FileHandler(os.path.join(outdir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def load_folds(p):
    return {r["video"]: int(r["fold"]) for r in csv.DictReader(open(p))}


# ---------------- datasets ----------------
def _norm(img, size):
    return A.Compose([A.LongestMaxSize(size), A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
                      A.Normalize(MEAN, STD)])(image=img)["image"]


class MBClsDS(Dataset):
    """MB frames -> (img, mb15 multihot). Classification source."""
    def __init__(self, cfg, v2f, is_val):
        self.img = cfg["data"]["img"]; self.train = not is_val
        self.items = []
        vf = cfg["data"]["val_fold"]
        for lf in sorted(glob.glob(os.path.join(cfg["data"]["label_dir"], "*.json"))):
            vid = os.path.splitext(os.path.basename(lf))[0]
            if vid not in v2f or (v2f[vid] == vf) != is_val:
                continue
            d = json.load(open(lf)); byimg = defaultdict(set)
            for a in d["annotations"]:
                byimg[a["image_id"]].add(a["target_id"])
            for im in d["images"]:
                s = byimg.get(im["id"], set())
                if not s and cfg["data"].get("drop_empty", True):
                    continue
                y = np.zeros(NT, np.float32)
                for t in s:
                    if 0 <= t < NT:
                        y[t] = 1.0
                self.items.append((os.path.join(cfg["data"]["video_dir"], vid, f"{im['id']:06d}.jpg"), y, vid))
        m = (0.485, 0.456, 0.406)
        if self.train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                                  A.ShiftScaleRotate(0.05, 0.1, 12, border_mode=cv2.BORDER_CONSTANT, p=0.4),
                                  A.LongestMaxSize(self.img),
                                  A.PadIfNeeded(self.img, self.img, border_mode=cv2.BORDER_CONSTANT),
                                  A.Normalize(MEAN, STD)])
        else:
            self.aug = A.Compose([A.LongestMaxSize(self.img),
                                  A.PadIfNeeded(self.img, self.img, border_mode=cv2.BORDER_CONSTANT),
                                  A.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fp, y, _ = self.items[i]
        img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        x = self.aug(image=img)["image"]
        return {"img": torch.from_numpy(x.transpose(2, 0, 1)), "task": "mb",
                "cls": torch.from_numpy(y)}


class _SegDS(Dataset):
    """Base for external segmentation sources. Subclass sets self.items, self.mb_map (src->MB id),
    and self._load(i)->(img RGB, seg HxW uint8). bg_val pads the mask.
    Also emits mb_label/mb_mask for mapped-cls co-training (masked BCE on the MB cls head)."""
    def __init__(self, img, train, seg_c, bg_val, mb_map):
        self.img = img; self.train = train; self.seg_c = seg_c; self.bg_val = bg_val; self.mb_map = mb_map
        aug = [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(0.2, 0.2, p=0.5)] if train else []
        self.aug = A.Compose(aug + [A.LongestMaxSize(img),
                                    A.PadIfNeeded(img, img, border_mode=cv2.BORDER_CONSTANT, fill=0,
                                                  fill_mask=bg_val),
                                    A.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, seg = self._load(i)
        present = set(int(v) for v in np.unique(seg))       # from full-res GT (before resize)
        mb_label = np.zeros(NT, np.float32); mb_mask = np.zeros(NT, np.float32)
        for src_c, mb_c in self.mb_map.items():
            mb_mask[mb_c] = 1.0                              # supervise this MB class (present or absent)
            if src_c in present:
                mb_label[mb_c] = 1.0
        r = self.aug(image=img, mask=seg)
        return {"img": torch.from_numpy(r["image"].transpose(2, 0, 1)), "task": "seg",
                "seg": torch.from_numpy(r["mask"].astype(np.int64)), "seg_ds": self.seg_name,
                "mb_label": torch.from_numpy(mb_label), "mb_mask": torch.from_numpy(mb_mask)}


class CholecSegDS(_SegDS):
    seg_name = "cholec"
    def __init__(self, img, train, split="train"):
        super().__init__(img, train, CHOLEC_SEG_C, bg_val=15, mb_map=CHOLEC2MB)
        self.items = [p for p in sorted(glob.glob(os.path.join(CHOLEC_ROOT, split, "ann_dir", "*.json")))]

    def _load(self, i):
        d = json.load(open(self.items[i]))
        ip = self.items[i].replace("/ann_dir/", "/img_dir/").replace(".json", ".png")
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        H, W = int(d.get("imageHeight") or img.shape[0]), int(d.get("imageWidth") or img.shape[1])
        seg = np.full((H, W), 15, np.uint8)
        for s in d["shapes"]:
            t = s.get("target")
            if t not in CH_NAME2ID:
                continue
            cv2.fillPoly(seg, [np.array(s["points"], np.int32)], CH_NAME2ID[t])
        return img, seg


class EndoVisSegDS(_SegDS):
    seg_name = "endovis"
    def __init__(self, img, train):
        super().__init__(img, train, ENDOVIS_SEG_C, bg_val=0, mb_map=ENDOVIS2MB)
        labels = json.load(open(os.path.join(os.path.dirname(ENDOVIS_ROOT), "labels.json")))["classes"]
        self.col2id = {}
        for c in labels:
            r, g, b, _ = c["color"]; self.col2id[(b, g, r)] = c["classid"]
        self.items = []
        for seq in sorted(glob.glob(os.path.join(ENDOVIS_ROOT, "seq_*"))):
            for lp in sorted(glob.glob(os.path.join(seq, "labels", "*.png"))):
                fp = os.path.join(seq, "left_frames", os.path.basename(lp))
                if os.path.exists(fp):
                    self.items.append((fp, lp))

    def _load(self, i):
        fp, lp = self.items[i]
        img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        m = cv2.imread(lp)  # BGR
        seg = np.zeros(m.shape[:2], np.uint8)
        for bgr, cid in self.col2id.items():
            seg[(m[:, :, 0] == bgr[0]) & (m[:, :, 1] == bgr[1]) & (m[:, :, 2] == bgr[2])] = cid
        return img, seg


class DSADSegDS(_SegDS):
    """DSAD multilabel: per-frame per-organ binary masks -> MB-target semantic seg (15 MB + bg=15).
    mb_map is identity on the mapped MB ids (seg already carries MB ids)."""
    seg_name = "dsad"
    def __init__(self, img, train):
        mbids = sorted(set(DSAD_ANATOMY2MB.values()))
        super().__init__(img, train, DSAD_SEG_C, bg_val=15, mb_map={c: c for c in mbids})
        self.items = []
        for vid in sorted(glob.glob(os.path.join(DSAD_ROOT, "*"))):
            for ip in sorted(glob.glob(os.path.join(vid, "image*.png"))):
                n = os.path.basename(ip)[5:-4]        # image{NN}.png
                self.items.append((ip, os.path.dirname(ip), n))

    def _load(self, i):
        ip, vdir, n = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        seg = np.full(img.shape[:2], 15, np.uint8)     # background
        for anat, mbid in DSAD_ANATOMY2MB.items():     # dict order = paint order
            mp = os.path.join(vdir, f"mask{n}_{anat}.png")
            if not os.path.exists(mp):
                continue
            mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mk is not None and mk.shape == seg.shape:
                seg[mk > 127] = mbid
        return img, seg


# ---------------- model ----------------
class LiteFPN(nn.Module):
    def __init__(self, chs, dim=128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in chs])
        self.out = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())

    def forward(self, feats):  # fine->coarse
        x = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="nearest") + self.lat[i](feats[i])
        return self.out(x)     # at finest stride (stride4)


class MTLTarget(nn.Module):
    def __init__(self, backbone="convnext_base.fb_in22k_ft_in1k_384"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=NT)
        # tap the 4 convnext stages
        self._feats = []
        for st in self.backbone.stages:
            st.register_forward_hook(lambda m, i, o: self._feats.append(o))
        chs = [self.backbone.stages[k].blocks[-1].conv_dw.out_channels for k in range(4)]
        self.dec = LiteFPN(chs, dim=128)
        self.seg_heads = nn.ModuleDict({"cholec": nn.Conv2d(128, CHOLEC_SEG_C, 1),
                                        "endovis": nn.Conv2d(128, ENDOVIS_SEG_C, 1),
                                        "dsad": nn.Conv2d(128, DSAD_SEG_C, 1)})

    def forward(self, x, seg_ds=None):
        self._feats = []
        logits = self.backbone(x)
        seg = None
        if seg_ds is not None:
            seg = self.seg_heads[seg_ds](self.dec(self._feats))
        return logits, seg


# ---------------- eval (MB cls) ----------------
@torch.no_grad()
def evaluate(model, loader, val_items, device):
    model.eval(); probs = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = model(b["img"].to(device))
        probs.append(torch.sigmoid(lg).float().cpu().numpy())
    P = np.concatenate(probs); Y = np.stack([it[1] for it in val_items]); vids = [it[2] for it in val_items]
    byv = defaultdict(list)
    for i, v in enumerate(vids):
        byv[v].append(i)
    vmaps = []
    for v, idx in byv.items():
        yv, pv = Y[idx], P[idx]
        aps = [average_precision_score(yv[:, c], pv[:, c]) for c in range(NT) if yv[:, c].sum() > 0]
        if aps:
            vmaps.append(np.mean(aps))
    per = {c: (average_precision_score(Y[:, c], P[:, c]) if Y[:, c].sum() > 0 else float("nan")) for c in range(NT)}
    return float(np.mean(vmaps)), per


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config_target_mtl.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); seed_all(cfg["experiment"]["seed"])
    outdir = os.path.join(HERE, "..", "results", cfg["experiment"]["name"], f"fold{cfg['data']['val_fold']}")
    os.makedirs(outdir, exist_ok=True); shutil.copy(args.config, os.path.join(outdir, "config.yaml"))
    lg = setup_logger(outdir); device = "cuda"
    img = cfg["data"]["img"]; bs = cfg["train"]["bs"]; nw = cfg["train"]["workers"]
    v2f = load_folds(cfg["data"]["folds_csv"])

    mb_tr = MBClsDS(cfg, v2f, is_val=False); mb_va = MBClsDS(cfg, v2f, is_val=True)
    chol = CholecSegDS(img, True); endo = EndoVisSegDS(img, True); dsad = DSADSegDS(img, True)
    lg.info(f"MB train={len(mb_tr)} val={len(mb_va)} | cholec seg={len(chol)} | endovis seg={len(endo)} | dsad seg={len(dsad)}")

    # pos_weight for MB cls from train
    Ytr = np.stack([it[1] for it in mb_tr.items]); pos = Ytr.sum(0)
    pw = np.clip(np.where(pos > 0, (len(Ytr) - pos) / np.maximum(pos, 1), 1.0), 0, cfg["train"]["pos_weight_cap"])
    clsloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))
    segloss = nn.CrossEntropyLoss()

    dl_mb = DataLoader(mb_tr, bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    dl_ch = DataLoader(chol, bs, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    dl_en = DataLoader(endo, bs, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    dl_ds = DataLoader(dsad, bs, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    vl = DataLoader(mb_va, bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = MTLTarget(cfg["model"]["backbone"]).to(device)
    head_keys = ("head", "fc", "classifier", "dec", "seg_heads")
    bb, hd = [], []
    for n, p in model.named_parameters():
        (hd if any(k in n for k in head_keys) else bb).append(p)
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["train"]["lr"] * cfg["train"]["backbone_lr_mult"]},
                             {"params": hd, "lr": cfg["train"]["lr"]}], weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    steps_per_ep = cfg["train"].get("steps_per_epoch", len(dl_mb))
    if args.limit:
        steps_per_ep = args.limit
    total = steps_per_ep * epochs; warm = int(total * cfg["train"]["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler()
    seg_w = cfg["train"].get("seg_weight", 0.5)
    mapcls_w = cfg["train"].get("mapcls_weight", 0.0)   # Path A: inject external mapped labels into MB cls head
    pw_t = torch.tensor(pw, dtype=torch.float32, device=device)
    # round-robin schedule weights: how many steps each source per 'round'
    sched_seq = cfg["train"].get("source_schedule", ["mb", "mb", "cholec", "endovis"])
    gens = {"mb": cycle(dl_mb), "cholec": cycle(dl_ch), "endovis": cycle(dl_en), "dsad": cycle(dl_ds)}

    start, best, hist = 0, -1.0, []
    last = os.path.join(outdir, "last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"] + 1; best = ck.get("best", -1); hist = ck.get("hist", [])
        lg.info(f"resumed ep{start} best={best:.4f}")

    for ep in range(start, epochs):
        model.train(); t0 = time.time(); run = defaultdict(float); cnt = defaultdict(int)
        for si in range(steps_per_ep):
            src = sched_seq[si % len(sched_seq)]
            b = next(gens[src])
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if src == "mb":
                    lgt, _ = model(b["img"].to(device))
                    loss = clsloss(lgt, b["cls"].to(device))
                else:
                    lgt, seg = model(b["img"].to(device), seg_ds=src)
                    gt = b["seg"].to(device)
                    if seg.shape[-2:] != gt.shape[-2:]:
                        gt = F.interpolate(gt.unsqueeze(1).float(), size=seg.shape[-2:], mode="nearest")[:, 0].long()
                    loss = seg_w * segloss(seg, gt)
                    if mapcls_w > 0:  # masked BCE on MB cls head for the mappable classes (same forward)
                        mm = b["mb_mask"].to(device)
                        bce = F.binary_cross_entropy_with_logits(
                            lgt, b["mb_label"].to(device), pos_weight=pw_t, reduction="none")
                        loss = loss + mapcls_w * (bce * mm).sum() / mm.sum().clamp(min=1.0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run[src] += loss.item(); cnt[src] += 1
            if si % 100 == 0:
                lg.debug(f"ep{ep} {si}/{steps_per_ep} {src} loss={loss.item():.4f} lr={sched.get_last_lr()[1]:.2e}")
        vmap, per = evaluate(model, vl, mb_va.items, device)
        losses = {k: run[k] / max(cnt[k], 1) for k in run}
        hist.append({"epoch": ep, "loss": losses, "videowise_tmAP": vmap})
        lg.info(f"[ep{ep}] loss={ {k: round(v,3) for k,v in losses.items()} } tmAP={vmap:.4f} ({time.time()-t0:.0f}s)")
        lg.info("  per-class AP: " + ", ".join(f"{c}:{per[c]:.2f}" for c in range(NT) if per[c] == per[c]))
        json.dump(hist, open(os.path.join(outdir, "training_log.json"), "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist}, last)
        if vmap > best:
            best = vmap
            # deploy-compatible: save ONLY the standard convnext backbone (backbone+15-head), like v1
            torch.save({"model": model.backbone.state_dict(), "epoch": ep, "videowise_tmAP": vmap,
                        "backbone": cfg["model"]["backbone"], "img": img},
                       os.path.join(outdir, "best_model.pth"))
            lg.info(f"  new best tmAP={best:.4f} -> best_model.pth (backbone-only, deploy-compatible)")
    lg.info(f"done. best videowise_tmAP={best:.4f}")


if __name__ == "__main__":
    main()
