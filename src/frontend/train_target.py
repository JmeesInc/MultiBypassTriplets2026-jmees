"""Step 1 of the triplet pipeline: MB TARGET (object) image-model.
15-class frame-level multi-label classification (timm backbone + linear head, BCE w/ pos_weight).
Video-level CV (fold v001). Later frozen; its pooled features + logits feed the Swin3D verb model.

Rules: AMP fp16, seed fixed, checkpoint resume (last+best), logging module, config copied,
outputs to results/{name}/fold{val_fold}/.

Usage: GPU=0 python src/train_target.py --config config_target.yaml
"""
import argparse
import csv
import glob
import json
import logging
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
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(__file__)
NT = 15


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def setup_logger(outdir):
    lg = logging.getLogger("tgt"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    fh = logging.FileHandler(os.path.join(outdir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def load_folds(csv_path):
    v2f = {}
    for r in csv.DictReader(open(csv_path)):
        v2f[r["video"]] = int(r["fold"])
    return v2f


def build_list(cfg, v2f, is_val):
    """[(img_path, multihot(15), video)] for frames in/out of val_fold."""
    items = []
    vf = cfg["data"]["val_fold"]
    for lf in sorted(glob.glob(os.path.join(cfg["data"]["label_dir"], "*.json"))):
        vid = os.path.splitext(os.path.basename(lf))[0]
        if vid not in v2f or (v2f[vid] == vf) != is_val:
            continue
        d = json.load(open(lf))
        byimg = defaultdict(set)
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
            fp = os.path.join(cfg["data"]["video_dir"], vid, f"{im['id']:06d}.jpg")
            items.append((fp, y, vid))
    return items


class TargetDS(Dataset):
    def __init__(self, items, img, train):
        self.items = items
        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
        if train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                                  A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                                  A.ShiftScaleRotate(0.05, 0.1, 12, border_mode=cv2.BORDER_CONSTANT, p=0.4),
                                  A.LongestMaxSize(img), A.PadIfNeeded(img, img, border_mode=cv2.BORDER_CONSTANT),
                                  A.Normalize(mean, std)])
        else:
            self.aug = A.Compose([A.LongestMaxSize(img), A.PadIfNeeded(img, img, border_mode=cv2.BORDER_CONSTANT),
                                  A.Normalize(mean, std)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fp, y, _ = self.items[i]
        img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        x = self.aug(image=img)["image"]
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, loader, val_items, device):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x.to(device)))
        probs.append(p.float().cpu().numpy())
    P = np.concatenate(probs)                       # [N,15]
    Y = np.stack([it[1] for it in val_items])       # [N,15]
    vids = [it[2] for it in val_items]
    # videowise mAP: per video, per-class AP over classes present in that video, mean
    byv = defaultdict(list)
    for i, v in enumerate(vids):
        byv[v].append(i)
    vmaps = []
    for v, idx in byv.items():
        yv, pv = Y[idx], P[idx]
        aps = [average_precision_score(yv[:, c], pv[:, c]) for c in range(NT) if yv[:, c].sum() > 0]
        if aps:
            vmaps.append(np.mean(aps))
    # overall per-class AP (all val frames)
    ov = [average_precision_score(Y[:, c], P[:, c]) for c in range(NT) if Y[:, c].sum() > 0]
    per_cls = {c: (average_precision_score(Y[:, c], P[:, c]) if Y[:, c].sum() > 0 else float("nan"))
               for c in range(NT)}
    return float(np.mean(vmaps)), float(np.mean(ov)), per_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config_target.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_all(cfg["experiment"]["seed"])
    name = cfg["experiment"]["name"]
    outdir = os.path.join(HERE, "..", "results", name, f"fold{cfg['data']['val_fold']}")
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(args.config, os.path.join(outdir, "config.yaml"))
    lg = setup_logger(outdir); device = "cuda"

    v2f = load_folds(cfg["data"]["folds_csv"])
    tr = build_list(cfg, v2f, is_val=False)
    va = build_list(cfg, v2f, is_val=True)
    if args.limit:
        tr = tr[:args.limit]; va = va[:max(50, args.limit // 4)]
    lg.info(f"train={len(tr)} val={len(va)} | val_fold={cfg['data']['val_fold']}")
    # pos_weight from train
    Ytr = np.stack([it[1] for it in tr])
    pos = Ytr.sum(0); neg = len(Ytr) - pos
    pw = np.clip(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), 0, cfg["train"]["pos_weight_cap"])
    lg.info(f"pos counts: {pos.astype(int).tolist()}")
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    img = cfg["data"]["img"]
    tl = DataLoader(TargetDS(tr, img, True), batch_size=cfg["train"]["bs"], shuffle=True,
                    num_workers=cfg["train"]["workers"], pin_memory=True, drop_last=True)
    vl = DataLoader(TargetDS(va, img, False), batch_size=cfg["train"]["bs"], shuffle=False,
                    num_workers=cfg["train"]["workers"], pin_memory=True)

    if cfg["model"].get("surgenet_init"):
        sys.path.insert(0, HERE)
        from surgenet_loader import load_surgenetxl_caformer
        model = load_surgenetxl_caformer(num_classes=cfg["model"]["num_classes"]).to(device)
        lg.info("model init: SurgeNetXL CAFormer-S18 (surgical SSL, timm caformer_s18)")
    else:
        model = timm.create_model(cfg["model"]["backbone"], pretrained=True,
                                  num_classes=cfg["model"]["num_classes"]).to(device)
    # differential LR: backbone vs head/classifier
    head_keys = ("head", "fc", "classifier")
    bb, hd = [], []
    for n, p in model.named_parameters():
        (hd if any(k in n for k in head_keys) else bb).append(p)
    opt = torch.optim.AdamW([
        {"params": bb, "lr": cfg["train"]["lr"] * cfg["train"]["backbone_lr_mult"]},
        {"params": hd, "lr": cfg["train"]["lr"]},
    ], weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]; steps = len(tl) * epochs
    warm = int(steps * cfg["train"]["warmup_ratio"])

    def lr_lambda(s):
        import math
        if s < warm:
            return s / max(warm, 1)
        prog = (s - warm) / max(steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_ep, best, hist = 0, -1.0, []
    last = os.path.join(outdir, "last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1; best = ck.get("best", -1); hist = ck.get("hist", [])
        lg.info(f"resumed ep{start_ep} best={best:.4f}")

    for ep in range(start_ep, epochs):
        model.train(); t0 = time.time(); run = 0.0
        for bi, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossfn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if bi % 100 == 0:
                lg.debug(f"ep{ep} {bi}/{len(tl)} loss={loss.item():.4f} lr={sched.get_last_lr()[1]:.2e}")
        vmap, omap, per = evaluate(model, vl, va, device)
        hist.append({"epoch": ep, "train_loss": run / len(tl), "videowise_tmAP": vmap, "overall_tmAP": omap})
        lg.info(f"[ep{ep}] loss={run/len(tl):.4f} videowise_tmAP={vmap:.4f} overall_tmAP={omap:.4f} ({time.time()-t0:.0f}s)")
        lg.info("  per-class AP: " + ", ".join(f"{c}:{per[c]:.2f}" for c in range(NT) if per[c] == per[c]))
        json.dump(hist, open(os.path.join(outdir, "training_log.json"), "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist}, last)
        if vmap > best:
            best = vmap
            torch.save({"model": model.state_dict(), "epoch": ep, "videowise_tmAP": vmap,
                        "backbone": cfg["model"]["backbone"], "img": img}, os.path.join(outdir, "best_model.pth"))
            lg.info(f"  new best videowise_tmAP={best:.4f} -> best_model.pth")
    lg.info(f"done. best videowise_tmAP={best:.4f}")


if __name__ == "__main__":
    main()
