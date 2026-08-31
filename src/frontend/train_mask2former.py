"""Train HF Mask2Former instance segmentation on a unified COCO (Stage A / B).

- Backbone init: facebook/mask2former-swin-large-coco-instance (class head resized).
- COCO polygons/RLE -> per-image instance map -> Mask2FormerImageProcessor.
- Logs to results/{exp}/, checkpoints + resume supported.

Usage:
  python src/train_mask2former.py --train coco/stageA_train.json --val coco/stageA_val.json \
      --exp stageA --num_labels 1 --epochs 20 --bs 4 --img 512
"""
import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import albumentations as A
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor, TrainingArguments, Trainer)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("train")


def ann_to_mask(ann, h, w):
    """Decode COCO segmentation (polygon list / uncompressed RLE / compressed RLE)
    to a binary mask. Returns None on malformed/empty input."""
    seg = ann.get("segmentation")
    try:
        if isinstance(seg, list):  # polygon(s); keep only valid polygons (>=3 pts)
            polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
            if not polys:
                return None
            rle = mask_utils.merge(mask_utils.frPyObjects(polys, h, w))
        elif isinstance(seg, dict):
            rle = dict(seg)
            if isinstance(rle.get("counts"), list):  # uncompressed RLE
                rle = mask_utils.frPyObjects(rle, h, w)
            elif isinstance(rle.get("counts"), str):  # compressed RLE (str -> bytes)
                rle["counts"] = rle["counts"].encode("ascii")
        else:
            return None
        return mask_utils.decode(rle)
    except Exception:
        return None


class CocoInstanceDS(torch.utils.data.Dataset):
    def __init__(self, coco_json, processor, train=True, img=512,
                 paste=None, paste_weight=1.0):
        self.coco = COCO(coco_json)
        self.ids = sorted(self.coco.imgs.keys())
        self.processor = processor
        self.train = train
        self.paste = paste          # ToolPasteMB instance (train only) or None
        self.paste_weight = paste_weight
        pad = A.PadIfNeeded(min_height=img, min_width=img, border_mode=0,
                            fill=0, fill_mask=0, position="top_left")
        if train:
            self.tf = A.Compose([A.LongestMaxSize(img), pad, A.HorizontalFlip(p=0.5)],
                                is_check_shapes=False)
        else:
            self.tf = A.Compose([A.LongestMaxSize(img), pad], is_check_shapes=False)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        rec = self.coco.imgs[iid]
        image = np.array(Image.open(rec["file_name"]).convert("RGB"))
        h, w = rec["height"], rec["width"]
        # ensure image matches the annotation reference size (polygons use h,w)
        if image.shape[0] != h or image.shape[1] != w:
            image = np.array(Image.fromarray(image).resize((w, h), Image.BILINEAR))
        anns = self.coco.imgToAnns[iid]
        items = []  # (mask, class_id, weight); paint order = list order (later on top)
        for a in anns:
            m = ann_to_mask(a, h, w)
            if m is None or m.shape != (h, w) or m.sum() == 0:
                continue
            items.append((m > 0, a["category_id"], float(a.get("weight", 1.0))))
        if self.paste is not None and self.train:
            existing = None
            if items:
                existing = np.zeros((h, w), dtype=bool)
                for m, _, _ in items:
                    existing |= m
            image, extra = self.paste(image, random.Random(random.getrandbits(64)),
                                      existing_mask=existing)
            items += [(m, cls, self.paste_weight) for m, cls in extra]
        inst = np.zeros((h, w), dtype=np.int32)  # 0 = background
        inst2sem = {}  # instance_id (1..N) -> 0-based semantic class
        wmap = {}
        for k, (m, cls, wt) in enumerate(items, start=1):
            inst[m] = k
            inst2sem[k] = cls  # 0-based class (no shift)
            wmap[k] = wt
        # drop instances almost fully occluded by later pastes (tiny remnants)
        ids_now, cnts = np.unique(inst, return_counts=True)
        for k, n_px in zip(ids_now, cnts):
            if k > 0 and n_px < 50:
                inst[inst == k] = 0
        surviving = [int(k) for k in np.unique(inst) if k > 0]
        weights = [wmap[k] for k in surviving]  # sorted-id order == processor order
        out = self.tf(image=image, mask=inst)
        image, inst = out["image"], out["mask"]
        enc = self.processor(images=[image], segmentation_maps=[inst],
                             instance_id_to_semantic_id=inst2sem, return_tensors="pt")
        cl = enc["class_labels"][0]
        # align per-instance weights to the processor's class_labels (fallback to 1.0 on mismatch)
        wt = torch.tensor(weights, dtype=torch.float32) if len(weights) == len(cl) \
            else torch.ones(len(cl), dtype=torch.float32)
        return {"pixel_values": enc["pixel_values"][0],
                "mask_labels": enc["mask_labels"][0],
                "class_labels": cl, "inst_weights": wt}


class DiffLRTrainer(Trainer):
    """Trainer with a lower LR for the backbone (pixel_level_module.encoder)."""
    backbone_lr = None

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        wd = self.args.weight_decay
        bb, rest = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (bb if "pixel_level_module.encoder" in n else rest).append(p)
        groups = [
            {"params": rest, "lr": self.args.learning_rate, "weight_decay": wd},
            {"params": bb, "lr": self.backbone_lr, "weight_decay": wd},
        ]
        self.optimizer = torch.optim.AdamW(groups, lr=self.args.learning_rate,
                                           betas=(0.9, 0.999), eps=1e-8)
        LOG.info(f"DiffLR optimizer: head_lr={self.args.learning_rate} "
                 f"backbone_lr={self.backbone_lr} (bb={len(bb)} rest={len(rest)} tensors)")
        return self.optimizer


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
        "inst_weights": [b["inst_weights"] for b in batch],
    }


# ---- per-instance loss weighting: patch HF Mask2FormerLoss to scale matched-instance
#      class/mask losses by the target's weight (stashed on criterion as _tgt_weights).
#      Reduces exactly to the original loss when all weights == 1.0. ----
def _install_weighted_loss():
    import torch.nn.functional as F
    from transformers.models.mask2former import modeling_mask2former as MM
    sample_point = MM.sample_point

    def loss_labels(self, class_queries_logits, class_labels, indices):
        B, Q, _ = class_queries_logits.shape
        idx = self._get_predictions_permutation_indices(indices)
        tco = torch.cat([t[j] for t, (_, j) in zip(class_labels, indices)])
        tc = torch.full((B, Q), self.num_labels, dtype=torch.int64, device=class_queries_logits.device)
        tc[idx] = tco
        ce = F.cross_entropy(class_queries_logits.transpose(1, 2), tc, weight=self.empty_weight, reduction="none")
        tw = getattr(self, "_tgt_weights", None)
        if tw is not None:
            sw = torch.ones((B, Q), device=ce.device, dtype=ce.dtype)
            sw[idx] = torch.cat([w[j] for w, (_, j) in zip(tw, indices)]).to(ce.dtype)
            ce = ce * sw
        return {"loss_cross_entropy": ce.mean()}

    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks):
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        pred_masks = pred_masks[:, None]; target_masks = target_masks[:, None]
        with torch.no_grad():
            pc = self.sample_points_using_uncertainty(
                pred_masks, lambda logits: self.calculate_uncertainty(logits),
                self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = sample_point(target_masks, pc, align_corners=False).squeeze(1)
        point_logits = sample_point(pred_masks, pc, align_corners=False).squeeze(1)
        n = point_logits.shape[0]
        tw = getattr(self, "_tgt_weights", None)
        if tw is not None and n > 0:
            pw = torch.cat([w[j] for w, (_, j) in zip(tw, indices)]).to(point_logits.dtype)
        else:
            pw = torch.ones(n, device=point_logits.device, dtype=point_logits.dtype)
        ce = F.binary_cross_entropy_with_logits(point_logits, point_labels, reduction="none").mean(1)
        loss_mask = (ce * pw).sum() / num_masks
        probs = point_logits.sigmoid()
        num = 2 * (probs * point_labels).sum(-1); den = probs.sum(-1) + point_labels.sum(-1)
        dice = 1 - (num + 1) / (den + 1)
        loss_dice = (dice * pw).sum() / num_masks
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    MM.Mask2FormerLoss.loss_labels = loss_labels
    MM.Mask2FormerLoss.loss_masks = loss_masks


def _weighted_compute_loss(self, model, inputs, return_outputs=False, **kw):
    weights = inputs.pop("inst_weights", None)
    crit = model.module.criterion if hasattr(model, "module") else model.criterion
    crit._tgt_weights = [w.to(model.device) for w in weights] if weights is not None else None
    outputs = model(**inputs)
    return (outputs.loss, outputs) if return_outputs else outputs.loss


class WeightedDiffLRTrainer(DiffLRTrainer):
    """DiffLR + per-instance loss weighting via criterion._tgt_weights."""
    compute_loss = _weighted_compute_loss


class WeightedTrainer(Trainer):
    compute_loss = _weighted_compute_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--exp", default="stageA")
    ap.add_argument("--num_labels", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--backbone_lr", type=float, default=0.0, help="0 = same as --lr")
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap train samples")
    ap.add_argument("--ckpt", default="facebook/mask2former-swin-large-coco-instance")
    ap.add_argument("--init_from", default="", help="Stage A best_model dir: load all "
                    "weights except the (size-mismatched) class head")
    ap.add_argument("--paste_csv", default="", help="tool cutout bank CSV "
                    "(coco/tool_cutouts/cutouts.csv); empty = ToolPaste off")
    ap.add_argument("--paste_p", type=float, default=0.5)
    ap.add_argument("--paste_max", type=int, default=2)
    ap.add_argument("--paste_power", type=float, default=0.5,
                    help="class sampling weight ~ train_count^(-power)")
    ap.add_argument("--paste_weight", type=float, default=1.0,
                    help="loss weight for pasted instances")
    ap.add_argument("--paste_classes", default="", help="comma-separated class ids to "
                    "paste (empty = all classes present in the bank)")
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), "..", "results", args.exp)
    os.makedirs(outdir, exist_ok=True)

    id2label = {i: f"class_{i}" for i in range(args.num_labels)}
    processor = Mask2FormerImageProcessor.from_pretrained(
        args.ckpt, do_resize=True, size={"shortest_edge": args.img, "longest_edge": args.img * 2},
        do_reduce_labels=False, ignore_index=0)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.ckpt, id2label=id2label, ignore_mismatched_sizes=True)

    if args.init_from:
        from safetensors.torch import load_file
        sd_path = os.path.join(args.init_from, "model.safetensors")
        src = load_file(sd_path)
        tgt = model.state_dict()
        copied, skipped = 0, 0
        new_sd = {}
        for k, v in tgt.items():
            if k in src and src[k].shape == v.shape:
                new_sd[k] = src[k]; copied += 1
            else:
                new_sd[k] = v; skipped += 1  # keep COCO-init for size-mismatched heads
        model.load_state_dict(new_sd, strict=True)
        LOG.info(f"init_from {sd_path}: copied {copied} tensors, kept {skipped} (head/mismatch)")

    paste = None
    if args.paste_csv:
        from collections import Counter
        from tool_paste_mb import ToolPasteMB
        tj = json.load(open(args.train))
        # fold leak safety: cutouts only from this fold's train videos
        allowed = {im["video"] for im in tj["images"] if "video" in im}
        counts = Counter(a["category_id"] for a in tj["annotations"])
        cls_ids = ([int(x) for x in args.paste_classes.split(",")]
                   if args.paste_classes else None)
        paste = ToolPasteMB(args.paste_csv, allowed_videos=allowed or None,
                            class_counts=counts, class_ids=cls_ids,
                            p=args.paste_p, max_tools=args.paste_max,
                            power=args.paste_power)
        per_cls = Counter(c["cls"] for c in paste.bank)
        LOG.info(f"ToolPaste: bank={len(paste.bank)} cutouts from "
                 f"{len({c['video'] for c in paste.bank})} train videos; "
                 f"per-class={dict(sorted(per_cls.items()))}; "
                 f"sampling_weights={dict(zip(paste.classes, [round(x, 3) for x in paste.cls_weights]))}")

    train_ds = CocoInstanceDS(args.train, processor, train=True, img=args.img,
                              paste=paste, paste_weight=args.paste_weight)
    val_ds = CocoInstanceDS(args.val, processor, train=False, img=args.img)
    if args.limit:
        train_ds.ids = train_ds.ids[:args.limit]
        val_ds.ids = val_ds.ids[:max(8, args.limit // 4)]
    LOG.info(f"train={len(train_ds)} val={len(val_ds)} num_labels={args.num_labels}")

    targs = TrainingArguments(
        output_dir=outdir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        fp16=(args.precision == "fp16"), bf16=(args.precision == "bf16"),
        dataloader_num_workers=args.workers,
        logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, remove_unused_columns=False, report_to="none",
        seed=42)
    _install_weighted_loss()  # per-instance loss weighting (identity when all weights==1.0)
    if args.backbone_lr > 0:
        WeightedDiffLRTrainer.backbone_lr = args.backbone_lr
        trainer = WeightedDiffLRTrainer(model=model, args=targs, train_dataset=train_ds,
                                        eval_dataset=val_ds, data_collator=collate)
    else:
        trainer = WeightedTrainer(model=model, args=targs, train_dataset=train_ds,
                                  eval_dataset=val_ds, data_collator=collate)
    trainer.train(resume_from_checkpoint=_last_ckpt(outdir))
    trainer.save_model(os.path.join(outdir, "best_model"))
    processor.save_pretrained(os.path.join(outdir, "best_model"))
    LOG.info("done")


def _last_ckpt(outdir):
    import glob
    cks = glob.glob(os.path.join(outdir, "checkpoint-*"))
    return sorted(cks, key=lambda p: int(p.split("-")[-1]))[-1] if cks else None


if __name__ == "__main__":
    main()
