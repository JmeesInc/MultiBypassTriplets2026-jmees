"""Run a trained tool instance-seg model over MultiBypass frames.

Produces per-frame tool detections that feed the triplet recognizer:
  out/{video}.json = {frame_id: [{cls, score, bbox[xywh], rle}, ...]}
plus a compact per-frame tool-presence matrix out/{video}_presence.npy ([N_frames, num_labels]).

The presence/bbox/mask signals are intended as instrument-centric features for the
MultiBypass triplet (i / ivt) heads, or as pseudo-labels.

Usage:
  python src/infer_multibypass.py --model results/stageB/best_model --num_labels 13 \
      --videos C1V1,C2V6 --score_thr 0.5 --img 512
  (--videos all  for the whole dataset)
"""
import argparse
import glob
import json
import logging
import os

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("infer")

DATA = os.path.join(os.environ.get("MB_DATA", "data"), "videos")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "mbt_tool_preds")


@torch.no_grad()
def run_video(model, processor, vid, num_labels, score_thr, device, batch=8):
    frames = sorted(glob.glob(os.path.join(DATA, vid, "*.jpg")))
    preds = {}
    presence = np.zeros((len(frames), num_labels), dtype=np.float32)
    for s in range(0, len(frames), batch):
        chunk = frames[s:s + batch]
        imgs = [Image.open(f).convert("RGB") for f in chunk]
        sizes = [(im.size[1], im.size[0]) for im in imgs]  # (H,W)
        pv = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        if device == "cuda":
            pv = pv.half()
        out = model(pixel_values=pv)
        res = processor.post_process_instance_segmentation(
            out, target_sizes=sizes, threshold=score_thr, return_binary_maps=True)
        for j, r in enumerate(res):
            fid = int(os.path.splitext(os.path.basename(chunk[j]))[0])
            dets = []
            seg = r["segmentation"]
            for info in r["segments_info"]:
                cls, score = int(info["label_id"]), float(info["score"])
                presence[s + j, cls] = max(presence[s + j, cls], score)
                if isinstance(seg, torch.Tensor) and seg.dim() == 3:
                    m = seg[info["id"]].cpu().numpy().astype(np.uint8)
                else:
                    m = (seg.cpu().numpy() == info["id"]).astype(np.uint8)
                if m.sum() == 0:
                    continue
                ys, xs = np.where(m)
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                dets.append({"cls": cls, "score": round(score, 4),
                             "bbox": [int(xs.min()), int(ys.min()),
                                      int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                             "rle": rle})
            preds[fid] = dets
    return preds, presence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num_labels", type=int, default=13)
    ap.add_argument("--videos", default="all")
    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Mask2FormerImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()

    if args.videos == "all":
        vids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, "*")) if os.path.isdir(p))
    else:
        vids = args.videos.split(",")

    for vid in vids:
        preds, presence = run_video(model, processor, vid, args.num_labels,
                                    args.score_thr, device, args.batch)
        json.dump(preds, open(os.path.join(OUT, f"{vid}.json"), "w"))
        np.save(os.path.join(OUT, f"{vid}_presence.npy"), presence)
        ndet = sum(len(v) for v in preds.values())
        LOG.info(f"{vid}: {len(preds)} frames, {ndet} detections -> {OUT}/{vid}.json")


if __name__ == "__main__":
    main()
