"""Step 2a: cache FROZEN branch features for the triplet pipeline.

Per keyframe (stride-sampled), run the frozen TARGET image-model (convnext) and the frozen
INSTRUMENT seg (cfg2 Mask2Former) and store:
  target_feat   [1024]   convnext pooled pre-logits feature
  target_logits [15]     target class logits
  inst_presence [12]     per-instrument max score (0 if absent)
  inst_mask     [12,M,M] per-instrument low-res mask (max over instances), M=--mask_res

Output per video: coco... -> results/frozen_cache/{video}.npz  (fids + the 4 arrays).
Later loaded by the Swin3D fusion model (keyframe features + mask spatial prior).

Usage: CUDA_VISIBLE_DEVICES=0 python src/cache_frozen_feats.py --stride 3 --mask_res 28
"""
import argparse
import glob
import os

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

REPO = os.environ.get("MB_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VID_DIR = os.environ.get("MB_VID_DIR", os.path.join(os.environ.get("MB_DATA", "data"), "videos"))
HERE = os.path.dirname(__file__)
TARGET_CKPT = os.path.join(HERE, "..", "results", "target_convnext_b_v1", "fold0", "best_model.pth")
CFG2 = os.path.join(REPO, "workspace/expB00_toolseg/results/phase3_cfg2_p1full_p2clean_fold0/best_model")
OUT = os.path.join(HERE, "..", "results", "frozen_cache")
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w); nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh))
    c = np.zeros((size, size, 3), np.uint8)
    c[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = r
    return c


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--mask_res", type=int, default=28)
    ap.add_argument("--timg", type=int, default=384, help="target model input size")
    ap.add_argument("--simg", type=int, default=512, help="cfg2 input size")
    ap.add_argument("--videos", default="all")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--target_ckpt", default=TARGET_CKPT, help="target model checkpoint (backbone+15head)")
    ap.add_argument("--cfg2", default=CFG2, help="cfg2 Mask2Former model dir (per-fold for OOF cache)")
    ap.add_argument("--out", default=OUT, help="output cache dir")
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"

    # target model
    tck = torch.load(args.target_ckpt, map_location=device)
    tmodel = timm.create_model(tck["backbone"], pretrained=False, num_classes=15).to(device).eval()
    tmodel.load_state_dict(tck["model"]); tmodel = tmodel.half()
    # cfg2
    proc = Mask2FormerImageProcessor.from_pretrained(args.cfg2)
    smodel = Mask2FormerForUniversalSegmentation.from_pretrained(args.cfg2).to(device).eval().half()
    M = args.mask_res

    vids = (sorted(os.path.basename(v) for v in glob.glob(os.path.join(VID_DIR, "*")))
            if args.videos == "all" else args.videos.split(","))
    for vid in vids:
        outp = os.path.join(out_dir, f"{vid}.npz")
        if os.path.exists(outp):
            print(f"[{vid}] exists, skip", flush=True); continue
        frames = sorted(glob.glob(os.path.join(VID_DIR, vid, "*.jpg")))[::args.stride]
        fids = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames]
        tf, tl, ip, im_ = [], [], [], []
        for bs in range(0, len(frames), args.batch):
            chunk = frames[bs:bs + args.batch]
            raw = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in chunk]
            # target model
            tb = []
            for r in raw:
                x = letterbox(r, args.timg).astype(np.float32) / 255.0
                x = (x - IMNET_MEAN) / IMNET_STD
                tb.append(torch.from_numpy(x.transpose(2, 0, 1)))
            tb = torch.stack(tb).to(device).half()
            feats = tmodel.forward_features(tb)
            pooled = tmodel.forward_head(feats, pre_logits=True)   # [B,1024]
            logits = tmodel.forward_head(feats)                    # [B,15]
            tf.append(pooled.float().cpu().numpy()); tl.append(logits.float().cpu().numpy())
            # cfg2 instrument
            pil = [Image.fromarray(r) for r in raw]
            pv = proc(images=pil, return_tensors="pt")["pixel_values"].to(device).half()
            out = smodel(pixel_values=pv)
            sizes = [(r.shape[0], r.shape[1]) for r in raw]
            res = proc.post_process_instance_segmentation(out, target_sizes=sizes,
                                                          threshold=0.5, return_binary_maps=True)
            for r in res:
                seg = r["segmentation"]; pres = np.zeros(12, np.float32); mask = np.zeros((12, M, M), np.float32)
                for info in r["segments_info"]:
                    c = int(info["label_id"]); sc = float(info["score"])
                    if c >= 12:
                        continue
                    pres[c] = max(pres[c], sc)
                    m = (seg[info["id"]].cpu().numpy().astype(np.uint8) if seg.dim() == 3
                         else (seg.cpu().numpy() == info["id"]).astype(np.uint8))
                    mm = cv2.resize(m, (M, M), interpolation=cv2.INTER_AREA)
                    mask[c] = np.maximum(mask[c], mm)
                ip.append(pres); im_.append(mask)
            if bs % (args.batch * 50) == 0:
                print(f"[{vid}] {bs}/{len(frames)}", flush=True)
        np.savez_compressed(outp, fids=np.array(fids, np.int32),
                            target_feat=np.concatenate(tf).astype(np.float16),
                            target_logits=np.concatenate(tl).astype(np.float16),
                            inst_presence=np.stack(ip).astype(np.float16),
                            inst_mask=np.stack(im_).astype(np.float16))
        print(f"[{vid}] cached {len(fids)} keyframes -> {outp}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
