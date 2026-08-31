"""Pool normalized per-dataset COCO files into a Stage A / Stage B training COCO.

Stage A: collapse all categories -> single 'instrument' class (class-agnostic).
Stage B: keep unified 13-class taxonomy.
Split is VIDEO-LEVEL (GroupKFold-style) to prevent frame leakage.

Usage: python src/build_stage.py --stage A --val_frac 0.12 --seed 42
"""
import argparse
import glob
import json
import os
import random

NORM_DIR = os.path.join(os.path.dirname(__file__), "..", "coco", "normalized")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "coco")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import taxonomy as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources", nargs="*", default=[],
                    help="substrings to match normalized filenames (default=all). "
                         "e.g. phakir surgtoolloc cholecinstanceseg")
    ap.add_argument("--tag", default="",
                    help="output prefix (default stage{A,B}); files {tag}_train.json / {tag}_val.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(NORM_DIR, "*.json")))
    if args.sources:
        files = [f for f in files if any(s in os.path.basename(f) for s in args.sources)]
    assert files, "no normalized coco files matched; run convert.py first / check --sources"
    print("pooling files:", [os.path.basename(f) for f in files])

    images, anns = [], []
    next_img, next_ann = 0, 0
    video_set = set()
    vid2src = {}
    for fp in files:
        d = json.load(open(fp))
        remap = {}
        for im in d["images"]:
            remap[im["id"]] = next_img
            rec = dict(im); rec["id"] = next_img
            images.append(rec); video_set.add(im["video"])
            vid2src[im["video"]] = im.get("source", "?"); next_img += 1
        for a in d["annotations"]:
            if a["image_id"] not in remap:
                continue
            # category_id < 0 = generic/untyped tool: kept in Stage A, dropped in Stage B
            if args.stage == "B" and a["category_id"] < 0:
                continue
            cat = 0 if args.stage == "A" else a["category_id"]
            anns.append({"id": next_ann, "image_id": remap[a["image_id"]],
                         "category_id": cat, "segmentation": a["segmentation"],
                         "bbox": a["bbox"], "area": a.get("area", 0),
                         "iscrowd": a.get("iscrowd", 0)})
            next_ann += 1

    # video-level split, STRATIFIED per source (each domain represented in val)
    from collections import defaultdict
    rng = random.Random(args.seed)
    src2vids = defaultdict(list)
    for v in sorted(video_set):
        src2vids[vid2src[v]].append(v)
    val_vids = set()
    for src, vs in sorted(src2vids.items()):
        vs = list(vs); rng.shuffle(vs)
        n_val = max(1, int(round(len(vs) * args.val_frac)))
        val_vids.update(vs[:n_val])
    vids = sorted(video_set)

    if args.stage == "A":
        cats = [{"id": 0, "name": "instrument"}]
    else:
        cats = [{"id": i, "name": n} for i, n in enumerate(T.UNIFIED)]

    def subset(is_val):
        sel_imgs = [im for im in images if (im["video"] in val_vids) == is_val]
        ids = {im["id"] for im in sel_imgs}
        sel_anns = [a for a in anns if a["image_id"] in ids]
        return sel_imgs, sel_anns

    tag = args.tag or f"stage{args.stage}"
    from collections import Counter
    for split, is_val in [("train", False), ("val", True)]:
        si, sa = subset(is_val)
        out = {"categories": cats, "images": si, "annotations": sa}
        path = os.path.join(OUT_DIR, f"{tag}_{split}.json")
        json.dump(out, open(path, "w"))
        srcc = Counter(im["source"] for im in si)
        catc = Counter(a["category_id"] for a in sa)
        print(f"[{tag}/{split}] videos:{len([v for v in vids if (v in val_vids)==is_val])} "
              f"images={len(si)} anns={len(sa)} sources={dict(srcc)}")
        print(f"    cat dist: {dict(sorted(catc.items()))}")
    print(f"val videos ({len(val_vids)}): {sorted(val_vids)}")


if __name__ == "__main__":
    main()
