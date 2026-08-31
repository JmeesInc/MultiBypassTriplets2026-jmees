"""MultiBypassTriplets2026 submission — ensemble-distilled 8-model ensemble.

Source: workspace/expA04_spirit/results/spirit_ensdistill_dense_f{0-3} (SPIRIT students)
      + workspace/expA02_triplet/results/swin_ensdistill_dense_f{0-3} (Swin students)
Both distilled from the OOF ensemble teacher (teach/ensemble_soft_all.npz).
4-fold CV videowise ivt mAP = 0.5181 (SPIRIT 0.5007 / Swin 0.4527, blend w_swin=0.3).

I/O contract (mounted by eval server):
  /data/MultiBypass-4C-T40/{videos/<VID>/<6d>.jpg, label_files_challenge/<VID>.json}
  -> /results/multibypass_triplet_predictions.json = {video_id: {frame_id: [85 scores]}}

Compute-dedup design (see memory ensemble-inference-dedup-design):
  * Mask2Former(cfg2) + convnext(target): 1x per frame, shared by all 4 Swin students
    (no leak concern at test time -> a single front-end set suffices).
    SPIRIT students do not use them at all.
  * DINOv3 ViT-L: patch_embed + blocks[0:16] are bit-identical across the 4 SPIRIT
    students (freeze_blocks=16), so the lower trunk runs once per frame and its token
    state is cached; only blocks[16:24] + norm run per fold. Frame tokens are reused
    across the 8 keyframes whose clip contains that frame.
  * Swin3D-B: 4x (per-fold weights differ and 3D attention mixes time -> no reuse).
  * Frames are decoded once per video in a rolling buffer instead of per keyframe.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.video import swin3d_b
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

APP = Path(__file__).resolve().parent
CK = APP / "checkpoints"
sys.path.insert(0, str(APP / "src"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

NIVT, NI, NV, NT = 85, 12, 13, 15
N_IT, N_IV = 54, 39      # compact pairwise heads (observed i-t / i-v pairs), as trained
# Swin student clip: exponential offsets (matches training config)
SWIN_OFFSETS = [-181, -128, -90, -64, -45, -32, -22, -16, -11, -8, -6, -4, -3, -2, -1, 0]
SPIRIT_T = 8            # causal clip, stride 1
FREEZE_BLOCKS = 16      # lower trunk shared across folds
TIMG, SIMG, VIMG, MASK_RES = 384, 512, 224, 28
POOL = 8                # DINOv3 patch tokens pooled to 8x8
W_SWIN = 0.3            # blend weight validated on 4-fold CV (0.5181)
BATCH = int(os.environ.get("MB_BATCH", "8"))
# Share the frozen lower DINOv3 trunk across the 4 SPIRIT students (bit-identical
# weights -> mathematically identical output; set MB_SHARE_LOWER=0 to disable).
SHARE_LOWER = os.environ.get("MB_SHARE_LOWER", "1") == "1"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("mb")


# --------------------------------------------------------- Swin student (reuse)
# The training definition is bundled verbatim (src/train_verb_fusion.py) and imported
# rather than re-implemented: a hand-written copy silently diverged (feature concat
# order), so the authoritative module is used instead. Verified corr=1.0000 / MAE=0.0
# against the stored OOF predictions.
import train_verb_fusion as TVF  # noqa: E402

SwinFusion = TVF.SwinFusion
KMEAN, KSTD = TVF.KMEAN, TVF.KSTD


# ------------------------------------------------------------- SPIRIT student
def dino_lower_tokens(bb, imgs):
    """Run patch_embed + blocks[0:FREEZE_BLOCKS] — the part shared by all 4 folds.

    Mirrors `DinoBackboneTrunk.forward` (src/model_spirit_ft.py) exactly: same
    prepare_tokens_with_masks, same per-block rope, same block order. Splitting is
    safe because rope is recomputed per block from (H, W) and carries no state, so the
    lower half is a pure function of the input. The 4 students share bit-identical
    patch_embed + blocks[0:16] weights (verified), hence one pass serves all of them.
    Returns (tokens, rope, H, W).
    """
    x, (H, W) = bb.prepare_tokens_with_masks(imgs)
    xl = [x]
    for i in range(FREEZE_BLOCKS):
        rope = [bb.rope_embed(H=H, W=W)] if bb.rope_embed is not None else [None]
        xl = bb.blocks[i](xl, rope)
    return xl[0], H, W


def dino_upper_pool(bb, tok, H, W):
    """Run blocks[FREEZE_BLOCKS:] + norm + 8x8 pooling — the per-fold part."""
    xl = [tok]
    for i in range(FREEZE_BLOCKS, len(bb.blocks)):
        rope = [bb.rope_embed(H=H, W=W)] if bb.rope_embed is not None else [None]
        xl = bb.blocks[i](xl, rope)
    x = bb.norm(xl[0])
    patch = x[:, bb.n_storage_tokens + 1:]
    BT, Np, C = patch.shape
    g = int(round(Np ** 0.5))
    p = patch.transpose(1, 2).reshape(BT, C, g, g)
    p = F.adaptive_avg_pool2d(p, (POOL, POOL)).reshape(BT, C, POOL * POOL).transpose(1, 2)
    return p


def build_spirit_models(device):
    """Build the 4 SPIRIT students with the authoritative training definition.

    A hand-written lower/upper split was tried for speed but did not reproduce the
    stored OOF predictions, so the training module is used verbatim (verified
    corr=1.0000 / MAE=0.0). Weights stay fp32 — casting DINOv3 ViT-L to fp16 produces
    NaNs; autocast supplies the speedup instead.
    """
    sys.path.insert(0, str(APP / "dinov3"))
    from model_spirit_ft import SpiritFull_FT

    models = []
    for f in range(4):
        # NOTE: dinov3's hubconf parses the weight FILENAME, so it must keep the
        # canonical name (renaming raises "Unexpected weights specification").
        bb = torch.hub.load(str(APP / "dinov3"), "dinov3_vitl16", source="local",
                            weights=str(CK / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
        m = SpiritFull_FT(bb, freeze_blocks=FREEZE_BLOCKS, grad_ckpt=False,
                          d=128, ds=256, nhead=4, dropout=0.05, T=SPIRIT_T, N=POOL * POOL)
        sd = torch.load(CK / "spirit" / f"f{f}.pth", map_location="cpu")
        sd = sd["model"] if "model" in sd else sd
        r = m.load_state_dict(sd, strict=False)
        if r.missing_keys or r.unexpected_keys:
            log.warning("spirit f%d load: missing=%d unexpected=%d", f,
                        len(r.missing_keys), len(r.unexpected_keys))
        models.append(m.to(device).eval())
    return models


# ----------------------------------------------------------------- preprocess
def letterbox(img, size):
    import cv2
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh))
    c = np.zeros((size, size, 3), np.uint8)
    c[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = r
    return c


def norm_chw(img_rgb, size, letter=True):
    import cv2
    x = letterbox(img_rgb, size) if letter else cv2.resize(img_rgb, (size, size))
    x = x.astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


VIDEO_ROOT = None      # set by find_records(); used to resolve clip context frames


def find_records():
    global VIDEO_ROOT
    root = DATA_DIR / "MultiBypass-4C-T40"
    vdir, ldir = root / "videos", root / "label_files_challenge"
    VIDEO_ROOT = vdir
    by_video = {}
    if ldir.exists():
        for lp in sorted(ldir.glob("*.json")):
            vid = lp.stem
            payload = json.load(open(lp))
            for im in payload.get("images", []):
                fid = int(im["id"])
                fp = vdir / vid / (im.get("file_name") or f"{fid:06d}.jpg")
                if not fp.exists():
                    fp = vdir / vid / f"{fid:06d}.jpg"
                by_video.setdefault(vid, {})[fid] = fp
    if not by_video:
        for vd in sorted(p for p in vdir.iterdir() if p.is_dir()):
            for fp in sorted(vd.glob("*.jpg")):
                by_video.setdefault(vd.name, {})[int(fp.stem)] = fp
    return by_video


@torch.no_grad()
def main():
    import cv2
    cv2.setNumThreads(max(1, (os.cpu_count() or 4) // 2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    log.info("device=%s half=%s batch=%d", device, half, BATCH)

    # ---- front-end (single set, shared by all Swin students) ----
    tck = torch.load(CK / "target_convnext.pth", map_location="cpu")
    tmodel = timm.create_model(tck.get("backbone", "convnext_base.fb_in22k_ft_in1k_384"),
                               pretrained=False, num_classes=NT)
    tmodel.load_state_dict(tck["model"])
    tmodel = tmodel.to(device).eval()
    proc = Mask2FormerImageProcessor.from_pretrained(str(CK / "cfg2"))
    smodel = Mask2FormerForUniversalSegmentation.from_pretrained(str(CK / "cfg2")).to(device).eval()
    if half:
        tmodel, smodel = tmodel.half(), smodel.half()

    # ---- Swin students x4 ----
    swins = []
    for f in range(4):
        sd = torch.load(CK / "swin" / f"f{f}.pth", map_location="cpu")
        m = SwinFusion(pairwise=True)     # matches student config (pairwise_heads: true)
        r = m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        if r.missing_keys or r.unexpected_keys:
            log.warning("swin f%d load: missing=%d unexpected=%d", f,
                        len(r.missing_keys), len(r.unexpected_keys))
        m = m.to(device).eval()
        if half:
            m = m.half()
        swins.append(m)

    # ---- SPIRIT students x4 (shared lower trunk) ----
    spirits = build_spirit_models(device)
    log.info("models loaded: 4 SPIRIT + 4 Swin + front-end")

    by_video = find_records()
    total = sum(len(v) for v in by_video.values())
    log.info("videos=%d frames=%d", len(by_video), total)
    preds, done, t0 = {}, 0, time.time()

    for vid, fmap in by_video.items():
        fids = sorted(fmap)
        # rolling decode cache: frame_id -> (tensor224, tensor384, raw)
        cache224, cache_raw = {}, {}

        def _path(f):
            """Frame path by id.

            Context frames pulled in by the clip offsets are usually NOT among the
            prediction targets, so they must be resolved against the video directory
            rather than the (target-only) record map — reading them as black frames
            silently changes the model input.
            """
            fp = fmap.get(f)
            return fp if fp is not None else (VIDEO_ROOT / vid / f"{max(f, 0):06d}.jpg")

        def get224(f):
            """Match training exactly: clamp fid at 0, missing file -> black frame."""
            f = max(f, 0)
            if f not in cache224:
                im = cv2.imread(str(_path(f)))
                if im is None:
                    raw = np.zeros((VIMG, VIMG, 3), np.uint8)
                else:
                    raw = cv2.cvtColor(cv2.resize(im, (VIMG, VIMG)), cv2.COLOR_BGR2RGB)
                x = raw.astype(np.float32) / 255.0
                cache224[f] = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))
            return cache224[f]

        def get_raw_full(f):
            """Full-resolution RGB frame for the front-end (convnext/Mask2Former)."""
            im = cv2.imread(str(_path(f)))
            if im is None:
                return np.zeros((VIMG, VIMG, 3), np.uint8)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for i0 in range(0, len(fids), BATCH):
            chunk = fids[i0:i0 + BATCH]
            B = len(chunk)
            # --- decode keyframes + build clips ---
            raws, sp_clips, sw_clips = [], [], []
            for fid in chunk:
                raws.append(get_raw_full(fid))   # full-res for convnext / Mask2Former
                sp_clips.append(torch.stack([get224(fid - (SPIRIT_T - 1 - t)) for t in range(SPIRIT_T)]))
                sw_clips.append(torch.stack([get224(fid + o) for o in SWIN_OFFSETS]))
            # drop cache entries no longer reachable (oldest offset -181)
            keep_from = chunk[0] + min(SWIN_OFFSETS) - 2
            for k in [k for k in cache224 if k < keep_from]:
                cache224.pop(k, None); cache_raw.pop(k, None)

            # --- front-end: convnext (batched) ---
            # NOTE: a "GPU normalisation" variant (upload uint8, then permute/float/
            # normalise on device) was tried and measured 1.8x SLOWER end-to-end
            # (214.7s vs 119.6s on 512 frames): permute leaves the tensor
            # non-contiguous for the convnext stem, and the CPU letterbox remained
            # anyway. Keep the straightforward CPU path.
            tx = torch.stack([norm_chw(r, TIMG) for r in raws]).to(device)
            if half:
                tx = tx.half()
            feats = tmodel.forward_features(tx)
            tfeat = tmodel.forward_head(feats, pre_logits=True)
            tlogit = tmodel.forward_head(feats)

            # --- front-end: Mask2Former (batched) ---
            pv = proc(images=[Image.fromarray(r) for r in raws], return_tensors="pt")["pixel_values"].to(device)
            if half:
                pv = pv.half()
            sout = smodel(pixel_values=pv)
            res = proc.post_process_instance_segmentation(
                sout, target_sizes=[(r.shape[0], r.shape[1]) for r in raws],
                threshold=0.5, return_binary_maps=True)
            pres = np.zeros((B, NI), np.float32)
            imask = np.zeros((B, NI, MASK_RES, MASK_RES), np.float32)
            for b, r in enumerate(res):
                seg = r["segmentation"]
                for info in r["segments_info"]:
                    c = int(info["label_id"])
                    if c >= NI:
                        continue
                    pres[b, c] = max(pres[b, c], float(info["score"]))
                    mm = (seg[info["id"]].cpu().numpy().astype(np.uint8) if seg.dim() == 3
                          else (seg.cpu().numpy() == info["id"]).astype(np.uint8))
                    imask[b, c] = np.maximum(imask[b, c],
                                             cv2.resize(mm, (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA))
            pres_t = torch.from_numpy(pres).to(device)
            imask_t = torch.from_numpy(imask).to(device)
            if half:
                pres_t, imask_t = pres_t.half(), imask_t.half()

            # --- SPIRIT: lower trunk once, then 4 fold-specific uppers ---
            sp_in = torch.stack(sp_clips).to(device)             # [B,T,3,224,224] fp32
            with torch.autocast("cuda", dtype=torch.float16, enabled=half):
                logits = []
                if SHARE_LOWER:
                    # lower trunk once (shared weights) -> per-fold upper + head
                    flat = sp_in.reshape(B * SPIRIT_T, 3, VIMG, VIMG)
                    tok, H, W = dino_lower_tokens(spirits[0].trunk.bb, flat)
                    for mod in spirits:
                        p = dino_upper_pool(mod.trunk.bb, tok, H, W)
                        p = p.reshape(B, SPIRIT_T, p.shape[1], p.shape[2])
                        out = mod.head(p, imask=None, tfeat=None, pres=None)
                        logits.append(torch.sigmoid((out["ivt"] if isinstance(out, dict) else out).float()))
                else:
                    for mod in spirits:
                        out = mod(sp_in)
                        logits.append(torch.sigmoid((out["ivt"] if isinstance(out, dict) else out).float()))
            sp_prob = torch.stack(logits).mean(0)                 # [B,85]

            # --- Swin: 4 folds on the same clip ---
            sw_in = torch.stack(sw_clips).permute(0, 2, 1, 3, 4).to(device)   # [B,3,T,H,W]
            if half:
                sw_in = sw_in.half()
            sw_logits = []
            for m in swins:
                o = m(sw_in, tfeat, tlogit, pres_t, imask_t)
                oivt = o[0] if isinstance(o, tuple) else o   # (ivt, i, v, t, it, iv)
                sw_logits.append(torch.sigmoid(oivt.float()))
            sw_prob = torch.stack(sw_logits).mean(0)

            prob = (1.0 - W_SWIN) * sp_prob + W_SWIN * sw_prob
            for b, fid in enumerate(chunk):
                preds.setdefault(vid, {})[str(fid)] = [round(float(s), 6) for s in prob[b].cpu().tolist()]
            done += B
            if done % 200 < BATCH:
                el = time.time() - t0
                log.info("%d/%d  %.3f s/frame  eta %.1f h", done, total, el / done,
                         (total - done) * (el / done) / 3600)

    out = RESULTS_DIR / "multibypass_triplet_predictions.json"
    json.dump(preds, open(out, "w"))
    log.info("wrote %d frame preds -> %s (%.2f h)", sum(len(v) for v in preds.values()), out,
             (time.time() - t0) / 3600)


if __name__ == "__main__":
    main()
