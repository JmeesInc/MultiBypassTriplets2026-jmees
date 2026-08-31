#!/bin/bash
# Stage 2 — cache front-end outputs for every annotated frame (stride 1 = dense).
# Out-of-fold: each video is encoded by the fold model that did NOT train on it, so the
# cache is leak-free for CV. Produces target_feat/target_logits/inst_presence/inst_mask.
set -euo pipefail
. "$(dirname "$0")/env.sh"
for f in 0 1 2 3; do
  vids=$(awk -F, -v ff="$f" 'NR>1 && $5==ff {printf "%s,",$1}' "$REPO/folds/folds.csv" | sed 's/,$//')
  uv run python "$REPO/src/swin/cache_frozen_feats.py" \
    --stride 1 --mask_res 28 --videos "$vids" \
    --target_ckpt "$WORK/target_convnext_f$f/best_model.pth" \
    --cfg2 "$WORK/cfg2_fold$f/best_model" \
    --out "$WORK/frozen_cache_oof_dense"
done
