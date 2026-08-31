#!/bin/bash
# Stage 1 — frozen front-end models used by the Swin branch (per fold, out-of-fold).
#   * ConvNeXt-B target model  (15 target classes)
#   * Mask2Former cfg2 instrument model (12 instruments)
# The SPIRIT branch does not use either of these.
set -euo pipefail
. "$(dirname "$0")/env.sh"
for f in 0 1 2 3; do
  uv run python "$REPO/src/frontend/train_target.py" \
    --config "$REPO/configs/frontend/target_convnext.yaml" --val_fold "$f" \
    --out "$WORK/target_convnext_f$f"
done
echo "Mask2Former (instrument) training: see src/frontend/train_mask2former.py."
echo "It consumes the staged COCO-style dataset built by src/frontend/build_stage.py."
