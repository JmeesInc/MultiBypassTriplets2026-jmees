#!/bin/bash
# Stage 3 — base models whose OOF predictions form the ensemble teacher.
# (dense SPIRIT, dense Swin, keyframe SPIRIT), one per fold.
set -euo pipefail
. "$(dirname "$0")/env.sh"
for f in 0 1 2 3; do
  uv run python "$REPO/src/spirit/train_spirit_ft.py" --config "$REPO/configs/spirit/ensdistill_dense_f$f.yaml"
  uv run python "$REPO/src/swin/train_verb_fusion.py" --config "$REPO/configs/swin/ensdistill_dense_f$f.yaml"
done
echo "Then dump each model's val-fold predictions with --dump_oof <out.npz> --dump_split val."
