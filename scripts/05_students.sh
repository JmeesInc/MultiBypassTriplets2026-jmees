#!/bin/bash
# Stage 5 — the submitted students: 4 SPIRIT + 4 Swin, dense, distilled from the teacher.
# ~10 epochs each; the distillation loss and its TS_n/TG_n sample weighting live in the
# trainers (see `distill:` in each config).
set -euo pipefail
. "$(dirname "$0")/env.sh"
for f in 0 1 2 3; do
  uv run python "$REPO/src/spirit/train_spirit_ft.py"  --config "$REPO/configs/spirit/ensdistill_dense_f$f.yaml"
  uv run python "$REPO/src/swin/train_verb_fusion.py"  --config "$REPO/configs/swin/ensdistill_dense_f$f.yaml"
done
