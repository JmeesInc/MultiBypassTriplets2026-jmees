#!/bin/bash
# Stage 6 — 4-fold CV of the 2-student ensemble. Expect ~0.5181 videowise ivt mAP.
set -euo pipefail
. "$(dirname "$0")/env.sh"
for f in 0 1 2 3; do
  uv run python "$REPO/src/spirit/train_spirit_ft.py" --config "$REPO/configs/spirit/ensdistill_dense_f$f.yaml" \
    --dump_oof "$WORK/oof/spirit_ens_oof_f$f.npz" --dump_split val
  uv run python "$REPO/src/swin/train_verb_fusion.py" --config "$REPO/configs/swin/ensdistill_dense_f$f.yaml" \
    --dump_oof "$WORK/oof/swin_ens_oof_f$f.npz" --dump_split val
done
uv run python "$REPO/scripts/eval_ensemble.py" --oof_dir "$WORK/oof" --w_swin 0.3
