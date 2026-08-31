#!/bin/bash
# Stage 4 — build the out-of-fold ensemble teacher.
# For every fold g, ensemble the three base models that held g out, then concatenate the
# folds so the teacher covers all frames without ever having trained on the frame it labels.
# Fixed weights (dSPIRIT 1.0 / dSwin 0.4 / kfSPIRIT 0.7) -> CV 0.492.
set -euo pipefail
. "$(dirname "$0")/env.sh"
uv run python "$REPO/scripts/build_teacher.py" \
  --oof_dir "$WORK/oof" --out "$WORK/teach/ensemble_soft_all.npz" \
  --w_dspirit 1.0 --w_dswin 0.4 --w_kfspirit 0.7
