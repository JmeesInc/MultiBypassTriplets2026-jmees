#!/bin/bash
# Build the submission ZIP (naming: <team><CC>_<initials>_<date>_<version>.zip).
# Usage: bash export.sh Jmees_JP_SK_2026-08-31_v2
#
# .torch/ MUST be included: SwinFusion.__init__ calls
# swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1), which downloads from
# download.pytorch.org unless the checkpoint is already cached under $TORCH_HOME
# (=/app/.torch, set in the Dockerfile). The eval server runs offline, so a missing
# cache fails the whole run — verified with `docker run --network none`.
#
# -0 (store, no deflate): the payload is .pth / .safetensors, already compressed, so
# deflating costs many minutes and saves almost nothing.
# The superseded split SPIRIT checkpoints (shared_lower / upper_f*) are excluded —
# the training-code path loads the full per-fold f*.pth instead.
set -e
cd "$(dirname "$0")"
NAME=${1:?usage: export.sh <team_CC_initials_date_version>}
rm -f "../${NAME}.zip"
zip -r -0 -q "../${NAME}.zip" Dockerfile main.py src dinov3 checkpoints .torch \
  -x '*__pycache__*' -x '*.pyc' \
  -x 'checkpoints/spirit/shared_lower.pth' -x 'checkpoints/spirit/upper_f*.pth'
echo "created ../${NAME}.zip"
du -sh "../${NAME}.zip"
