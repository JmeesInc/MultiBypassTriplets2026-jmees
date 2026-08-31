#!/bin/bash
# Local regression test: runs the container against a small slice of real data and
# checks the output contract (85 scores per frame, values in [0,1], all frames present).
set -e
cd "$(dirname "$0")"
TD=${1:?usage: test.sh <data_dir_with_MultiBypass-4C-T40> [out_dir]}
OUT=${2:-$(pwd)/output_localtest}
mkdir -p "$OUT"
# GPU access: this host has nvidia as the DEFAULT docker runtime, so --gpus is not
# used (it fails with "failed to discover GPU vendor from CDI"). Override with
# MB_GPU_FLAGS if your host needs --gpus all.
docker run --rm ${MB_GPU_FLAGS:-} -e NVIDIA_VISIBLE_DEVICES=${MB_GPU:-0} \
  -v "$TD":/data:ro -v "$OUT":/results \
  mbt2026_v002
python3 - "$OUT/multibypass_triplet_predictions.json" << 'PY'
import json, sys
p = json.load(open(sys.argv[1]))
n = sum(len(v) for v in p.values())
bad = [(v, f) for v, fr in p.items() for f, s in fr.items()
       if len(s) != 85 or not all(0.0 <= x <= 1.0 for x in s)]
print(f"videos={len(p)} frames={n} malformed={len(bad)}")
assert not bad, bad[:5]
print("OUTPUT CONTRACT OK")
PY
