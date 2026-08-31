#!/bin/bash
# Shared paths. Source this (or export the same variables) before running any stage.
#
#   MB_DATA : challenge data root, containing videos/<VID>/<6d>.jpg and label_files_challenge/<VID>.json
#   REPO    : this repository
#   WORK    : scratch space for caches, checkpoints and teacher soft labels (needs ~200 GB)
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MB_DATA="${MB_DATA:?set MB_DATA to the MultiBypass-4C-T40 root}"
export WORK="${WORK:-$REPO/work}"
# the training scripts read the data through these:
export MB_REPO="$REPO"
export MB_VID_DIR="$MB_DATA/videos"
export MB_LBL_DIR="$MB_DATA/label_files_challenge"
# DINOv3 backbone (SPIRIT branch). Keep the upstream filename — hubconf parses it.
export DINOV3_REPO="${DINOV3_REPO:-$REPO/third_party/dinov3}"
export VITL_WEIGHTS="${VITL_WEIGHTS:-$WORK/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
mkdir -p "$WORK"/{weights,teach,frozen_cache_oof_dense}
