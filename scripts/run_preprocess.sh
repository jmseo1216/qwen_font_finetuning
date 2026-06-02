#!/usr/bin/env bash
set -euo pipefail
python preprocess/build_dataset.py \
  --outline_dir "${OUTLINE_DIR:-data/raw/outline}" \
  --skeleton_dir "${SKELETON_DIR:-data/raw/skeleton}" \
  --out_dir "${OUT_DIR:-dataset/processed}" \
  --split_mode "${SPLIT_MODE:-font_glyph}"
