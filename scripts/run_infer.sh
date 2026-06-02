#!/usr/bin/env bash
set -euo pipefail
: "${INPUT_SVG:?Set INPUT_SVG to an outline SVG file}"
python inference/generate.py --input_svg "$INPUT_SVG" \
  --adapter_dir "${ADAPTER_DIR:-checkpoints/qwen2.5-coder-1.5b-qlora}" \
  --output "${OUTPUT:-predicted_skeleton.svg}" "${@}"
