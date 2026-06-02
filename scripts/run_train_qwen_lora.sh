#!/usr/bin/env bash
set -euo pipefail
python training/train_qwen_lora.py --config "${CONFIG:-configs/qwen2.5-coder-1.5b-qlora.yaml}"
