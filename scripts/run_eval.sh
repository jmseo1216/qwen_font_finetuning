#!/usr/bin/env bash
set -euo pipefail
python evaluation/evaluate.py \
  --targets "${TARGETS:-dataset/processed/test.jsonl}" \
  --predictions "${PREDICTIONS:-predictions/test_predictions.jsonl}" \
  --output "${OUTPUT:-evaluation/results.json}"
