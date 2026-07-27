#!/usr/bin/env bash
# Run evaluation on the test set.
# Usage: bash scripts/run_eval.sh --checkpoint checkpoints/finetune/best_model.pth
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "===== Evaluation ====="
python evaluate.py \
    --config configs/finetune_config.yaml \
    "$@"
