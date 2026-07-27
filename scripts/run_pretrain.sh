#!/usr/bin/env bash
# Launch SimCLR pre-training.
# Usage: bash scripts/run_pretrain.sh [--epochs 100] [--batch_size 256]
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "===== SimCLR Pre-training ====="
python train_pretrain.py \
    --config configs/pretrain_config.yaml \
    "$@"
