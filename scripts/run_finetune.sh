#!/usr/bin/env bash
# Launch fine-tuning / linear probe.
# Usage: bash scripts/run_finetune.sh [--mode full_finetune|linear_probe|imagenet_baseline]
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "===== Fine-tuning / Classification ====="
python train_finetune.py \
    --config configs/finetune_config.yaml \
    "$@"
