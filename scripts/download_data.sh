#!/usr/bin/env bash
# Download NIH Chest X-ray14 dataset from Kaggle.
#
# Prerequisites:
#   1. Install Kaggle CLI:  pip install kaggle
#   2. Place your API token at ~/.kaggle/kaggle.json
#      (Download from https://www.kaggle.com/settings -> API -> Create New Token)
#   3. Accept dataset terms at https://www.kaggle.com/datasets/nih-chest-xrays/data
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

DATA_DIR="data/raw"
DATASET_SLUG="nih-chest-xrays/data"

echo "===== NIH Chest X-ray14 Data Download ====="

# Check kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Check credentials exist
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'API' -> 'Create New Token'"
    echo "  3. Move the downloaded kaggle.json to ~/.kaggle/"
    exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"

mkdir -p "$DATA_DIR"

echo "Downloading dataset: $DATASET_SLUG"
echo "Target directory:    $DATA_DIR"
echo ""

kaggle datasets download \
    -d "$DATASET_SLUG" \
    --path "$DATA_DIR" \
    --unzip

echo ""
echo "Download complete."

# Normalize directory name
if [ -d "$DATA_DIR/data" ] && [ ! -d "$DATA_DIR/nih-chest-xrays" ]; then
    mv "$DATA_DIR/data" "$DATA_DIR/nih-chest-xrays"
fi

echo "Dataset location: $DATA_DIR/nih-chest-xrays"
echo ""
echo "Running preprocessing (building train/val/test splits)..."
python src/data/preprocess.py

echo ""
echo "Done! You can now run:"
echo "  bash scripts/run_pretrain.sh"
