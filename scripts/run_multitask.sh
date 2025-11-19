#!/bin/bash
# Stage 2: Multi-Task Fine-Tuning
# Fine-tune encoder + χ head and train solubility head

set -e  # Exit on error

# Smart argument detection:
# - If arg 1 ends with .pt, treat it as pretrained model path
#   Usage: bash run_multitask.sh results/dft_pretrain_*/checkpoints/best_model.pt [config] [device]
# - Otherwise, use standard argument order
#   Usage: bash run_multitask.sh [config] [pretrained_path] [device]

if [[ "$1" == *.pt ]]; then
    # First argument is a .pt file - treat as pretrained model
    PRETRAINED_PATH="$1"
    CONFIG_PATH="${2:-configs/config.yaml}"
    DEVICE="${3:-cuda}"
else
    # Standard argument order
    CONFIG_PATH="${1:-configs/config.yaml}"
    PRETRAINED_PATH="$2"
    DEVICE="${3:-cuda}"
fi

# If pretrained path not provided, find most recent DFT pretrain model
if [ -z "$PRETRAINED_PATH" ]; then
    echo "No pretrained model path provided. Searching for most recent DFT pretrain model..."

    # Find most recent dft_pretrain_* directory
    LATEST_DFT_DIR=$(ls -dt results/dft_pretrain_* 2>/dev/null | head -n 1)

    if [ -z "$LATEST_DFT_DIR" ]; then
        echo "ERROR: No DFT pretrain results found in results/ directory!"
        echo "Please run Stage 1 (DFT pretraining) first:"
        echo "  bash scripts/run_pretrain_dft.sh"
        exit 1
    fi

    PRETRAINED_PATH="$LATEST_DFT_DIR/checkpoints/best_model.pt"
    echo "Found: $PRETRAINED_PATH"
fi

echo "=================================================="
echo "Stage 2: Multi-Task Fine-Tuning"
echo "=================================================="
echo "Config: $CONFIG_PATH"
echo "Pretrained model: $PRETRAINED_PATH"
echo "Device: $DEVICE"
echo ""

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained model not found at: $PRETRAINED_PATH"
    echo ""
    echo "Usage: $0 [config_path] [pretrained_model_path] [device]"
    echo ""
    echo "Example:"
    echo "  $0 configs/config.yaml results/dft_pretrain_20251119_032826/checkpoints/best_model.pt cuda"
    exit 1
fi

# Run training
python -m src.training.train_multitask \
    --config "$CONFIG_PATH" \
    --pretrained "$PRETRAINED_PATH" \
    --device "$DEVICE"

echo ""
echo "Multi-task training completed!"
echo "Check results/ directory for outputs."
