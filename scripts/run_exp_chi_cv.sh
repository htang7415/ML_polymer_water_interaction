#!/bin/bash
# Experimental χ K-Fold Cross-Validation
# Evaluate χ(T) head on small experimental dataset

set -e  # Exit on error

# Smart argument detection:
# - If arg 1 ends with .pt, treat it as pretrained model path
#   Usage: bash run_exp_chi_cv.sh results/dft_pretrain_*/checkpoints/best_model.pt [config] [device]
# - Otherwise, use standard argument order
#   Usage: bash run_exp_chi_cv.sh [config] [pretrained_path] [device]

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
        echo "WARNING: No DFT pretrain results found in results/ directory!"
        echo "Training will start from scratch (no pretrained weights)."
        echo "For better performance, run Stage 1 first:"
        echo "  bash scripts/run_pretrain_dft.sh"
        PRETRAINED_PATH=""
    else
        PRETRAINED_PATH="$LATEST_DFT_DIR/checkpoints/best_model.pt"
        echo "Found: $PRETRAINED_PATH"
    fi
fi

echo "=================================================="
echo "Experimental χ K-Fold Cross-Validation"
echo "=================================================="
echo "Config: $CONFIG_PATH"
if [ -n "$PRETRAINED_PATH" ]; then
    echo "Pretrained model: $PRETRAINED_PATH"
else
    echo "Pretrained model: None (training from scratch)"
fi
echo "Device: $DEVICE"
echo ""

# Check if pretrained model exists (if specified)
if [ -n "$PRETRAINED_PATH" ] && [ ! -f "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained model not found at: $PRETRAINED_PATH"
    echo ""
    echo "Usage: $0 [config_path] [pretrained_model_path] [device]"
    echo ""
    echo "Example:"
    echo "  $0 configs/config.yaml results/dft_pretrain_*/checkpoints/best_model.pt cuda"
    echo "  $0 results/dft_pretrain_*/checkpoints/best_model.pt"
    exit 1
fi

# Run CV
if [ -n "$PRETRAINED_PATH" ]; then
    python -m src.training.cv_exp_chi \
        --config "$CONFIG_PATH" \
        --pretrained "$PRETRAINED_PATH" \
        --device "$DEVICE"
else
    python -m src.training.cv_exp_chi \
        --config "$CONFIG_PATH" \
        --device "$DEVICE"
fi

echo ""
echo "Cross-validation completed!"
echo "Check results/ directory for aggregated metrics."
