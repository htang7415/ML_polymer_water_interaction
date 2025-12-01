#!/usr/bin/env bash

################################################################################
# direct_learning.sh
#
# Run direct learning training on binary solubility classification.
#
# This script:
# 1. Checks if best hyperparameters exist (from Optuna optimization)
# 2. Trains classification model from scratch (no pretraining)
# 3. Generates all plots and metrics (same as fine-tuning stage)
#
# Prerequisites:
# - Run MF_descriptors.sh first to precompute features
# - Optionally run direct_learning_optimization.sh to find best parameters
#
# Usage:
#   bash direct_learning/direct_learning.sh         (from transfer_learning_classification/)
#   bash direct_learning.sh                         (from direct_learning/)
################################################################################

echo "================================================================================"
echo "DIRECT LEARNING: BINARY SOLUBILITY CLASSIFICATION (NO TRANSFER LEARNING)"
echo "================================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if precomputed features exist
if [ ! -f "../Data/binary_features.csv" ]; then
    echo "ERROR: Precomputed features not found!"
    echo "Please run ../MF_descriptors.sh first to precompute features."
    exit 1
fi

# Check if best hyperparameters exist
BEST_PARAMS_FILE="hyperparameter_optimization/best_hyperparameters.txt"

if [ -f "$BEST_PARAMS_FILE" ]; then
    echo "Found best hyperparameters from Optuna optimization."
    echo "Location: $BEST_PARAMS_FILE"
    echo ""
    echo "Best hyperparameters:"
    cat "$BEST_PARAMS_FILE"
    echo ""
    echo "NOTE: You may need to manually update config.yaml with these parameters"
    echo "      or use the default parameters in config.yaml."
    echo ""
else
    echo "No best hyperparameters found."
    echo "Using default hyperparameters from config.yaml"
    echo ""
    echo "To find optimal hyperparameters, run: bash direct_learning_optimization.sh"
    echo ""
fi

# Create output directories
mkdir -p outputs
mkdir -p hyperparameter_optimization

# Run training
echo "================================================================================"
echo "STARTING DIRECT LEARNING TRAINING"
echo "================================================================================"
echo ""

python scripts/train_direct.py --config config.yaml

echo ""
echo "================================================================================"
echo "DIRECT LEARNING COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo ""
echo "Direct Learning (Classification):"
echo "  - outputs/metrics.json"
echo "  - outputs/roc_fold*.png (per-fold ROC curves)"
echo "  - outputs/roc_aggregate_train.png"
echo "  - outputs/roc_aggregate_val.png"
echo "  - outputs/confusion_fold*.png (per-fold confusion matrices)"
echo "  - outputs/confusion_aggregate.png"
echo "  - outputs/probability_histogram.png"
echo "  - outputs/loss_curves_aggregate.png"
echo ""
echo "================================================================================"
echo "COMPARISON WITH TRANSFER LEARNING"
echo "================================================================================"
echo ""
echo "To compare direct learning with transfer learning, check:"
echo "  Transfer learning results: ../outputs/classification/"
echo "  Direct learning results:   ./outputs/"
echo ""
echo "Key metrics to compare (from metrics.json):"
echo "  - aggregate.val_f1_mean"
echo "  - aggregate.val_roc_auc_mean"
echo "  - aggregate.val_accuracy_mean"
echo ""
