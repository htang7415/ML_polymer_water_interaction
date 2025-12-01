#!/usr/bin/env bash

################################################################################
# transfer_learning.sh
#
# Run final transfer learning training with best hyperparameters.
#
# This script:
# 1. Checks if best hyperparameters exist (from Optuna optimization)
# 2. Updates config.yaml with best parameters (if available)
# 3. Runs pretraining (chi regression)
# 4. Runs fine-tuning (binary solubility classification)
# 5. Generates all plots and metrics
#
# Prerequisites:
# - Run MF_descriptors.sh first to precompute features
# - Optionally run hyperparameter_optimization.sh to find best parameters
################################################################################

echo "================================================================================"
echo "TRANSFER LEARNING: CHI REGRESSION -> BINARY SOLUBILITY CLASSIFICATION"
echo "================================================================================"
echo ""

# Check if precomputed features exist
if [ ! -f "Data/DFT_features.csv" ] || [ ! -f "Data/binary_features.csv" ]; then
    echo "ERROR: Precomputed features not found!"
    echo "Please run MF_descriptors.sh first to precompute features."
    exit 1
fi

# Activate your conda environment if needed (uncomment and modify as needed)
# conda activate your_environment_name

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
    echo "To find optimal hyperparameters, run: bash hyperparameter_optimization.sh"
    echo ""
fi

# Create output directories
mkdir -p outputs/regression
mkdir -p outputs/classification

# Run training
echo "================================================================================"
echo "STARTING TRAINING"
echo "================================================================================"
echo ""

python scripts/train.py --config config.yaml

echo ""
echo "================================================================================"
echo "TRANSFER LEARNING COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo ""
echo "Regression (Pretraining):"
echo "  - outputs/regression/metrics.json"
echo "  - outputs/regression/parity_train.png"
echo "  - outputs/regression/parity_val.png"
echo "  - outputs/regression/parity_test.png"
echo "  - outputs/regression/calibration_train.png"
echo "  - outputs/regression/calibration_val.png"
echo "  - outputs/regression/calibration_test.png"
echo "  - outputs/regression/loss_curves.png"
echo "  - outputs/regression/best_regression_model.pt"
echo ""
echo "Classification (Fine-tuning):"
echo "  - outputs/classification/metrics.json"
echo "  - outputs/classification/roc_fold*.png (per-fold ROC curves)"
echo "  - outputs/classification/roc_aggregate_train.png"
echo "  - outputs/classification/roc_aggregate_val.png"
echo "  - outputs/classification/confusion_fold*.png (per-fold confusion matrices)"
echo "  - outputs/classification/confusion_aggregate.png"
echo "  - outputs/classification/probability_histogram.png"
echo "  - outputs/classification/loss_curves_aggregate.png"
echo ""
