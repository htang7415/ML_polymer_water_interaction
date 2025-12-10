#!/usr/bin/env bash

# Run final transfer learning with best hyperparameters
# This script runs the complete transfer learning pipeline:
# 1. Loads best hyperparameters (or uses defaults)
# 2. Pretrains on DFT data
# 3. Fine-tunes on experimental data (5-fold CV)
# 4. Generates all plots
# 5. Saves all metrics

set -e  # Exit on error

echo "========================================="
echo "Starting Transfer Learning"
echo "========================================="
echo ""

# Check if feature files exist
if [ ! -f "DFT_features.csv" ] || [ ! -f "EXP_features.csv" ]; then
    echo "ERROR: Feature files not found!"
    echo "Please run ./MF_descriptors.sh first to precompute features."
    exit 1
fi

# Activate virtual environment if needed
# Uncomment and modify the line below if using a virtual environment
# source /path/to/venv/bin/activate

# Create output directories
mkdir -p outputs/plots
mkdir -p outputs/metrics
mkdir -p outputs/models

# Run transfer learning
echo "Running transfer learning pipeline..."
echo ""

# Check if best hyperparameters exist
if [ -f "hyperparameter_optimization/best_hyperparameters.txt" ]; then
    echo "Using best hyperparameters from hyperparameter_optimization/best_hyperparameters.txt"
    python scripts/run_transfer_learning.py \
        --config config.yaml \
        --best-params hyperparameter_optimization/best_hyperparameters.txt
else
    echo "Best hyperparameters not found. Using default hyperparameters from config.yaml"
    echo "To use optimized hyperparameters, run ./hyperparameter_optimization.sh first."
    python scripts/run_transfer_learning.py \
        --config config.yaml \
        --best-params hyperparameter_optimization/best_hyperparameters.txt
fi

echo ""
echo "========================================="
echo "Transfer learning completed!"
echo "========================================="
echo ""
echo "Results saved in outputs/"
echo "  - outputs/metrics/metrics.txt: Performance metrics"
echo "  - outputs/metrics/metrics.json: Metrics in JSON format"
echo "  - outputs/plots/: All plots (parity plots, training curves, calibration plots)"
echo "  - outputs/models/: Saved pretrained model"
echo ""
echo "You can view the plots and metrics to evaluate the model performance."
