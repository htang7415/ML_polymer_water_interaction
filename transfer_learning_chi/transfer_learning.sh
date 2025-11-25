#!/bin/bash

# ============================================================================
# Transfer Learning Script
# ============================================================================
# This script runs the final transfer learning training and evaluation using
# the best hyperparameters found by Optuna (or defaults from config.yaml).
#
# The script:
# 1. Loads best hyperparameters from hyperparameter_optimization/best_hyperparameters.txt
#    (or uses defaults from config.yaml if not found)
# 2. Pretrains on DFT data
# 3. Fine-tunes with 5-fold CV on experimental data
# 4. Generates all plots (parity plots, calibration plots, training curves)
# 5. Saves all metrics and plots to outputs/ directory
# ============================================================================

echo "=========================================="
echo "Transfer Learning: Final Training"
echo "=========================================="
echo ""

# Activate virtual environment (uncomment and modify if needed)
# source /path/to/your/venv/bin/activate
# conda activate your_env_name

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found in current directory"
    exit 1
fi

# Check if data files exist
if [ ! -f "Data/OMG_DFT_COSMOC_chi.csv" ]; then
    echo "Error: Data/OMG_DFT_COSMOC_chi.csv not found"
    exit 1
fi

if [ ! -f "Data/Experiment_chi_data.csv" ]; then
    echo "Error: Data/Experiment_chi_data.csv not found"
    exit 1
fi

# Check for best hyperparameters
if [ -f "hyperparameter_optimization/best_hyperparameters.txt" ]; then
    echo "Using best hyperparameters from: hyperparameter_optimization/best_hyperparameters.txt"
else
    echo "Warning: Best hyperparameters not found, using defaults from config.yaml"
fi

echo ""
echo "Starting transfer learning training..."
echo ""

# Run transfer learning
python scripts/run_transfer_learning.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Transfer learning completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  - outputs/metrics.json"
    echo "  - outputs/plots/"
    echo ""
else
    echo ""
    echo "Error: Transfer learning failed"
    exit 1
fi
