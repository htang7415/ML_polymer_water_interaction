#!/bin/bash

# ============================================================================
# Direct Learning Script
# ============================================================================
# This script runs the final direct learning training and evaluation using
# the best hyperparameters found by Optuna (or defaults from config.yaml).
#
# The script:
# 1. Loads best hyperparameters from hyperparameter_optimization/best_hyperparameters.txt
#    (or uses defaults from config.yaml if not found)
# 2. Trains directly on experimental data with 5-fold CV
# 3. Generates all plots (parity plots, calibration plots)
# 4. Saves all metrics and plots to Results/ and Figures/ directories
# ============================================================================

echo "=========================================="
echo "Direct Learning: Final Training"
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

# Check if data file exists
if [ ! -f "../Data/Experiment_chi_data.csv" ]; then
    echo "Error: ../Data/Experiment_chi_data.csv not found"
    exit 1
fi

# Check for best hyperparameters
if [ -f "hyperparameter_optimization/best_hyperparameters.txt" ]; then
    echo "Using best hyperparameters from: hyperparameter_optimization/best_hyperparameters.txt"
else
    echo "Warning: Best hyperparameters not found, using defaults from config.yaml"
fi

echo ""
echo "Starting direct learning training..."
echo ""

# Run direct learning
python scripts/run_direct_learning.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Direct learning completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  - Results/metrics.json"
    echo "  - Figures/"
    echo ""
else
    echo ""
    echo "Error: Direct learning failed"
    exit 1
fi
