#!/bin/bash

# ============================================================================
# Hyperparameter Optimization Script for Direct Learning
# ============================================================================
# This script runs Optuna-based hyperparameter optimization for the
# polymer-water chi parameter direct learning pipeline (no transfer learning).
#
# The optimization:
# 1. Trains models directly on experimental data with 5-fold CV
# 2. Maximizes mean validation R² across folds
# 3. Saves all trial results to hyperparameter_optimization/hy.txt
# 4. Saves best hyperparameters to hyperparameter_optimization/best_hyperparameters.txt
# ============================================================================

echo "=========================================="
echo "Direct Learning: Hyperparameter Optimization"
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

echo "Starting Optuna hyperparameter optimization..."
echo "Configuration: config.yaml"
echo ""

# Run Optuna optimization
python scripts/optuna_objective.py

# Check if optimization completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Optimization completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  - hyperparameter_optimization/hy.txt"
    echo "  - hyperparameter_optimization/best_hyperparameters.txt"
    echo ""
else
    echo ""
    echo "Error: Optimization failed"
    exit 1
fi
