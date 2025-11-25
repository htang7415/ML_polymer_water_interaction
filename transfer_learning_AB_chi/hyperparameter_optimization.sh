#!/usr/bin/env bash

# Run Optuna hyperparameter optimization
# This script runs the Optuna study to find the best hyperparameters

set -e  # Exit on error

echo "========================================="
echo "Starting Hyperparameter Optimization with Optuna"
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

# Create output directory
mkdir -p hyperparameter_optimization

# Run Optuna optimization
echo "Running Optuna optimization..."
echo "This may take a while depending on the number of trials..."
echo ""

python scripts/optuna_objective.py --config config.yaml

echo ""
echo "========================================="
echo "Hyperparameter optimization completed!"
echo "========================================="
echo ""
echo "Results saved in hyperparameter_optimization/"
echo "  - all_trials.txt: All trial results"
echo "  - best_hyperparameters.txt: Best hyperparameters found"
echo "  - optuna_study.db: Optuna study database"
echo ""
echo "You can now run ./transfer_learning.sh to train with the best hyperparameters."
