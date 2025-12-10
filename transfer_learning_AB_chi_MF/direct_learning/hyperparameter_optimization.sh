#!/bin/bash
#
# Hyperparameter optimization for direct learning (no transfer learning)
#
# This script runs Optuna hyperparameter optimization to find the best
# hyperparameters for training directly on experimental chi data.
#
# Usage:
#   bash hyperparameter_optimization.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Direct Learning Hyperparameter Optimization"
echo "=========================================="
echo ""

# Change to direct_learning directory
cd "$(dirname "$0")"

# Check if experimental features exist
if [ ! -f "EXP_features.csv" ]; then
    echo "ERROR: EXP_features.csv not found!"
    echo "Please run the feature computation script first."
    echo ""
    echo "You can create a symlink to the parent directory's features:"
    echo "  ln -s ../EXP_features.csv EXP_features.csv"
    exit 1
fi

# Run Optuna optimization
echo "Starting Optuna optimization..."
echo "This will run 1000 trials (configurable in config.yaml)"
echo ""

python scripts/optuna_objective_direct.py --config config.yaml

echo ""
echo "=========================================="
echo "Optimization Complete!"
echo "=========================================="
echo ""
echo "Results saved in hyperparameter_optimization/"
echo "  - best_hyperparameters.txt: Best hyperparameters found"
echo "  - hy.txt: All trial results"
echo ""
echo "Next step: Run direct learning with best hyperparameters"
echo "  bash direct_learning.sh"
