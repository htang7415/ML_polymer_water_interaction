#!/bin/bash
#
# Run direct learning with best hyperparameters
#
# This script trains a model directly on experimental chi data using 5-fold CV.
# No transfer learning - trains from random initialization.
#
# Usage:
#   bash direct_learning.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Direct Learning (No Transfer Learning)"
echo "=========================================="
echo ""

# Change to direct_learning directory
cd "$(dirname "$0")"

# Check if experimental features exist
if [ ! -f "EXP_features.csv" ]; then
    echo "ERROR: EXP_features.csv not found!"
    echo "Please create a symlink to the parent directory's features:"
    echo "  ln -s ../EXP_features.csv EXP_features.csv"
    exit 1
fi

# Check if best hyperparameters exist
if [ -f "hyperparameter_optimization/best_hyperparameters.txt" ]; then
    echo "Using best hyperparameters from hyperparameter_optimization/"
else
    echo "WARNING: best_hyperparameters.txt not found!"
    echo "Using default hyperparameters from config.yaml"
    echo ""
fi

# Run direct learning
echo "Starting direct learning training..."
echo ""

python scripts/run_direct_learning.py --config config.yaml

echo ""
echo "=========================================="
echo "Direct Learning Complete!"
echo "=========================================="
echo ""
echo "Results saved in outputs/"
echo "  - outputs/metrics/metrics.json: Performance metrics"
echo "  - outputs/metrics/metrics.txt: Human-readable metrics"
echo "  - outputs/plots/: Visualization plots"
echo ""
echo "Compare with transfer learning results in ../outputs/"
