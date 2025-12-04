#!/usr/bin/env bash

################################################################################
# direct_learning_optimization.sh
#
# Run Optuna hyperparameter optimization for direct learning.
#
# This script:
# 1. Runs Optuna optimization to find the best hyperparameters
# 2. Saves trial results to hyperparameter_optimization/hy.txt
# 3. Saves best parameters to hyperparameter_optimization/best_hyperparameters.txt
#
# Prerequisites:
# - Run MF_descriptors.sh first to precompute features
#
# Usage:
#   bash direct_learning/direct_learning_optimization.sh    (from transfer_learning_classification/)
#   bash direct_learning_optimization.sh                    (from direct_learning/)
#
# Note: This optimization can take many hours depending on n_trials in config.yaml
################################################################################

echo "================================================================================"
echo "HYPERPARAMETER OPTIMIZATION WITH OPTUNA (DIRECT LEARNING)"
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

# Create output directory
mkdir -p hyperparameter_optimization

# Run Optuna optimization
echo "Starting Optuna optimization for direct learning..."
echo "Configuration: config.yaml"
echo ""
echo "This may take a while depending on the number of trials..."
echo "(Default: 1000 trials, can take 12-24 hours)"
echo ""

python scripts/optuna_objective_direct.py --config config.yaml

echo ""
echo "================================================================================"
echo "HYPERPARAMETER OPTIMIZATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - hyperparameter_optimization/hy.txt (all trials)"
echo "  - hyperparameter_optimization/best_hyperparameters.txt (best parameters)"
echo ""
echo "Next steps:"
echo "  1. Review best hyperparameters in best_hyperparameters.txt"
echo "  2. Update config.yaml with the best parameters (optional)"
echo "  3. Run: bash direct_learning.sh"
echo ""
