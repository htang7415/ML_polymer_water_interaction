#!/usr/bin/env bash

################################################################################
# hyperparameter_optimization.sh
#
# Run Optuna hyperparameter optimization for transfer learning.
#
# This script:
# 1. Runs Optuna optimization to find the best hyperparameters
# 2. Saves trial results to hyperparameter_optimization/hy.txt
# 3. Saves best parameters to hyperparameter_optimization/best_hyperparameters.txt
#
# Prerequisites:
# - Run MF_descriptors.sh first to precompute features
################################################################################

echo "================================================================================"
echo "HYPERPARAMETER OPTIMIZATION WITH OPTUNA"
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

# Create output directory
mkdir -p hyperparameter_optimization

# Run Optuna optimization
echo "Starting Optuna optimization..."
echo "Configuration: config.yaml"
echo ""
echo "This may take a while depending on the number of trials..."
echo ""

python scripts/optuna_objective.py --config config.yaml

echo ""
echo "================================================================================"
echo "HYPERPARAMETER OPTIMIZATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - hyperparameter_optimization/hy.txt (all trials)"
echo "  - hyperparameter_optimization/best_hyperparameters.txt (best parameters)"
echo ""
echo "You can now run transfer_learning.sh with the best parameters."
echo ""
