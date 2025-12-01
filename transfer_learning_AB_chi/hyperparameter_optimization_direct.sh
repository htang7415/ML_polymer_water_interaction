#!/bin/bash
#
# Wrapper script to run direct learning hyperparameter optimization from parent directory
#
# This script allows you to run direct learning hyperparameter optimization
# without changing into the direct_learning/ subdirectory.
#
# Usage (from transfer_learning_AB_chi/):
#   bash hyperparameter_optimization_direct.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Running Direct Learning Hyperparameter Optimization"
echo "=========================================="
echo ""
echo "Working directory: $(pwd)"
echo "Changing to direct_learning/ subdirectory..."
echo ""

# Change to direct_learning directory
cd "$(dirname "$0")/direct_learning"

# Execute the actual hyperparameter optimization script
bash hyperparameter_optimization.sh

# Return to parent directory
cd ..

echo ""
echo "=========================================="
echo "Returned to: $(pwd)"
echo "=========================================="
echo ""
echo "Results are in direct_learning/hyperparameter_optimization/"
echo "Next step: bash run_direct_learning.sh"
