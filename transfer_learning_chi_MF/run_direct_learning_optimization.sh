#!/bin/bash

# ============================================================================
# Direct Learning: Hyperparameter Optimization (Wrapper Script)
# ============================================================================
# Run this from the transfer_learning_chi/ directory
# ============================================================================

echo "=========================================="
echo "Direct Learning: Hyperparameter Optimization"
echo "=========================================="
echo ""

# Check if direct_learning folder exists
if [ ! -d "direct_learning" ]; then
    echo "Error: direct_learning folder not found"
    exit 1
fi

# Run optimization from direct_learning folder
cd direct_learning
./direct_learning_optimization.sh
