#!/bin/bash

# ============================================================================
# Direct Learning: Final Training (Wrapper Script)
# ============================================================================
# Run this from the transfer_learning_chi/ directory
# ============================================================================

echo "=========================================="
echo "Direct Learning: Final Training"
echo "=========================================="
echo ""

# Check if direct_learning folder exists
if [ ! -d "direct_learning" ]; then
    echo "Error: direct_learning folder not found"
    exit 1
fi

# Run training from direct_learning folder
cd direct_learning
./direct_learning.sh
