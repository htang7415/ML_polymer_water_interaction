#!/bin/bash
#
# Wrapper script to run direct learning from parent directory
#
# This script allows you to run direct learning without changing into the
# direct_learning/ subdirectory.
#
# Usage (from transfer_learning_AB_chi/):
#   bash run_direct_learning.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Running Direct Learning"
echo "=========================================="
echo ""
echo "Working directory: $(pwd)"
echo "Changing to direct_learning/ subdirectory..."
echo ""

# Change to direct_learning directory
cd "$(dirname "$0")/direct_learning"

# Execute the actual direct learning script
bash direct_learning.sh

# Return to parent directory
cd ..

echo ""
echo "=========================================="
echo "Returned to: $(pwd)"
echo "=========================================="
echo ""
echo "Results are in direct_learning/outputs/"
echo "Compare with transfer learning: outputs/metrics/metrics.txt"
