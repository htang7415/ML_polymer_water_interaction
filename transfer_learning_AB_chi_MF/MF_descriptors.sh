#!/usr/bin/env bash

# Precompute Morgan fingerprints and molecular descriptors
# This script should be run ONCE before hyperparameter optimization or training

set -e  # Exit on error

echo "========================================="
echo "Precomputing Morgan Fingerprints and Molecular Descriptors"
echo "========================================="
echo ""

# Activate virtual environment if needed
# Uncomment and modify the line below if using a virtual environment
# source /path/to/venv/bin/activate

# Run feature precomputation
echo "Running feature precomputation..."
python scripts/features.py --precompute --config config.yaml

echo ""
echo "========================================="
echo "Feature precomputation completed!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - DFT_features.csv"
echo "  - EXP_features.csv"
echo ""
echo "You can now run hyperparameter optimization or transfer learning."
