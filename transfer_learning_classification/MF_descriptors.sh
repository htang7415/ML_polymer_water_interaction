#!/usr/bin/env bash

################################################################################
# MF_descriptors.sh
#
# Precompute Morgan fingerprints and molecular descriptors for both datasets:
# - OMG_DFT_COSMOC_chi.csv (DFT chi data)
# - Binary_solubility.csv (binary solubility data)
#
# This script must be run before hyperparameter optimization or training.
################################################################################

echo "================================================================================"
echo "PRECOMPUTING MORGAN FINGERPRINTS AND MOLECULAR DESCRIPTORS"
echo "================================================================================"
echo ""

# Activate your conda environment if needed (uncomment and modify as needed)
# conda activate your_environment_name

# Run the feature precomputation script
python3 scripts/features.py --precompute

echo ""
echo "================================================================================"
echo "PRECOMPUTATION COMPLETE"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  - Data/DFT_features.csv"
echo "  - Data/binary_features.csv"
echo ""
echo "You can now run hyperparameter optimization or direct training."
echo ""
