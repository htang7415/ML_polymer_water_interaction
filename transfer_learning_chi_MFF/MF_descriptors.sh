#!/bin/bash

# ============================================================================
# Morgan Fingerprints and Descriptors Pre-calculation Script
# ============================================================================
# This script pre-calculates Morgan fingerprints and RDKit descriptors for
# all molecules in both DFT and experimental datasets, then saves them to disk.
#
# Benefits:
# - Dramatically speeds up hyperparameter optimization by avoiding redundant
#   feature calculations across trials
# - Features are computed once and reused for all training runs
#
# Output files (saved to Data/):
# - features_fp.pkl: Morgan fingerprints (binary) for all unique molecules
# - features_fpf.pkl: Morgan fingerprints (frequency) for all unique molecules
# - features_fpf_metadata.pkl: Metadata for frequency fingerprints (hash codes, filtering)
# - features_desc.pkl: RDKit descriptors for all unique molecules
# - features_metadata.pkl: Metadata about feature settings
#
# Usage:
#   bash MF_descriptors.sh
#
# Note: Run this script before hyperparameter optimization for best performance
# ============================================================================

echo "=========================================="
echo "Pre-calculating Features"
echo "=========================================="
echo ""

# Activate virtual environment (uncomment and modify if needed)
# source /path/to/your/venv/bin/activate
# conda activate your_env_name

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found in current directory"
    exit 1
fi

# Check if data files exist
if [ ! -f "Data/OMG_DFT_COSMOC_chi.csv" ]; then
    echo "Error: Data/OMG_DFT_COSMOC_chi.csv not found"
    exit 1
fi

if [ ! -f "Data/Experiment_chi_data.csv" ]; then
    echo "Error: Data/Experiment_chi_data.csv not found"
    exit 1
fi

echo "Starting feature calculation..."
echo "This may take a few minutes depending on dataset size."
echo ""

# Run feature calculation
python scripts/calculate_features.py

# Check if calculation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Feature calculation completed!"
    echo "=========================================="
    echo ""
    echo "Features saved to Data/ directory:"
    echo "  - features_fp.pkl (Morgan binary fingerprints)"
    echo "  - features_fpf.pkl (Morgan frequency fingerprints)"
    echo "  - features_fpf_metadata.pkl (FPF metadata)"
    echo "  - features_desc.pkl (RDKit descriptors)"
    echo "  - features_metadata.pkl (metadata)"
    echo ""
    echo "You can now run hyperparameter optimization"
    echo "with pre-computed features for faster training."
    echo ""
else
    echo ""
    echo "Error: Feature calculation failed"
    exit 1
fi
