#!/bin/bash
#
# Run CV-aware hyperparameter search (Strategy 2)
#
# This script launches the Optuna-based hyperparameter optimization
# that finds the best Stage 1 hyperparameters for CV exp chi fine-tuning.
#
# Usage:
#   bash scripts/run_hparam_search_cv.sh [OPTIONS]
#
# Options:
#   --n_trials N         Number of trials (default: 50)
#   --timeout_hours H    Timeout in hours (default: 48)
#   --n_cv_folds K       Number of CV folds (default: 5)
#   --resume PATH        Resume from existing study database
#   --study_name NAME    Custom study name
#

set -e  # Exit on error

# Default parameters
N_TRIALS=50
TIMEOUT_HOURS=48
N_CV_FOLDS=5
CV_EPOCHS=20
CONFIG="configs/config_hparam_search.yaml"
STUDY_NAME=""
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --timeout_hours)
            TIMEOUT_HOURS="$2"
            shift 2
            ;;
        --n_cv_folds)
            N_CV_FOLDS="$2"
            shift 2
            ;;
        --cv_epochs)
            CV_EPOCHS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --study_name)
            STUDY_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "CV-Aware Hyperparameter Search (Strategy 2)"
echo "=========================================="
echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Number of trials: $N_TRIALS"
echo "  Timeout: $TIMEOUT_HOURS hours"
echo "  CV folds: $N_CV_FOLDS"
echo "  CV epochs: $CV_EPOCHS"
if [ -n "$STUDY_NAME" ]; then
    echo "  Study name: $STUDY_NAME"
fi
if [ -n "$RESUME" ]; then
    echo "  Resuming from: $RESUME"
fi
echo "=========================================="
echo

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Create log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/hparam_search/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/search_cv_$TIMESTAMP.log"

# Build command
CMD="python -m src.training.hparam_search_cv_aware"
CMD="$CMD --config $CONFIG"
CMD="$CMD --n_trials $N_TRIALS"
CMD="$CMD --timeout_hours $TIMEOUT_HOURS"
CMD="$CMD --n_cv_folds $N_CV_FOLDS"
CMD="$CMD --cv_epochs $CV_EPOCHS"

if [ -n "$STUDY_NAME" ]; then
    CMD="$CMD --study_name $STUDY_NAME"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Run hyperparameter search
echo "Starting CV-aware hyperparameter search..."
echo "Logging to: $LOG_FILE"
echo
echo "Command: $CMD"
echo

# Run with logging
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo "=========================================="
    echo "✅ CV-aware hyperparameter search completed successfully!"
    echo "=========================================="
    echo "Log saved to: $LOG_FILE"
    echo
    echo "Results saved to: results/hparam_search/"
    echo
    echo "Next steps:"
    echo "  1. Check results/hparam_search/cv_aware_*/best_params.json"
    echo "  2. Update configs/config.yaml with best hyperparameters"
    echo "  3. Run full CV training with optimized hyperparameters"
else
    echo
    echo "=========================================="
    echo "❌ CV-aware hyperparameter search failed!"
    echo "=========================================="
    echo "Check log file for details: $LOG_FILE"
    exit 1
fi
