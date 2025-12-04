# Direct Learning for Binary Solubility Classification

This directory contains the implementation of **direct learning** (training from scratch without transfer learning) for binary polymer water solubility classification. It enables comparison with the transfer learning approach in the parent directory.

## Overview

### Direct Learning vs Transfer Learning

**Transfer Learning (parent directory):**
- Stage 1: Pretrain encoder on DFT chi regression data (large dataset)
- Stage 2: Fine-tune classifier with pretrained encoder on binary data (small dataset)

**Direct Learning (this directory):**
- Single stage: Train classifier from scratch on binary data only
- No pretraining - random weight initialization

## Directory Structure

```
direct_learning/
├── config.yaml                          # Configuration for direct learning
├── scripts/
│   ├── train_direct.py                  # Main training script
│   └── optuna_objective_direct.py       # Hyperparameter optimization
├── direct_learning.sh                   # Run training
├── direct_learning_optimization.sh      # Run optimization
├── outputs/                             # Training results
│   ├── metrics.json
│   ├── roc_*.png
│   ├── confusion_*.png
│   └── ...
└── hyperparameter_optimization/         # Optuna results
    ├── hy.txt
    └── best_hyperparameters.txt
```

## Prerequisites

1. **Precompute features** (if not already done):
   ```bash
   cd ..
   bash MF_descriptors.sh
   ```

2. **Activate conda environment** (same as transfer learning):
   ```bash
   conda activate your_environment_name
   ```

## Usage

**Note:** Run scripts from the parent `transfer_learning_classification/` directory:

### Option 1: Train with default hyperparameters

```bash
# From transfer_learning_classification/ directory
bash direct_learning/direct_learning.sh
```

This will:
- Train a classification model from scratch using 5-fold cross-validation
- Generate all metrics and plots
- Save results to `direct_learning/outputs/`

### Option 2: Optimize hyperparameters first (recommended)

```bash
# From transfer_learning_classification/ directory

# Run Optuna optimization (may take 12-24 hours)
bash direct_learning/direct_learning_optimization.sh

# Review best hyperparameters
cat direct_learning/hyperparameter_optimization/best_hyperparameters.txt

# Update direct_learning/config.yaml with best parameters (optional)

# Run training with optimized parameters
bash direct_learning/direct_learning.sh
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture (`model.hidden_dims`, `model.dropout_rate`)
- Training hyperparameters (`training.direct.epochs`, `learning_rate`, `batch_size`)
- Feature mode (`features.feature_mode`: `fp`, `desc`, or `fp_desc`)
- Optuna search space (`optuna.search_space`)

## Outputs

After training, `outputs/` will contain:

**Metrics:**
- `metrics.json` - Per-fold and aggregate metrics (F1, ROC-AUC, accuracy, etc.)

**Plots:**
- `roc_fold1.png` to `roc_fold5.png` - Per-fold ROC curves
- `roc_aggregate_train.png` - Aggregate ROC curve (training)
- `roc_aggregate_val.png` - Aggregate ROC curve (validation)
- `confusion_fold1.png` to `confusion_fold5.png` - Per-fold confusion matrices
- `confusion_aggregate.png` - Aggregate confusion matrix
- `probability_histogram.png` - Predicted probability distribution
- `loss_curves_aggregate.png` - Training curves

## Comparing with Transfer Learning

After running both approaches, compare results:

```bash
# From transfer_learning_classification/ directory

# Transfer learning results
cat outputs/classification/metrics.json

# Direct learning results
cat direct_learning/outputs/metrics.json
```

**Key metrics to compare:**
- `aggregate.val_f1_mean` - Validation F1 score
- `aggregate.val_roc_auc_mean` - Validation ROC-AUC
- `aggregate.val_accuracy_mean` - Validation accuracy

**Expected outcome:** Transfer learning should outperform direct learning if the DFT regression pretraining provides useful feature representations for solubility classification.

## Implementation Details

### Key Difference in Code

The only code difference from transfer learning is model initialization:

**Transfer learning (fine-tuning):**
```python
# Load pretrained encoder
reg_model = RegressionModel(...)
reg_model.load_state_dict(torch.load(pretrained_model_path))
pretrained_encoder = reg_model.get_encoder()

model = ClassificationModel(
    pretrained_encoder=pretrained_encoder,  # Use pretrained weights
    n_freeze_layers=n_freeze_layers
)
```

**Direct learning:**
```python
model = ClassificationModel(
    pretrained_encoder=None,  # Random initialization
    n_freeze_layers=0
)
```

### Code Reuse

Direct learning reuses utilities from parent `../scripts/`:
- `data_utils.py` - Data loading and feature extraction
- `models.py` - `ClassificationModel` (supports both pretrained and random initialization)
- `utils.py` - Training loops, evaluation, metrics computation
- `plotting.py` - All visualization functions

This ensures fair comparison - the only difference is weight initialization.

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'torch'`
**Solution:** Activate your conda environment first

**Issue:** `FileNotFoundError: ../Data/binary_features.csv`
**Solution:** Run `../MF_descriptors.sh` to precompute features first

**Issue:** Training is very slow
**Solution:** Check if GPU is available. The code automatically uses GPU if available.

## Files

- `config.yaml` - Configuration file
- `scripts/train_direct.py` - Main training script (439 lines)
- `scripts/optuna_objective_direct.py` - Hyperparameter optimization (260 lines)
- `direct_learning.sh` - Shell script to run training
- `direct_learning_optimization.sh` - Shell script to run optimization

## Notes

- Direct learning may require more epochs than fine-tuning since it starts from random weights
- Hyperparameter optimization searches: feature mode, architecture, learning rate, batch size, dropout rate
- Results are reproducible with fixed random seed (default: 42)
- 5-fold stratified cross-validation ensures robust evaluation on small dataset
