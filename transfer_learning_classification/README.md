# Transfer Learning: Chi Regression → Binary Water Solubility Classification

A transfer learning project for polymer water interaction prediction. The model is pretrained on DFT-COSMO-SAC chi (χ) data using regression, then fine-tuned on binary water solubility classification.

## Project Overview

This project implements a two-stage transfer learning pipeline:

1. **Pretraining**: Train a regression model on a large DFT chi dataset to learn polymer-water interaction patterns
2. **Fine-tuning**: Transfer the learned encoder to a binary classification task for water solubility prediction

### Key Features

- **Feature Engineering**: Morgan fingerprints and RDKit molecular descriptors
- **MC Dropout**: Uncertainty estimation using Monte Carlo dropout
- **Stratified K-Fold CV**: 5-fold cross-validation for robust evaluation
- **Hyperparameter Optimization**: Automated search using Optuna
- **Comprehensive Visualization**: Parity plots, calibration plots, ROC curves, confusion matrices

## Project Structure

```
.
├── config.yaml                          # Configuration file (all hyperparameters)
├── scripts/                             # Python modules
│   ├── data_utils.py                    # Data loading utilities
│   ├── features.py                      # Feature engineering (SMILES, fingerprints, descriptors)
│   ├── models.py                        # Neural network architectures
│   ├── utils.py                         # Common utilities (metrics, MC dropout, etc.)
│   ├── plotting.py                      # Plotting functions
│   ├── train.py                         # Training script (pretraining + fine-tuning)
│   └── optuna_objective.py              # Hyperparameter optimization with Optuna
├── MF_descriptors.sh                    # Precompute features
├── hyperparameter_optimization.sh       # Run Optuna search
├── transfer_learning.sh                 # Run final training with best params
└── README.md                            # This file
```

## Data

Two CSV files are required in the `Data/` directory:

1. **`Data/OMG_DFT_COSMOC_chi.csv`**
   - Columns: `SMILES`, `chi`
   - Large DFT-COSMO-SAC χ dataset for pretraining

2. **`Data/Binary_solubility.csv`**
   - Columns: `SMILES`, `water_soluble`
   - Binary labels: 1 = water-soluble, 0 = water-insoluble

## Installation

### Requirements

- Python 3.8+
- PyTorch
- RDKit
- scikit-learn
- Optuna
- pandas, numpy, matplotlib, seaborn, PyYAML

### Setup

```bash
# Create conda environment (recommended)
conda create -n transfer_learning python=3.9
conda activate transfer_learning

# Install dependencies
conda install pytorch -c pytorch
conda install rdkit -c conda-forge
pip install scikit-learn optuna pandas numpy matplotlib seaborn pyyaml
```

## Usage

### Step 1: Precompute Features

Precompute Morgan fingerprints and molecular descriptors:

```bash
bash MF_descriptors.sh
```

This creates:
- `Data/DFT_features.csv`
- `Data/binary_features.csv`

**Note**: This step must be run first before training or optimization.

### Step 2: Hyperparameter Optimization

Find optimal hyperparameters using Optuna:

```bash
bash hyperparameter_optimization.sh
```

This runs hyperparameter search and saves:
- `hyperparameter_optimization/hy.txt` - All trials
- `hyperparameter_optimization/best_hyperparameters.txt` - Best parameters

### Step 3: Transfer Learning Training

Run final training with best hyperparameters:

```bash
bash transfer_learning.sh
```

This performs:
1. Pretraining on DFT chi data (regression)
2. Fine-tuning on binary solubility data (classification with 5-fold CV)
3. Generates all metrics and plots

## Model Architecture

### Encoder (Shared)

- Multi-layer perceptron (MLP)
- **Flexible hidden layer dimensions**: Each layer can have a different size
  - Configured via `hidden_dims: [256, 128, 64]` in config.yaml
  - Examples:
    - `[256, 128, 64]` - Decreasing (traditional)
    - `[128, 128, 128]` - Constant width
    - `[64, 256, 128]` - Mixed pattern
    - `[512, 256]` - 2 layers only
- ReLU activation
- Dropout for regularization
- During hyperparameter optimization, Optuna independently samples each layer's dimension to discover optimal patterns

### Regression Head (Pretraining)

- Single linear layer: `encoder_output → 1` (chi value)
- Loss: Mean Squared Error (MSE)

### Classification Head (Fine-tuning)

- Single linear layer: `encoder_output → 1` (logit)
- Loss: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- Transfer learning: Initialize encoder from pretrained regression model

## Loss Functions

### Pretraining (Regression)
- **Mean Squared Error (MSE)**
  ```
  Loss = mean((y_pred - y_true)²)
  ```
  - Measures average squared difference between predicted and true chi values
  - Penalizes large errors more heavily

### Fine-tuning (Classification)
- **Binary Cross-Entropy with Logits (BCEWithLogitsLoss)**
  ```
  Loss = -mean(y_true * log(σ(logit)) + (1 - y_true) * log(1 - σ(logit)))
  ```
  where σ is the sigmoid function
  - Standard loss for binary classification
  - Combines sigmoid activation and BCE loss for numerical stability

## Feature Modes

Three feature modes are available (configured in `config.yaml`):

1. **`fp`**: Morgan fingerprint only
2. **`desc`**: Molecular descriptors only
3. **`fp_desc`**: Concatenated fingerprints and descriptors (default)

## Uncertainty Estimation

MC Dropout is used for uncertainty estimation:

- Keep dropout enabled during inference
- Run multiple forward passes (default: 100)
- Compute mean (prediction) and std (uncertainty)
- Calibration plots show uncertainty vs. actual error

## Outputs

### Regression Outputs (`outputs/regression/`)

- `metrics.json` - Train/val/test MAE, RMSE, R²
- `parity_train.png`, `parity_val.png`, `parity_test.png` - Parity plots
- `calibration_train.png`, `calibration_val.png`, `calibration_test.png` - Uncertainty calibration
- `loss_curves.png` - Training and validation loss curves
- `best_regression_model.pt` - Saved model weights

### Classification Outputs (`outputs/classification/`)

- `metrics.json` - Per-fold and aggregate metrics
- `roc_fold*.png` - ROC curves for each fold
- `roc_aggregate_train.png`, `roc_aggregate_val.png` - Aggregate ROC curves
- `confusion_fold*.png` - Confusion matrices for each fold
- `confusion_aggregate.png` - Aggregate confusion matrix
- `probability_histogram.png` - Distribution of predicted probabilities
- `loss_curves_aggregate.png` - Aggregate training curves across folds

## Configuration

All hyperparameters are defined in `config.yaml`:

### Data Settings
- Feature mode (`fp`, `desc`, `fp_desc`)
- Morgan fingerprint parameters (radius, n_bits)
- Train/val/test split ratios

### Model Settings
- **Hidden layer dimensions** (list): Each layer can have a different size, e.g., `[256, 128, 64]`
  - During hyperparameter optimization, Optuna independently samples each layer's dimension
- **Dropout rate**: Applied after each hidden layer
- **Activation function**: ReLU, Tanh, or LeakyReLU

### Training Settings
- Learning rates (pretraining and fine-tuning)
- Batch sizes
- Number of epochs
- Weight decay
- Number of layers to freeze during fine-tuning

### Optuna Settings
- Number of trials
- Search space ranges for all hyperparameters

## Hyperparameter Optimization

Optuna searches over:

- Feature mode (`fp`, `desc`, `fp_desc`)
- Model architecture:
  - Number of hidden layers (2-4)
  - Each layer's dimension independently sampled from [64, 128, 256, 512]
  - Dropout rate
  - Activation function
- Learning rates and weight decay
- Batch sizes and epochs
- Number of frozen layers
- Random seed for data splitting

**Objective**: Maximize validation F1 score (mean across 5 folds)

## Reproducibility

- Random seeds are set for NumPy, PyTorch, and scikit-learn
- Default seed: 42
- All splits use fixed random states
- Use the same seed for reproducible results

## Advanced Usage

### Run Only Pretraining

```bash
python scripts/train.py --pretrain_only
```

### Run Only Fine-tuning

```bash
python scripts/train.py --finetune_only --pretrained_model outputs/regression/best_regression_model.pt
```

### Custom Configuration

```bash
python scripts/train.py --config my_config.yaml
```

### Manual Hyperparameter Search

```bash
python scripts/optuna_objective.py --config config.yaml
```

## Performance Tips

1. Use precomputed features (`MF_descriptors.sh`) to avoid redundant computation
2. Start with a small number of Optuna trials (e.g., 10-20) to test
3. Use GPU if available (automatic detection)
4. Reduce MC dropout samples for faster evaluation during optimization

## Troubleshooting

### RDKit Errors
- Ensure RDKit is installed: `conda install rdkit -c conda-forge`
- Check SMILES validity in your CSV files

### Out of Memory
- Reduce batch size in `config.yaml`
- Reduce number of MC dropout samples
- Use smaller model (fewer layers or smaller hidden dimensions)

### Poor Performance
- Run hyperparameter optimization
- Try different feature modes
- Check data quality and class balance
- Increase number of training epochs

## Citation

If you use this code, please cite the original work on transfer learning for polymer water interaction prediction.

## License

MIT License

## Contact

For questions or issues, please open an issue on the project repository.
