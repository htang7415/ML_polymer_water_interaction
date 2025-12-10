# Polymer-Water χ Transfer Learning

A transfer learning framework for predicting polymer-water Flory-Huggins parameter (χ) as a function of temperature using a physically motivated neural network model.

## Problem of this method

When T is a constant in chi = A/T +B, model cannot have a good A and B. Because it is like chi = A + B, so A increase, B decrease, can meet the training goal. No physical meaning here. 

## Overview

This project implements a transfer learning pipeline that:

1. **Pretrains** a neural network on a large DFT-COSMO-SAC χ dataset
2. **Fine-tunes** the pretrained model on a small experimental χ dataset using 5-fold cross-validation
3. Uses **Optuna** for hyperparameter optimization to maximize validation R² on experimental data
4. Employs **MC dropout** for uncertainty quantification

### Physical Model

The neural network predicts two parameters **A** and **B** for each polymer structure, from which χ at temperature T (in Kelvin) is computed using the physical relation:

```
χ(T) = A / T + B
```

This physically motivated formulation ensures that the temperature dependence follows the expected thermodynamic behavior.

### Loss Function

The training loss is **Mean Squared Error (MSE)** between the predicted χ(T) and the target χ:

```python
# Model outputs A and B
A_pred, B_pred = model(features)

# Compute χ using physical relation
chi_pred = A_pred / T + B_pred

# MSE loss
loss = MSE(chi_pred, chi_true)
```

Where:
- **Input features**: Morgan fingerprints and/or molecular descriptors (configurable)
- **Temperature T** is NOT an input feature; it's only used in the loss computation
- **Output**: Two scalars (A, B) per sample

## Project Structure

```
.
├── config.yaml                          # Configuration file
├── scripts/
│   ├── features.py                      # Feature computation (MF + descriptors)
│   ├── data_utils.py                    # Data loading and splitting
│   ├── model.py                         # PyTorch MLP model
│   ├── train.py                         # Training utilities and MC dropout
│   ├── plotting.py                      # Plotting utilities
│   ├── optuna_objective.py              # Optuna hyperparameter optimization
│   └── run_transfer_learning.py         # Final transfer learning pipeline
├── MF_descriptors.sh                    # Precompute features
├── hyperparameter_optimization.sh       # Run Optuna optimization
├── transfer_learning.sh                 # Run final training with best hyperparameters
└── README.md                            # This file
```

## Requirements

- Python 3.7+
- PyTorch
- RDKit
- Optuna
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyYAML

Install dependencies:

```bash
pip install torch rdkit optuna numpy pandas matplotlib scikit-learn pyyaml
```

## Data Format

The project expects two CSV files in the project root:

1. **OMG_DFT_COSMOC_chi.csv** (DFT dataset for pretraining)
   - Columns: `SMILES`, `temp`, `chi`

2. **Experiment_chi_data.csv** (Experimental dataset for fine-tuning)
   - Columns: `SMILES`, `temp`, `chi`

Where:
- `SMILES`: Polymer SMILES string
- `temp`: Temperature in Kelvin
- `chi`: Flory-Huggins parameter value

## Usage

### Step 1: Precompute Features

First, precompute Morgan fingerprints and molecular descriptors for all datasets:

```bash
bash MF_descriptors.sh
```

This creates:
- `DFT_features.csv`: DFT dataset with precomputed features
- `EXP_features.csv`: Experimental dataset with precomputed features

### Step 2: Hyperparameter Optimization (Optional but Recommended)

Run Optuna hyperparameter optimization to find the best model configuration:

```bash
bash hyperparameter_optimization.sh
```

This performs Bayesian optimization over:
- Feature mode (fingerprints, descriptors, or both)
- Model architecture (layers, hidden dimensions, dropout)
- Training hyperparameters (learning rates, batch sizes, epochs)
- Regularization (weight decay)
- Transfer learning strategy (number of layers to freeze: 0 to n_layers)

Results are saved incrementally to:
- `hyperparameter_optimization/hy.txt`: All trial results (updated after each trial)
- `hyperparameter_optimization/best_hyperparameters.txt`: Best hyperparameters (updated when improved)

**Note**:
- Results are saved incrementally after each trial completes, so you won't lose progress if the optimization is interrupted
- This step can take considerable time depending on the number of trials (configured in `config.yaml`)
- You can monitor progress in real-time by watching `hy.txt` with: `tail -f hyperparameter_optimization/hy.txt`

### Step 3: Transfer Learning

Run the final transfer learning pipeline with the best hyperparameters:

```bash
bash transfer_learning.sh
```

If you skipped Step 2, this will use default hyperparameters from `config.yaml`.

This script:
1. Loads best hyperparameters (or defaults)
2. Pretrains on DFT data
3. Fine-tunes on experimental data with 5-fold cross-validation
4. Evaluates with MC dropout for uncertainty quantification
5. Generates all plots and saves metrics

## Outputs

All outputs are saved in the `outputs/` directory:

### Metrics

- `outputs/metrics/metrics.txt`: Human-readable performance metrics
- `outputs/metrics/metrics.json`: Metrics in JSON format

### Plots

- `outputs/plots/dft_training_curves.png`: DFT training and validation loss curves
- `outputs/plots/dft_train_parity.png`: DFT training set parity plot
- `outputs/plots/dft_val_parity.png`: DFT validation set parity plot
- `outputs/plots/dft_test_parity.png`: DFT test set parity plot
- `outputs/plots/dft_*_calibration.png`: Uncertainty calibration plots for DFT splits
- `outputs/plots/exp_val_parity.png`: Experimental out-of-fold validation parity plot (colored by fold)
- `outputs/plots/exp_val_calibration.png`: Experimental validation uncertainty calibration

### Models

- `outputs/models/pretrained_model.pt`: Pretrained model weights on DFT data

## Configuration

All settings are controlled via `config.yaml`:

- **Data paths**: Input and output file paths
- **Feature settings**: Morgan fingerprint parameters, descriptor selection
- **Model defaults**: Default hyperparameters
- **Optuna settings**: Number of trials, optimization objective
- **Hyperparameter search space**: Ranges for all hyperparameters

## Model Architecture

The neural network is a feedforward MLP:

```
Input (features) → [Linear → ReLU → Dropout] × n_layers → Linear(2) → (A, B)
```

Where:
- `n_layers`: Number of hidden layers (2-4)
- `hidden_dim`: Hidden layer width (32, 64, 128, 256, or 512)
- `dropout_rate`: Dropout probability (0.1-0.4)

## Data Splitting

### DFT Dataset
- Split at the **polymer level** (by unique canonical SMILES)
- 80% train / 10% validation / 10% test
- **Fixed seed** (42) to ensure reproducibility across trials

### Experimental Dataset
- 5-fold cross-validation at the **polymer level**
- Split seed is a **hyperparameter** optimized by Optuna
- All samples from the same polymer are in the same fold

## Uncertainty Quantification

The model uses **MC Dropout** for uncertainty estimation:

1. During evaluation, dropout remains **ON**
2. Multiple forward passes (default: 50) are performed
3. Mean prediction: μ = mean of predictions
4. Uncertainty: σ = std of predictions
5. Metrics (R², MAE, RMSE) are computed using mean predictions

## Transfer Learning Strategy

1. **Pretraining Phase**:
   - Train on large DFT dataset
   - Learn general polymer-χ relationships

2. **Fine-tuning Phase**:
   - Initialize with pretrained weights
   - Apply flexible layer freezing strategy:
     - `n_freeze_layers`: Number of hidden layers to freeze from input side (0 to n_layers)
     - 0 = all layers trainable
     - 1-3 = freeze first 1-3 layers, train remaining layers + output
     - n_layers = freeze all hidden layers, only output trainable
   - Fine-tune on small experimental dataset

The flexible freezing strategy allows Optuna to find the optimal balance between preserving pretrained knowledge (frozen layers) and adapting to experimental data (trainable layers).

## Performance Evaluation

The Optuna optimization objective is to **maximize mean validation R²** across 5 folds on the experimental dataset.

Additional metrics reported:
- R² (coefficient of determination)
- MAE (mean absolute error)
- RMSE (root mean squared error)

All metrics are computed separately for:
- DFT train, validation, and test sets
- Experimental training and out-of-fold validation sets (per fold and aggregated)

## Citation

If you use this code, please cite the relevant papers for:
- RDKit: https://www.rdkit.org/
- Optuna: https://optuna.org/
- PyTorch: https://pytorch.org/

## License

This project is provided as-is for research and educational purposes.
