# Polymer–Water χ(T) + Solubility ML Repository

A clean, modular PyTorch repository for predicting polymer–water interaction parameters (χ) and solubility from polymer repeat-unit SMILES, with uncertainty quantification via MC Dropout.

## Overview

This repository implements a multi-task deep learning framework for:

1. **DFT χ prediction**: Predict COSMO-SAC/DFT-computed χ(polymer–water, T_ref) from ~47,676 data points
2. **Experimental χ(T) modeling**: Predict temperature-dependent χ using the form χ(T) = A/T + B from ~40 experimental points
3. **Binary solubility classification**: Predict water solubility (soluble/insoluble) for 430 polymers
4. **Uncertainty quantification**: Epistemic uncertainty via MC Dropout

### Key Features

- **Config-driven**: Single YAML file controls all hyperparameters
- **Modular design**: Easy to extend and reuse components
- **Two-stage training**: DFT pretraining → multi-task fine-tuning
- **K-fold CV**: Robust evaluation on small experimental dataset
- **Publication-ready outputs**: High-quality figures and comprehensive metrics
- **Reproducible**: Seeding and logging for full reproducibility

## Project Structure

```
polymer_water_interaction/
├── README.md                    # This file
├── pyproject.toml              # Package configuration
├── .gitignore                  # Git ignore patterns
├── env.yml                     # Conda environment specification
├── configs/
│   └── config.yaml             # Single source of truth for all hyperparameters
├── Data/                       # Data directory (existing)
│   ├── OMG_DFT_COSMOC_chi.csv
│   ├── Experiment_chi_data.csv
│   └── Binary_solubility.csv
├── data/
│   └── processed/              # Cached features and splits (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── featurization.py   # SMILES → Morgan FP + RDKit descriptors
│   │   ├── datasets.py        # PyTorch Dataset classes
│   │   └── splits.py          # Train/val/test splits + k-fold CV
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py         # Shared MLP encoder
│   │   └── multitask_model.py # χ(T) head + solubility head
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py          # Multi-task loss functions
│   │   ├── train_dft.py       # Stage 1: DFT χ pretraining
│   │   ├── train_multitask.py # Stage 2: Multi-task fine-tuning
│   │   └── cv_exp_chi.py      # K-fold CV for experimental χ
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Regression + classification metrics
│   │   ├── plots.py           # Publication-quality figures
│   │   ├── uncertainty.py     # MC Dropout utilities
│   │   └── analysis.py        # χ_RT vs solubility analysis
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # YAML loading and validation
│       ├── logging_utils.py   # Logging setup
│       └── seed_utils.py      # Reproducibility seeding
├── scripts/
│   ├── run_pretrain_dft.sh    # Run DFT pretraining
│   ├── run_multitask.sh       # Run multi-task training
│   ├── run_exp_chi_cv.sh      # Run k-fold CV
│   ├── run_hparam_search.sh   # Hyperparameter optimization
│   └── hparam_opt.py          # HPO driver script
└── results/                   # Auto-created per-run directories
```

## Data Format

### 1. DFT χ Data (`OMG_DFT_COSMOC_chi.csv`)

Each row represents one polymer with COSMO-SAC computed χ:

- `SMILES` (string): Polymer repeat-unit SMILES with two `*` connection points (e.g., `*CC(=O)OCC*`)
- `chi` (float): DFT-computed χ for polymer–water at reference temperature
- `temp` (float, optional): Temperature in Kelvin

### 2. Experimental χ Data (`Experiment_chi_data.csv`)

Each row is a single χ(T) measurement:

- `SMILES` (string): Polymer repeat-unit SMILES with two `*`
- `chi` (float): Experimental χ for polymer–water
- `temp` (float): Measurement temperature in Kelvin

### 3. Solubility Data (`Binary_solubility.csv`)

Each row represents one polymer with solubility label:

- `SMILES` (string): Polymer repeat-unit SMILES with two `*`
- `soluble` (int): Binary label (1 = soluble, 0 = insoluble)

## Environment Setup

### 1. Create Conda Environment

```bash
conda env create -f env.yml
conda activate polymer_chi_ml
```

### 2. Verify Installation

```bash
python -c "import torch; import rdkit; print('Environment ready!')"
```

## Configuration

All hyperparameters are controlled via `configs/config.yaml`:

- **Paths**: Input CSVs, processed data, results directory
- **Chemistry**: Morgan fingerprint parameters, RDKit descriptors
- **Training**: Batch sizes, learning rates, epochs, optimizer settings
- **Model**: Encoder/head architectures, dropout rates
- **Loss weights**: Multi-task loss balancing
- **CV**: K-fold settings for experimental χ
- **Uncertainty**: MC Dropout sample count

**Example modifications:**

```yaml
# Change model capacity
model:
  encoder_latent_dim: 256  # Increase from 128

# Adjust learning rates
training:
  lr_pretrain: 5e-4       # Decrease from 1e-3

# Rebalance multi-task losses
loss_weights:
  lambda_exp: 3.0         # Increase experimental χ weight
```

## How to Run

### Stage 1: DFT χ Pretraining

Train encoder + χ(T) head on large DFT dataset:

```bash
bash scripts/run_pretrain_dft.sh
```

**Outputs:**
- Model checkpoint: `results/dft_pretrain_<timestamp>/best_model.pt`
- Metrics: `metrics_summary.json`, `metrics_summary.csv`
- Predictions: `predictions_dft_test.csv`
- Figures: DFT parity plot

### Stage 2: Multi-Task Fine-Tuning

Fine-tune on DFT + experimental χ + solubility:

```bash
bash scripts/run_multitask.sh
```

**Outputs:**
- Full model checkpoint with solubility head
- Comprehensive metrics (χ and solubility)
- Per-polymer predictions
- Publication figures: ROC, PR, calibration, confusion matrix, χ_RT vs solubility

### Experimental χ K-Fold CV

Robustly evaluate χ(T) head on experimental data:

```bash
bash scripts/run_exp_chi_cv.sh
```

**Outputs:**
- Per-fold metrics: `exp_chi_cv_metrics.csv`
- Aggregated statistics (mean ± std across folds)
- Combined parity plot for experimental χ

### Hyperparameter Optimization

Automated search using Optuna:

```bash
bash scripts/run_hparam_search.sh
```

**Outputs:**
- Trial log: `results/hparam_search/trials.csv`
- Best config: `results/hparam_search/best_config.yaml`
- Optimization history plots

## Model Architecture

### Shared Encoder

```
Input (Morgan FP + descriptors)
  ↓
Linear(input_dim → 512) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(256 → latent_dim) → ReLU
  ↓
z(polymer) [latent representation]
```

### χ(T) Head

```
z(polymer)
  ↓
Linear(latent_dim → 64) → ReLU → Dropout(0.1)
  ↓
Linear(64 → 2) → [A, B]
  ↓
χ(T) = A/T + B
```

### Solubility Head

```
[z(polymer), χ_RT]  (concatenated)
  ↓
Linear(latent_dim+1 → 64) → ReLU → Dropout(0.1)
  ↓
Linear(64 → 1) → Sigmoid
  ↓
P(soluble)
```

## Training Strategy

1. **Stage 1 (Pretraining)**:
   - Train encoder + χ(T) head on DFT χ only
   - Loss: MSE on χ_DFT
   - Early stopping on DFT validation set

2. **Stage 2 (Multi-task fine-tuning)**:
   - Load pretrained encoder + χ head
   - Add solubility head
   - Train on combined: DFT χ + experimental χ + solubility
   - Loss: L_total = λ₁·L_DFT + λ₂·L_exp + λ₃·L_sol
   - Early stopping on validation metrics (ROC-AUC + χ MAE)

3. **K-Fold CV**:
   - SMILES-level k-fold split (default k=5)
   - Ensures all measurements for same polymer stay in same fold
   - Reports mean ± std metrics across folds

## Uncertainty Quantification

MC Dropout is used for epistemic uncertainty:

- During training: Dropout used for regularization
- During inference: Dropout layers stay active
- Run multiple forward passes (default: 50) with different dropout masks
- Compute mean and std of predictions

**For χ_RT:**
- `chi_mean`: Point prediction
- `chi_std`: Epistemic uncertainty

**For solubility:**
- `p_mean`: Point prediction
- `p_std`: Prediction uncertainty

High uncertainty indicates out-of-distribution or ambiguous cases.

## Results and Outputs

Each run creates a timestamped directory under `results/` containing:

### Metrics Files
- `metrics_summary.json`: Comprehensive metrics dictionary
- `metrics_summary.csv`: Tabular metrics
- `predictions_*.csv`: Per-sample predictions with uncertainties

### Figures (PNG + PDF)
- `dft_parity.{png,pdf}`: DFT χ parity plot
- `exp_parity.{png,pdf}`: Experimental χ parity plot (color by T)
- `exp_residual_vs_T.{png,pdf}`: Residual analysis vs temperature
- `sol_roc_curve.{png,pdf}`: Solubility ROC curve
- `sol_pr_curve.{png,pdf}`: Precision-Recall curve
- `sol_calibration.{png,pdf}`: Calibration plot
- `sol_confusion_matrix.{png,pdf}`: Confusion matrix heatmap
- `chi_rt_vs_solubility.{png,pdf}`: χ_RT distribution by class
- `uncertainty_vs_error_*.{png,pdf}`: Uncertainty calibration plots

### Metadata
- `config_used.yaml`: Exact configuration for this run
- `train.log`: Detailed training logs
- `git_info.txt`: Git commit hash (if applicable)

## Metrics

### Regression (DFT/Experimental χ)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² (coefficient of determination)
- Spearman rank correlation

### Classification (Solubility)
- ROC-AUC
- PR-AUC (positive class)
- Accuracy & Balanced Accuracy
- Precision, Recall, F1
- Matthews Correlation Coefficient (MCC)
- Brier score
- Confusion matrix

## Extensibility

This repository is designed for easy extension:

### 1. Swap Encoder
Replace the MLP encoder with a graph neural network:
- Modify `src/models/encoder.py`
- Update featurization to generate graph representations
- Training code remains unchanged

### 2. Add New Tasks
Add prediction heads for other properties:
- Create new head class in `src/models/multitask_model.py`
- Add corresponding loss in `src/training/losses.py`
- Update config with new task weight

### 3. Different Solvents
Extend to polymer–solvent interactions beyond water:
- Add solvent identifier to dataset
- Optionally condition encoder on solvent features
- Train separate models or use multi-solvent training

### 4. Alternative Featurization
Use different molecular representations:
- Modify `src/data/featurization.py`
- Update config with new featurization parameters
- Cache mechanism handles recomputation automatically

## Citation

If you use this code in your research, please cite:

```bibtex
@software{polymer_chi_solubility_ml,
  title = {Polymer–Water χ(T) and Solubility Prediction Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/polymer_chi_solubility}
}
```

## License

[Specify license here, e.g., MIT, Apache 2.0]

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Acknowledgments

- DFT χ data from COSMO-SAC calculations
- Experimental χ data from [citation]
- Solubility data from [citation]
