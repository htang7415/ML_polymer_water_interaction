# Transfer Learning for Polymer-Water χ Parameter Prediction

This project implements a transfer learning pipeline to predict the Flory-Huggins χ parameter for polymer-water interactions using deep learning.

## Goal

Build a neural network model that:
1. **Pretrains** on a large DFT-COSMO-SAC dataset
2. **Fine-tunes** on a smaller experimental dataset using 5-fold cross-validation
3. Predicts χ from polymer structure (SMILES) and temperature
4. Estimates prediction uncertainty using MC Dropout

## Project Structure

```
.
├── config.yaml                          # Configuration and hyperparameter search space
├── scripts/                             # Python source code
│   ├── data_utils.py                    # Data loading and splitting
│   ├── features.py                      # Feature engineering (RDKit descriptors, Morgan FP)
│   ├── model.py                         # PyTorch MLP with MC Dropout
│   ├── train.py                         # Pretraining and fine-tuning logic
│   ├── plotting.py                      # Plotting utilities
│   ├── optuna_objective.py              # Optuna hyperparameter optimization
│   ├── run_transfer_learning.py         # Final training script
│   └── calculate_features.py            # Pre-calculate features for faster training
├── MF_descriptors.sh                    # Pre-calculate Morgan fingerprints & descriptors
├── hyperparameter_optimization.sh       # Shell script to run Optuna search
├── transfer_learning.sh                 # Shell script to run final training
└── README.md                            # This file
```

## Data Requirements

Place the following CSV files in the `Data/` directory:

1. **`Data/OMG_DFT_COSMOC_chi.csv`** - Large DFT dataset for pretraining
   - Columns: `SMILES`, `temp`, `chi`

2. **`Data/Experiment_chi_data.csv`** - Small experimental dataset for fine-tuning
   - Columns: `SMILES`, `temp`, `chi`

## Model Architecture

- **Type**: Multi-layer perceptron (MLP) regression model
- **Features**: Three modes available (selected via hyperparameter optimization)
  - `fp_T`: Morgan fingerprints + temperature
  - `desc_T`: RDKit descriptors + temperature
  - `fp_desc_T`: Morgan fingerprints + RDKit descriptors + temperature
- **Architecture**: Configurable number of hidden layers, hidden dimensions, and dropout
- **Optimizer**: AdamW with weight decay
- **Uncertainty**: MC Dropout for predictive uncertainty estimation

## Loss Function

The model uses **Mean Squared Error (MSE)** loss for both pretraining and fine-tuning:

```
L = (1/N) Σ (y_pred - y_true)²
```

where:
- `y_pred` is the predicted χ value
- `y_true` is the ground truth χ value
- `N` is the number of samples in the batch

The MSE loss is optimized using the AdamW optimizer with configurable learning rates and weight decay for regularization.

## Training Strategy

### Pretraining (DFT Data)
- Fixed train/val/test split (80/10/10) based on unique polymers
- No early stopping - trains for a fixed number of epochs
- Evaluates on all three splits using MC Dropout

### Fine-tuning (Experimental Data)
- 5-fold cross-validation based on unique polymers
- Configurable batch size (optimized via hyperparameter search)
- Two freezing strategies:
  - `all_trainable`: All parameters are updated
  - `freeze_lower`: Only the final layer is updated
- No early stopping - trains for a fixed number of epochs per fold

### Hyperparameter Optimization
- Uses Optuna to maximize mean validation R² across the 5 folds
- Search space includes:
  - Feature mode (fp_T, desc_T, fp_desc_T)
  - Model architecture (layers, hidden dim, dropout)
  - Pretraining: learning rate, epochs, batch size
  - Fine-tuning: learning rate, epochs, batch size
  - Weight decay
  - Freezing strategy
  - Random seed for experimental data splitting

## Usage

### Step 0 (Optional): Pre-calculate Features for Faster Training

**Recommended for hyperparameter optimization!** Pre-computing features once dramatically speeds up training:

```bash
bash MF_descriptors.sh
```

This computes Morgan fingerprints and RDKit descriptors for all molecules and saves them to `Data/` directory. The training scripts automatically use pre-computed features when available.

**Output:**
- `Data/features_fp.pkl` - Morgan fingerprints
- `Data/features_desc.pkl` - RDKit descriptors
- `Data/features_metadata.pkl` - Metadata

**Benefits:**
- Speeds up hyperparameter optimization by 10-100× (features computed once, reused for all trials)
- No changes to workflow needed - scripts automatically detect and use cached features

### Step 1: Hyperparameter Optimization

Run Optuna optimization to find the best hyperparameters:

```bash
bash hyperparameter_optimization.sh
```

or

```bash
chmod +x hyperparameter_optimization.sh
./hyperparameter_optimization.sh
```

**Output:**
- `hyperparameter_optimization/hy.txt` - All trial results (hyperparameters and summary metrics)
- `hyperparameter_optimization/best_hyperparameters.txt` - Best hyperparameters

**Configuration:** Edit `config.yaml` to change:
- Number of Optuna trials (`optuna.n_trials`)
- Hyperparameter search ranges (`optuna.search_space`)

### Step 2: Final Training and Evaluation

Run transfer learning with the best hyperparameters:

```bash
bash transfer_learning.sh
```

or

```bash
chmod +x transfer_learning.sh
./transfer_learning.sh
```

**Note:** If `best_hyperparameters.txt` doesn't exist, the script will use default values from `config.yaml`.

## Outputs

All final results are saved to the `outputs/` directory:

### Metrics
- **`outputs/metrics.json`** - Complete metrics for DFT and experimental splits
  - R², MAE, RMSE for each split
  - Per-fold results for experimental data
  - Mean and standard deviation across folds

### Plots
All plots saved to `outputs/plots/` with font size 12 and clean styling (no grid):

**DFT Pretraining (size 4.5×6 inches):**
- `dft_training_curves.png` - Training and validation loss vs. epoch
- `dft_train_parity.png` - Parity plot for train set with uncertainty coloring
- `dft_val_parity.png` - Parity plot for validation set with uncertainty coloring
- `dft_test_parity.png` - Parity plot for test set with uncertainty coloring
- `dft_train_calibration.png` - Uncertainty calibration for train set
- `dft_val_calibration.png` - Uncertainty calibration for validation set
- `dft_test_calibration.png` - Uncertainty calibration for test set

**Experimental Fine-tuning (size 4.5×4.5 inches):**
- `exp_train_parity.png` - Combined train parity plot (all folds, colored by fold)
- `exp_val_parity.png` - Combined validation parity plot (out-of-fold predictions)
- `exp_train_calibration.png` - Combined train calibration plot
- `exp_val_calibration.png` - Combined validation calibration plot

## Dependencies

Install required packages:

```bash
pip install torch numpy pandas scikit-learn rdkit optuna pyyaml matplotlib
```

Or with conda:

```bash
conda install pytorch numpy pandas scikit-learn rdkit optuna pyyaml matplotlib -c conda-forge
```

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Feature settings (fingerprint radius, bits)
- Model architecture defaults
- Training hyperparameters (learning rates, epochs, batch sizes)
- Optuna search space (ranges for all hyperparameters)
- Output directories
- Plotting settings

## Key Features

### Robustness
- **NaN/Inf Sanitization**: Automatically handles invalid descriptor values for edge-case molecules
- **Pre-computed Features**: Optional caching system for faster training
- **Polymer-based Splits**: All splits based on unique SMILES to prevent data leakage

### Performance
- **MC Dropout**: Uncertainty estimation with 50 forward passes
- **Hyperparameter Optimization**: Comprehensive search over architecture and training parameters
- **GPU Support**: Automatically uses CUDA when available

### Reproducibility
- Fixed random seeds for DFT splits
- Configurable seeds for experimental data splits
- All hyperparameters and metrics logged

## Notes

- **No Early Stopping**: Both pretraining and fine-tuning run for a fixed number of epochs
- **Feature Pre-calculation**: Highly recommended for hyperparameter optimization (10-100× speedup)
- **Batch Sizes**: Both pretraining and fine-tuning batch sizes are optimizable hyperparameters
- **Plot Styling**: Clean plots without grid lines; DFT plots use taller aspect ratio (4.5×6)

## References

- **RDKit**: Chemical informatics library for molecular descriptors and fingerprints
- **Optuna**: Hyperparameter optimization framework
- **PyTorch**: Deep learning framework
- **MC Dropout**: Gal & Ghahramani (2016) - Dropout as a Bayesian Approximation
