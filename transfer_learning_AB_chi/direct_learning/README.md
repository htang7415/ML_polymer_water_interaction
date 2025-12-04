# Direct Learning for Polymer-Water χ Prediction

This directory contains the implementation for **direct learning** (non-transfer learning) on experimental chi data. It serves as a baseline comparison for the transfer learning approach.

## Overview

**Key Differences from Transfer Learning:**

| Aspect | Transfer Learning (Parent Dir) | Direct Learning (This Dir) |
|--------|-------------------------------|----------------------------|
| Training data | DFT pretraining → Exp fine-tuning | Exp data only |
| Initialization | Pretrained weights from DFT | Random initialization |
| Layer freezing | Optional layer freezing | N/A (all layers trainable) |
| Training phases | 2 phases (pretrain + fine-tune) | 1 phase (direct training) |
| Hyperparameters | `lr_pre`, `lr_ft`, `n_freeze_layers` | `lr` (single learning rate) |

## Directory Structure

```
direct_learning/
├── config.yaml                          # Configuration (no DFT data)
├── scripts/
│   ├── model.py -> ../../scripts/model.py           # Symlink to shared model
│   ├── train.py -> ../../scripts/train.py           # Symlink to shared training
│   ├── data_utils.py -> ../../scripts/data_utils.py # Symlink to shared data utils
│   ├── features.py -> ../../scripts/features.py     # Symlink to shared features
│   ├── plotting.py -> ../../scripts/plotting.py     # Symlink to shared plotting
│   ├── optuna_objective_direct.py       # Hyperparameter optimization (direct learning)
│   └── run_direct_learning.py           # Main training script (direct learning)
├── EXP_features.csv -> ../EXP_features.csv  # Symlink to experimental features
├── hyperparameter_optimization.sh       # Script to run Optuna
├── direct_learning.sh                   # Script to run final training
├── hyperparameter_optimization/         # Optuna results (created after optimization)
│   ├── best_hyperparameters.txt
│   └── hy.txt
└── outputs/                             # Training results (created after training)
    ├── plots/                           # Parity plots, calibration plots
    ├── metrics/                         # metrics.json, metrics.txt
    └── models/                          # Saved models (if needed)
```

## Workflow

### Step 1: Ensure Features Exist

The experimental features should already be computed in the parent directory. Verify the symlink:

```bash
ls -lh EXP_features.csv
```

### Step 2: Hyperparameter Optimization (Recommended)

Run Optuna to find the best hyperparameters for direct learning:

```bash
bash hyperparameter_optimization_direct.sh
```

This will:
- Run 1000 Optuna trials (configurable in `config.yaml`)
- Train models from random initialization on experimental data (5-fold CV)
- Optimize for maximum validation R²
- Save results to `hyperparameter_optimization/`

**Time estimate:** Several hours to days depending on hardware and number of trials.

### Step 3: Train with Best Hyperparameters

Run the final training with optimized hyperparameters:

```bash
bash run_direct_learning.sh
```

This will:
- Load best hyperparameters (or use defaults if optimization was skipped)
- Train 5 models (one per fold) from random initialization
- Evaluate with MC dropout for uncertainty quantification
- Generate plots and save metrics to `outputs/`

## Output Files

After training, you'll find:

### Metrics
- `outputs/metrics/metrics.json` - Machine-readable performance metrics
- `outputs/metrics/metrics.txt` - Human-readable summary:
  ```
  Validation (Out-of-Fold):
  R²:   0.XXXX ± 0.XXXX
  MAE:  0.XXXX ± 0.XXXX
  RMSE: 0.XXXX ± 0.XXXX
  ```

### Plots
- `outputs/plots/exp_val_parity.png` - Validation parity plot (colored by fold)
- `outputs/plots/exp_val_calibration.png` - Uncertainty calibration
- `outputs/plots/exp_train_parity.png` - Training parity plot

## Comparing with Transfer Learning

To compare direct learning vs transfer learning:

1. **Run both approaches:**
   - Transfer learning: `cd .. && bash transfer_learning.sh`
   - Direct learning: `cd direct_learning && bash direct_learning.sh`

2. **Compare metrics:**
   ```bash
   # Transfer learning metrics
   cat ../outputs/metrics/metrics.txt

   # Direct learning metrics
   cat outputs/metrics/metrics.txt
   ```

3. **Compare validation R²:**
   - If transfer learning R² > direct learning R²: Transfer learning helps!
   - If transfer learning R² ≈ direct learning R²: DFT pretraining provides no benefit
   - If transfer learning R² < direct learning R²: Negative transfer (DFT data hurts)

## Configuration

Edit `config.yaml` to customize:

### Hyperparameter Search Space
```yaml
hyperparameters:
  lr: {type: loguniform, low: 1.0e-6, high: 1.0e-2}
  epochs: {type: int, low: 500, high: 2000}
  batch_size: {type: categorical, choices: [1, 2, 4, 8, 16, 32]}
  n_layers: {type: int, low: 2, high: 7}
  dropout_rate: {type: float, low: 0.1, high: 0.4}
  ...
```

### Optuna Settings
```yaml
optuna:
  n_trials: 1000  # Number of trials to run
  direction: maximize  # Maximize validation R²
```

## Key Implementation Details

### Training from Scratch
```python
# Create model with RANDOM initialization (no pretrained weights)
model = create_model(
    input_dim=input_dim,
    hidden_dims=hp['hidden_dims'],
    n_layers=hp['n_layers'],
    dropout_rate=hp['dropout_rate'],
    device=device
)

# Train from scratch (no pretrained weights loaded)
train_model(model, train_loader, val_loader, ...)
```

### Per-Fold Descriptor Scaling
```python
# Scale descriptors using THIS FOLD'S training data
# (different from transfer learning which uses DFT statistics)
train_scaled, val_scaled, _, valid_desc_cols, _, _ = \
    filter_and_scale_descriptors(train_df, val_df, None)
```

## Troubleshooting

### Missing Features File
```
ERROR: EXP_features.csv not found!
```
**Solution:** Create symlink to parent directory's features:
```bash
ln -s ../EXP_features.csv EXP_features.csv
```

### GPU Memory Issues
If you encounter OOM errors, try:
1. Reduce `batch_size` in `config.yaml`
2. Reduce `hidden_dim_per_layer` choices
3. Reduce `n_layers` upper bound

### Poor Performance
If validation R² is low:
1. Run more Optuna trials (increase `n_trials`)
2. Expand hyperparameter search space
3. Check for data quality issues

## Expected Results

Based on typical transfer learning scenarios:

- **If DFT data is relevant:** Transfer learning should outperform direct learning
- **If DFT data is irrelevant:** Direct learning may perform similarly or better
- **Small experimental dataset:** Transfer learning usually provides larger gains

## Questions?

Compare your results with the transfer learning approach in the parent directory to determine if pretraining on DFT data provides benefits for predicting experimental chi values.
