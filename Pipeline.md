# Polymer-Water Interaction Prediction Pipeline

Two-stage transfer learning for predicting polymer-water Flory-Huggins chi parameters and solubility using limited experimental data.

---

## Quick Start

```bash
# Stage 1: DFT Pretraining (learn polymer representations)
python -m src.training.train_dft_pretrain --config configs/config.yaml

# Stage 2: Multitask Fine-tuning (experimental chi + solubility)
python -m src.training.train_multitask \
    --config configs/config.yaml \
    --pretrained results/dft_pretrain_*/checkpoints/best_model.pt

# Optional: Hyperparameter Optimization (find best Stage 1 hyperparameters for Stage 2 transfer)
bash scripts/run_hparam_search.sh --n_trials 50 --timeout_hours 48
```

---

## Overview

**Goal**: Predict polymer solubility in water with minimal experimental data
- 40 experimental chi measurements
- 443 binary solubility labels

**Approach**: Two-stage transfer learning
1. Stage 1: Pretrain on large DFT dataset (47,676 samples)
2. Stage 2: Fine-tune on experimental data with frozen encoder

**Challenge**: Avoid overfitting on tiny experimental datasets

---

## Architecture

```
SMILES → Morgan FP (2048-bit) + RDKit Descriptors (13) → [2061-dim features]
                                ↓
                    Shared Encoder [512 → 256 → 128]
                                ↓
              ┌─────────────────┴─────────────────┐
              ↓                                   ↓
      Chi Head [128 → 64 → 2]          Solubility Head [128+1 → 64 → 1]
      Outputs: (A, B)                   Inputs: [z, chi_RT]
      χ(T) = A/T + B                    Outputs: P(soluble)
```

---

## Stage 1: DFT Pretraining

**Data**: ~47,676 DFT chi samples (T=298K)

**Training**:
```bash
python -m src.training.train_dft_pretrain --config configs/config.yaml
```

**Settings**:
- Learning rate: 1e-3
- Batch size: 256
- Epochs: 200
- Optimizer: AdamW

**Output**: `results/dft_pretrain_*/checkpoints/best_model.pt`

---

## Stage 2: Multitask Fine-tuning

**Data**:
- Experimental chi: 40 samples (T ∈ [276, 527]K, chi ∈ [-0.05, 4.4])
- Solubility: 443 samples (~40% soluble, ~60% insoluble)

**Training**:
```bash
python -m src.training.train_multitask \
    --config configs/config.yaml \
    --pretrained results/dft_pretrain_*/checkpoints/best_model.pt
```

**Key Strategy** (prevents overfitting on 40 samples):
1. **Drop DFT data** → Focus 100% on target tasks
2. **Freeze encoder** → Only fine-tune task heads
3. **Discriminative LR** → Encoder: 1e-5, Heads: 3e-4
4. **Dynamic threshold** → Optimize decision boundary for recall
5. **Strong regularization** → Dropout 0.3, weight decay 1e-3

**Loss**:
```
Total = 3.0 × L_exp_chi + 1.0 × L_solubility
        (weighted BCE with pos_weight=5.0)
```

**Output**: `results/multitask_*/checkpoints/best_model.pt`

---

## Hyperparameter Optimization (Optional)

**Goal**: Find optimal Stage 1 hyperparameters that transfer best to Stage 2

**Usage**:
```bash
# Run with default settings (50 trials, 48 hours)
bash scripts/run_hparam_search.sh

# Custom settings
bash scripts/run_hparam_search.sh --n_trials 25 --timeout_hours 24
```

**Search Space**:
- Encoder latent dim: [64, 128, 256]
- Encoder hidden dims: ["512_256", "1024_512_256", "256_128"]
- Encoder dropout: [0.15, 0.35]
- Chi head dropout: [0.05, 0.25]
- Learning rate: [5e-4, 2e-3]
- Weight decay: [1e-5, 1e-3]

**Objective**: Minimize `1.5×exp_mae + 2.0×(1-sol_f1) + 1.5×(1-sol_recall) + 0.1×stage1_mae`

**Time**: ~150 GPU-hours for 50 trials (6-7 days on single GPU)

**Output**: `results/hparam_search/*/best_params.json`

See [HPARAM_SEARCH_GUIDE.md](HPARAM_SEARCH_GUIDE.md) for details.

---

## Key Configuration

All settings in [configs/config.yaml](configs/config.yaml):

```yaml
training:
  batch_size_exp: 64
  freeze_encoder_stage2: true
  use_discriminative_lr: true
  lr_encoder: 1.0e-5
  lr_finetune: 3.0e-4
  weight_decay: 1.0e-3

model:
  encoder_hidden_dims: [512, 256]
  encoder_latent_dim: 128
  encoder_dropout: 0.3
  chi_head_dropout: 0.2

loss_weights:
  lambda_dft: 0.0    # Dropped in Stage 2
  lambda_exp: 3.0
  lambda_sol: 1.0

solubility:
  class_weight_pos: 5.0
  optimize_threshold: true
  threshold_metric: "f1"
```

---

## Expected Performance

| Metric | Baseline | After Improvements |
|--------|----------|-------------------|
| Exp Chi MAE | 0.82 | 0.5-0.6 (↓35%) |
| Exp Chi R² | -0.06 | 0.3-0.5 |
| Sol Recall | 25% | 60-75% (3x) |
| Sol F1 | 0.36 | 0.60-0.70 |

**Why it works**:
- Frozen encoder prevents overfitting on 40 samples
- No DFT gradient dilution in Stage 2
- Optimized threshold fixes prediction bias
- Higher class weight improves recall

---

## Monitoring Training

**Experimental Chi** (bottleneck):
- Watch `val_exp_mae` → should decrease to 0.5-0.6
- Watch `val_exp_r2` → should increase to 0.3-0.5
- If R² negative → increase λ_exp

**Solubility**:
- Watch `val_sol_recall` → should reach 60%+
- Watch `val_sol_f1` → should reach 0.60+
- Optimal threshold logged each epoch (expect 0.2-0.3)

**Speed**:
- Stage 2: ~10-15 minutes for 200 epochs (5 batches/epoch vs 149 previously)

---

## File Structure

```
├── configs/
│   ├── config.yaml                         # Main config
│   └── config_hparam_search.yaml          # Hyperparameter search config
├── src/
│   ├── training/
│   │   ├── train_dft_pretrain.py          # Stage 1
│   │   ├── train_multitask.py             # Stage 2
│   │   ├── train_multitask_quick.py       # Quick Stage 2 (for hparam search)
│   │   ├── hparam_search_transfer_aware.py # Optuna optimization
│   │   ├── optuna_utils.py                # Study management
│   │   └── threshold_optimization.py      # Dynamic threshold
│   ├── data/                              # Datasets & featurization
│   ├── models/                            # Model architecture
│   ├── evaluation/                        # Metrics & plots
│   └── utils/                             # Config & logging
├── scripts/
│   └── run_hparam_search.sh              # Hyperparameter search launcher
├── Data/
│   ├── OMG_DFT_COSMOC_chi.csv           # DFT data (47,676 samples)
│   ├── Experiment_chi_data.csv           # Experimental chi (40 samples)
│   └── Binary_solubility.csv             # Solubility (443 samples)
└── results/
    ├── dft_pretrain_*/                   # Stage 1 results
    ├── multitask_*/                      # Stage 2 results
    └── hparam_search_*/                  # Optimization results
```

---

## Troubleshooting

**Exp chi not improving?**
- Verify encoder frozen: Check logs for "Encoder frozen"
- Verify λ_exp = 3.0, batch_size_exp = 64
- Try higher λ_exp (5.0 or 10.0)

**Low solubility recall?**
- Verify class_weight_pos = 5.0
- Check threshold optimization enabled
- Optimal threshold should be 0.2-0.3 (not 0.5)
- Try higher class_weight_pos (8.0)

**Out of memory?**
- Reduce batch sizes
- Use smaller encoder (latent_dim: 64, hidden: [256, 128])

---

## Results & Evaluation

### Multitask Training Results

After training, you get comprehensive evaluation for both **best model** (saved during training) and **final model** (last epoch):

```
results/multitask_*/
├── checkpoints/
│   ├── best_model.pt        # Evaluated on train + test
│   └── final_model.pt       # Evaluated on train + test
├── figures/
│   ├── best/                # Validation plots during training
│   │   ├── parity_exp_chi.png
│   │   ├── roc_solubility.png
│   │   └── confusion_matrix_solubility.png
│   ├── test/                # Test set (best model)
│   │   ├── parity_exp_chi.png
│   │   ├── roc_solubility.png
│   │   ├── confusion_matrix_solubility.png
│   │   └── calibration_solubility.png
│   ├── train/               # Train set (best model)
│   │   ├── parity_exp_chi.png
│   │   ├── roc_solubility.png
│   │   ├── confusion_matrix_solubility.png
│   │   └── calibration_solubility.png
│   └── final/               # Final model results
│       ├── test/            # All plots for test set
│       └── train/           # All plots for train set
├── test_predictions_exp_chi.csv           # Best model predictions
├── test_predictions_solubility.csv
├── train_predictions_exp_chi.csv          # Train set predictions
├── train_predictions_solubility.csv
├── *_final.csv              # Final model predictions
└── cv_summary.json          # Final metrics
```

**Binary Classification Plots**:
- ROC curves (with AUC)
- Confusion matrices (TP, TN, FP, FN)
- Calibration plots (probability calibration)

**All parity plots show**: MAE, RMSE, R² only (clean, no clutter)

### Cross-Validation Results

For experimental chi evaluation:

```bash
# Option 1: Using bash script (recommended)
bash scripts/run_exp_chi_cv.sh

# Option 2: With pretrained model
bash scripts/run_exp_chi_cv.sh configs/config.yaml results/dft_pretrain_*/checkpoints/best_model.pt

# Option 3: Direct Python command
python -m src.training.cv_exp_chi --config configs/config.yaml
```

Results structure:
```
results/cv_exp_chi_*/
├── cv_summary.json                    # Train + validation metrics
├── fold_results.csv                   # Per-fold metrics (train + val)
├── parity_plot_validation.png         # All folds (colored by fold)
├── parity_plot_train.png              # All folds (colored by fold)
├── metrics_across_folds.png
└── fold_*/
    ├── predictions.csv
    └── parity_plot.png                # Per-fold plot
```

**Features**:
- Each fold has distinct color in aggregated plots
- Train + validation evaluation (assess overfitting)
- Clean metric boxes: MAE, RMSE, R² only
- Fold-wise variation statistics

---

## Recent Updates

**2025-01-19**: Major improvements
- Dropped DFT from Stage 2
- Encoder freezing + discriminative learning rates
- Dynamic threshold optimization
- Transfer-aware hyperparameter search with Optuna
- Fixed 5 critical bugs

**Latest**: Complete evaluation pipeline
- Train + test evaluation for best & final models
- Fixed confusion matrix generation
- Simplified plot legends (MAE, RMSE, R² only)
- Cross-validation with train set evaluation
- Colored fold visualization in CV plots

---

**For detailed hyperparameter optimization guide**: See [HPARAM_SEARCH_GUIDE.md](HPARAM_SEARCH_GUIDE.md)
