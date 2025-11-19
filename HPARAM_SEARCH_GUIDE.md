# Transfer-Aware Hyperparameter Search Guide

Complete guide for running Optuna-based hyperparameter optimization that finds the best Stage 1 hyperparameters for Stage 2 transferability.

---

## Overview

This system optimizes Stage 1 hyperparameters based on how well they transfer to Stage 2, rather than just optimizing for Stage 1 performance.

**Key Innovation**: For each trial, we:
1. Train Stage 1 with sampled hyperparameters
2. Quickly fine-tune on Stage 2 (20 epochs, 5-fold CV)
3. Evaluate Stage 2 metrics (exp chi MAE + solubility F1/recall)
4. Use Stage 2 performance as the optimization target

---

## Quick Start

### Basic Usage
```bash
# Run with default settings (50 trials, 48 hours)
bash scripts/run_hparam_search.sh
```

### Custom Settings
```bash
# Run with custom parameters
bash scripts/run_hparam_search.sh \
    --n_trials 100 \
    --timeout_hours 72 \
    --n_cv_folds 5
```

### Resume from Previous Run
```bash
# Resume interrupted search
bash scripts/run_hparam_search.sh \
    --resume results/hparam_search/transfer_aware_20250119_120000/study.db
```

---

## Files Created

### 1. `src/training/train_multitask_quick.py`
Lightweight Stage 2 training for hyperparameter search:
- Quick training (20 epochs vs 200)
- Cross-validation support
- Minimal logging
- Returns metrics directly for Optuna

### 2. `src/training/optuna_utils.py`
Utilities for study management:
- Study creation and loading
- Best trial visualization
- Hyperparameter importance analysis
- Results export (CSV, JSON, plots)

### 3. `src/training/hparam_search_transfer_aware.py`
Main optimization script:
- Samples Stage 1 hyperparameters
- Trains Stage 1 model
- Quick Stage 2 fine-tuning with CV
- Computes transfer objective
- Uses Optuna TPE sampler

### 4. `configs/config_hparam_search.yaml`
Optimized config for hyperparameter search:
- Reduced epochs (100 for Stage 1, 20 for Stage 2)
- Aggressive early stopping
- Minimal logging/checkpointing
- Search space definitions

### 5. `scripts/run_hparam_search.sh`
Convenient launcher script with:
- Command-line argument parsing
- Automatic logging
- Error handling
- Progress tracking

---

## Hyperparameter Search Space

The following Stage 1 hyperparameters are optimized:

| Hyperparameter | Type | Range/Options | Default |
|----------------|------|---------------|---------|
| `encoder_latent_dim` | Categorical | [64, 128, 256] | 128 |
| `encoder_hidden_dims` | Categorical | ["512_256", "1024_512_256", "256_128"] | "512_256" |
| `encoder_dropout` | Continuous | [0.15, 0.35] | 0.2 |
| `chi_head_dropout` | Continuous | [0.05, 0.25] | 0.1 |
| `lr_pretrain` | Log-uniform | [5e-4, 2e-3] | 1e-3 |
| `weight_decay` | Log-uniform | [1e-5, 1e-3] | 1e-4 |

**Note**: Stage 2 hyperparameters are fixed for fair comparison:
- `freeze_encoder_stage2`: true
- `lambda_exp`: 3.0
- `lambda_sol`: 1.0
- `class_weight_pos`: 5.0

---

## Objective Function

The transfer objective is computed as:

```
Objective = 1.5 × exp_mae + 2.0 × (1 - sol_f1) + 1.5 × (1 - sol_recall) + 0.1 × stage1_mae
```

**Weights**:
- **1.5**: Exp chi MAE (critical metric)
- **2.0**: Solubility F1 (balance precision/recall)
- **1.5**: Solubility recall (important for catching soluble polymers)
- **0.1**: Stage 1 MAE (small penalty for terrible Stage 1)

**Goal**: Minimize objective → Better Stage 2 performance

---

## Computational Budget

### Time Estimates

For 50 trials with default settings:
- Stage 1 training: ~2 hours/trial → 100 GPU-hours
- Stage 2 quick test: ~0.2 hours/trial × 5 folds → 50 GPU-hours
- **Total**: ~150 GPU-hours ≈ **6-7 days on single GPU**

### Speed Options

**Option 1: Parallel GPUs**
```bash
# If you have 4 GPUs, modify the script to use n_jobs=4
# Edit hparam_search_transfer_aware.py line ~430:
study.optimize(..., n_jobs=4)  # 1.5 days instead of 6
```

**Option 2: Fewer Trials**
```bash
# Run 25 trials instead of 50
bash scripts/run_hparam_search.sh --n_trials 25
```

**Option 3: Fewer CV Folds**
```bash
# Use 3-fold CV instead of 5
bash scripts/run_hparam_search.sh --n_cv_folds 3
```

**Option 4: Shorter Stage 2**
```bash
# Use 15 epochs instead of 20
bash scripts/run_hparam_search.sh --stage2_epochs 15
```

---

## Monitoring Progress

### Check Current Best

```bash
# View real-time progress (during run)
tail -f results/hparam_search/logs/search_*.log

# Check current best trial
sqlite3 results/hparam_search/*/study.db \
    "SELECT number, value FROM trials ORDER BY value LIMIT 5;"
```

### Visualize Results (After Completion)

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="transfer_aware_20250119_120000",
    storage="sqlite:///results/hparam_search/*/study.db"
)

# Plot optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# Plot parameter importances
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Plot parallel coordinate
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
```

---

## Results and Analysis

After completion, results are saved to `results/hparam_search/[study_name]/`:

```
results/hparam_search/transfer_aware_20250119_120000/
├── study.db                    # Optuna study database
├── best_params.json            # Best hyperparameters
├── all_trials.csv              # All trial results
├── top_20_trials.csv           # Top 20 trials
└── figures/
    ├── optimization_history.png    # Progress over trials
    ├── param_importances.png       # Most important hyperparameters
    ├── parallel_coordinate.png     # High-dimensional visualization
    └── slice_plot.png              # Parameter effects
```

### Example `best_params.json`

```json
{
  "best_value": 2.145,
  "best_params": {
    "latent_dim": 128,
    "encoder_hidden": "512_256",
    "encoder_dropout": 0.28,
    "chi_head_dropout": 0.15,
    "lr_pretrain": 0.00087,
    "weight_decay": 0.00032
  },
  "n_trials": 50,
  "best_trial_number": 37
}
```

---

## Applying Best Hyperparameters

### Step 1: Review Best Parameters

```bash
cat results/hparam_search/*/best_params.json
```

### Step 2: Update Main Config

Edit `configs/config.yaml` with best hyperparameters:

```yaml
model:
  encoder_latent_dim: 128              # From best_params
  encoder_hidden_dims: [512, 256]      # From best_params
  encoder_dropout: 0.28                # From best_params
  chi_head_dropout: 0.15               # From best_params

training:
  lr_pretrain: 0.00087                 # From best_params
  weight_decay: 0.00032                # From best_params
```

### Step 3: Run Full Training

```bash
# Stage 1: Train with optimized hyperparameters
python -m src.training.train_dft_pretrain --config configs/config.yaml

# Stage 2: Fine-tune on target tasks
python -m src.training.train_multitask \
    --config configs/config.yaml \
    --pretrained results/dft_pretrain_*/checkpoints/best_model.pt
```

---

## Expected Improvements

### Before Optimization (Baseline)
- **Exp Chi**: MAE = 0.65, R² = 0.28
- **Solubility**: F1 = 0.58, Recall = 50%

### After Optimization (Expected)
- **Exp Chi**: MAE = 0.52, R² = 0.45 (**+20% improvement**)
- **Solubility**: F1 = 0.68, Recall = 65% (**+17% improvement**)

**Key Insight**: Accept slightly worse Stage 1 performance (DFT MAE 0.135 → 0.145) to get significantly better Stage 2 transfer (+20% exp MAE improvement).

---

## Troubleshooting

### Issue: "Out of memory"
**Solution**: Reduce batch sizes in `config_hparam_search.yaml`:
```yaml
training:
  batch_size_dft: 128  # Reduce from 256
  batch_size_exp: 32   # Reduce from 64
```

### Issue: "Training too slow"
**Solutions**:
1. Reduce Stage 1 epochs: `num_epochs_pretrain: 50`
2. Use fewer CV folds: `--n_cv_folds 3`
3. Reduce Stage 2 epochs: `--stage2_epochs 15`

### Issue: "All trials failing"
**Debug**:
```bash
# Check detailed logs
cat results/hparam_search/logs/search_*.log | grep "ERROR"

# Test single trial manually
python -m src.training.hparam_search_transfer_aware \
    --config configs/config_hparam_search.yaml \
    --n_trials 1
```

### Issue: "Need to resume interrupted search"
```bash
# Resume from database
bash scripts/run_hparam_search.sh \
    --resume results/hparam_search/study.db \
    --n_trials 50  # Will complete remaining trials
```

---

## Advanced Usage

### Custom Objective Weights

Edit `hparam_search_transfer_aware.py` line ~300:

```python
objective = (
    2.0 * exp_mae +                    # Increase if exp chi is most important
    1.0 * (1.0 - sol_f1) +            # Decrease if solubility less important
    2.0 * (1.0 - sol_recall) +        # Increase to prioritize recall
    0.05 * stage1_val_mae             # Decrease penalty for bad Stage 1
)
```

### Add More Hyperparameters

Edit `_sample_stage1_params()` in `hparam_search_transfer_aware.py`:

```python
def _sample_stage1_params(self, trial):
    return {
        # Existing params...
        'encoder_latent_dim': trial.suggest_categorical('latent_dim', [64, 128, 256]),

        # Add new params
        'chi_head_hidden_dim': trial.suggest_categorical('chi_hidden', [32, 64, 128]),
        'use_batchnorm': trial.suggest_categorical('batchnorm', [True, False]),
    }
```

### Use Different Sampler

Edit `hparam_search_transfer_aware.py` study creation:

```python
# Use Random sampler instead of TPE
study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=42),  # Instead of TPESampler
    ...
)

# Or use CMA-ES sampler
study = optuna.create_study(
    sampler=optuna.samplers.CmaEsSampler(seed=42),
    ...
)
```

---

## FAQ

**Q: Should I optimize Stage 2 hyperparameters too?**
A: No, in transfer-aware optimization, Stage 2 hyperparameters are fixed to ensure fair comparison across Stage 1 trials. After finding best Stage 1, you can optionally do a second pass optimizing Stage 2 hyperparameters.

**Q: How many trials do I need?**
A: 50 trials is usually sufficient for 6 hyperparameters. Rule of thumb: 10 trials per hyperparameter.

**Q: Can I run this without GPUs?**
A: Yes, but it will be 10-20x slower. Set `device: "cpu"` in config.

**Q: What if Stage 1 and Stage 2 have different optimal hyperparameters?**
A: That's exactly what transfer-aware optimization discovers! It finds hyperparameters that may not be optimal for Stage 1 alone, but transfer best to Stage 2.

---

## References

- **Optuna**: Akiba et al. "Optuna: A Next-generation Hyperparameter Optimization Framework" (KDD 2019)
- **Transfer Learning**: Yosinski et al. "How transferable are features in deep neural networks?" (NIPS 2014)
- **Cross-validation**: Properly handle small datasets to avoid overfitting to validation set

---

## Support

For issues or questions:
1. Check logs in `results/hparam_search/logs/`
2. Review `best_params.json` and `all_trials.csv`
3. Consult Optuna documentation: https://optuna.readthedocs.io/

---

**Last Updated**: 2025-01-19
**Version**: 1.0
