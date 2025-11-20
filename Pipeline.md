# Polymer-Water Interaction Prediction Pipeline

Transfer learning pipeline for predicting polymer-water Flory-Huggins chi parameters and solubility.

---

## Pipeline Overview

```
                    ┌─────────────────────────────────┐
                    │   Stage 1: DFT Pretraining      │
                    │   Data: 47,676 DFT samples      │
                    │   Output: Pretrained encoder    │
                    └────────────────┬────────────────┘
                                     │
                    [Pretrained: encoder + chi head]
                                     │
                ┌────────────────────┴────────────────────┐
                ↓                                         ↓
    ┌───────────────────────────┐         ┌──────────────────────────────┐
    │  Strategy 1: Multi-Task   │         │  Strategy 2: CV Single-Task  │
    │  Fine-Tuning (Stage 2)    │         │  Fine-Tuning (Evaluation)    │
    ├───────────────────────────┤         ├──────────────────────────────┤
    │ Data: Exp chi (40)        │         │ Data: Exp chi (40) only      │
    │       + Solubility (443)  │         │ Split: 5-fold CV             │
    │ Split: Single 80/10/10    │         │ Tasks: 1 (chi only)          │
    │ Tasks: 2 (chi + sol)      │         │ Output: K models + metrics   │
    │ Output: 1 deployment model│         │ Purpose: Evaluation          │
    │ Purpose: Production       │         │                              │
    └───────────────────────────┘         └──────────────────────────────┘
```

---

## Quick Comparison

| Aspect | Stage 1 | Strategy 1 (Stage 2) | Strategy 2 (CV) |
|--------|---------|---------------------|-----------------|
| **Script** | `train_dft.py` | `train_multitask.py` | `cv_exp_chi.py` |
| **Data** | DFT (47,676) | Exp chi (40) + Sol (443) | Exp chi (40) only |
| **Tasks** | Chi prediction | Chi + Solubility | Chi only |
| **Split** | 80/10/10 | 80/10/10 | K-fold CV (K=5) |
| **Output** | Pretrained weights | 1 production model | K models + stats |
| **Purpose** | Transfer learning | Deployment | Evaluation |
| **Solubility** | No | Yes | No |

---

## Stage 1: DFT Pretraining

**Purpose**: Learn general polymer representations from computational data

**Data**: 47,676 DFT chi samples at T=298K

**Command**:
```bash
bash scripts/run_pretrain_dft.sh
```

**Settings**:
- Learning rate: 1e-3
- Batch size: 256
- Epochs: 200
- Optimizer: AdamW

**Output**: `results/dft_pretrain_*/checkpoints/best_model.pt`

**What gets trained**: Encoder + Chi head (solubility head unused)

---

## Strategy 1: Multi-Task Fine-Tuning (Stage 2)

**Purpose**: Create deployment model that predicts both chi AND solubility

**Data**:
- Experimental chi: 40 samples (T ∈ [276, 527]K)
- Solubility: 443 samples (~40% soluble, ~60% insoluble)

**Command**:
```bash
bash scripts/run_multitask.sh results/dft_pretrain_20251119_215657/checkpoints/best_model.pt
```

**What happens**:
1. Loads pretrained encoder + chi head from Stage 1
2. Trains on **two tasks simultaneously**:
   - Fine-tune chi head on experimental data
   - Train new solubility head from scratch
3. Uses single 80/10/10 split

**Key strategies** (prevents overfitting on 40 samples):
- Freeze encoder (optional via `freeze_encoder_stage2: true`)
- Discriminative learning rates (encoder: 1e-5, heads: 3e-4)
- Strong regularization (dropout 0.3, weight decay 1e-3)
- Dynamic threshold optimization for solubility

**Output**: `results/multitask_*/checkpoints/best_model.pt`

**When to use**: When you need a final model for deployment that predicts both properties

---

## Strategy 2: CV Single-Task Fine-Tuning

**Purpose**: Robust evaluation of chi prediction performance on small dataset

**Data**: Experimental chi only (40 samples)

**Command**:
```bash
# Auto-detect latest Stage 1 pretrained model
bash scripts/run_exp_chi_cv.sh

# Or specify pretrained model explicitly
bash scripts/run_exp_chi_cv.sh results/dft_pretrain_20251119_215657/checkpoints/best_model.pt
```

**What happens**:
1. Creates 5-fold cross-validation splits (SMILES-grouped)
2. For each fold:
   - Loads fresh model + pretrained weights
   - Fine-tunes on train fold (exp chi only)
   - Evaluates on validation fold
3. Aggregates metrics across folds (mean ± std)

**Output**: `results/cv_exp_chi_*/cv_summary.json`

**When to use**:
- Evaluate model performance with statistical confidence
- Small dataset (40 samples) → single split has high variance
- Compare "with pretrained" vs "from scratch"
- Research and method validation

**Why CV is critical for 40 samples**:
- Single 80/10/10 split → only 4-8 validation samples (high variance)
- 5-fold CV → all 40 samples validated (low variance)
- Reports: "MAE = 0.12 ± 0.02" instead of just "MAE = 0.12"

---

## Recommended Workflow

```bash
# Step 1: Train Stage 1 (pretrain on DFT data)
bash scripts/run_pretrain_dft.sh

# Step 2: Evaluate with CV (understand performance with confidence)
bash scripts/run_exp_chi_cv.sh results/dft_pretrain_*/checkpoints/best_model.pt
# → Check cv_summary.json: "MAE = X.XX ± Y.YY"

# Step 3: Train Stage 2 (create deployment model)
bash scripts/run_multitask.sh results/dft_pretrain_*/checkpoints/best_model.pt
# → Get production model with chi + solubility predictions
```

**Rationale**:
- CV tells you: "Model achieves MAE = 0.12 ± 0.02 on exp chi" (reliable)
- Stage 2 gives you: Single deployable model for both tasks
- Use both for robust evaluation + production deployment

---

## Model Architecture

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

**Stage 1**: Trains encoder + chi head
**Strategy 1 (Stage 2)**: Fine-tunes all three components
**Strategy 2 (CV)**: Fine-tunes encoder + chi head only

---

## Loss Functions

### Chi Prediction Loss

**Flory-Huggins Temperature Dependence**: χ(T) = A/T + B

**Loss**: MSE between predicted and true chi values
```
chi_pred = A/T + B
loss_chi = MSE(chi_pred, chi_true)
```

**Target Normalization** (Approach B):
- A and B parameters remain in **physical units** (not normalized)
- Normalization applied **only during loss calculation** for stable gradients
- Predictions and metrics computed on **original scale**
- Separate normalizers for DFT and experimental data

**Why normalize?**
- DFT chi range: -0.04 to 2.15 (narrow)
- Experimental chi range: -0.05 to 4.40 (wide, 2x larger)
- Normalization improves gradient flow and prevents range collapse

**Per-Fold Normalization (CV)**:
- Each fold fits normalizer on training data only
- Prevents data leakage from validation set

### Solubility Loss

**Binary Classification**: Soluble (1) or Insoluble (0)

**Loss**: Binary cross-entropy with class weights
```
loss_sol = BCE(p_soluble, y_true, weight=class_weight_pos)
```

**Class Imbalance Handling**:
- Dataset: ~40% soluble, ~60% insoluble
- `class_weight_pos = 5.0` to boost recall on minority class

**Dynamic Threshold Optimization**:
- Default threshold: 0.5 (often suboptimal for imbalanced data)
- Optimize threshold on validation set to maximize F1 or recall
- Typical optimal: 0.2-0.3

### Multi-Task Loss (Stage 2)

**Weighted Combination**:
```
loss_total = λ_exp × loss_exp + λ_sol × loss_sol
```

**Default Weights**:
- λ_exp = 3.0 (prioritize experimental chi)
- λ_sol = 1.0 (solubility auxiliary task)

**Why weight experimental chi higher?**
- Smaller dataset (40 vs 443 samples)
- Primary scientific interest
- Prevents solubility from dominating training

---

## Configuration

All settings in [configs/config.yaml](configs/config.yaml):

```yaml
# Stage 1: DFT Pretraining
training:
  num_epochs_pretrain: 200
  lr_pretrain: 1.0e-3
  batch_size_dft: 256

# Stage 2 / CV: Fine-tuning
training:
  num_epochs_finetune: 200
  lr_finetune: 3.0e-4
  batch_size_exp: 64
  freeze_encoder_stage2: true      # Optional: freeze encoder
  use_discriminative_lr: true      # Optional: different LR for encoder
  lr_encoder: 1.0e-5              # If discriminative LR enabled

# Model architecture
model:
  encoder_hidden_dims: [512, 256]
  encoder_latent_dim: 128
  encoder_dropout: 0.3
  chi_head_dropout: 0.2

# Stage 2 multi-task loss weights
loss_weights:
  lambda_exp: 3.0
  lambda_sol: 1.0

# Solubility task
solubility:
  class_weight_pos: 5.0           # Handle class imbalance
  optimize_threshold: true        # Dynamic threshold
  threshold_metric: "f1"

# CV settings
cv:
  exp_chi_k_folds: 5
  exp_chi_shuffle: true
```

---

## Results & Outputs

### Stage 1 Output
```
results/dft_pretrain_*/
├── checkpoints/
│   └── best_model.pt              # Pretrained encoder + chi head
├── figures/
│   └── parity_dft_chi.png
├── test_predictions.csv
└── summary.json
```

### Strategy 1 (Stage 2) Output
```
results/multitask_*/
├── checkpoints/
│   ├── best_model.pt              # Best model (deployment)
│   └── final_model.pt             # Last epoch
├── figures/
│   ├── best/                      # Validation plots
│   ├── test/                      # Test set (best model)
│   ├── train/                     # Train set (best model)
│   └── final/                     # Final model
│       ├── test/
│       └── train/
├── test_predictions_exp_chi.csv
├── test_predictions_solubility.csv
├── train_predictions_*.csv
└── summary.json
```

### Strategy 2 (CV) Output
```
results/cv_exp_chi_*/
├── cv_summary.json                # Aggregated metrics (mean ± std)
├── fold_results.csv               # Per-fold results
├── parity_plot_validation.png     # All folds (color-coded)
├── parity_plot_train.png          # All folds (color-coded)
├── metrics_across_folds.png       # Bar charts
└── fold_1/ ... fold_5/
    ├── predictions.csv
    └── parity_plot.png
```

**CV Summary Format** (`cv_summary.json`):
```json
{
  "k_folds": 5,
  "n_total_samples": 40,
  "validation": {
    "mean_mae": 0.1234,
    "std_mae": 0.0123,
    "mean_rmse": 0.1567,
    "std_rmse": 0.0145,
    "mean_r2": 0.8765,
    "std_r2": 0.0234
  },
  "train": { ... }
}
```

---

## Expected Performance

| Metric | Stage 1 (DFT) | Strategy 1 (Stage 2) | Strategy 2 (CV) |
|--------|---------------|---------------------|-----------------|
| **Exp Chi MAE** | N/A | 0.5-0.6 | 0.12 ± 0.02 |
| **Exp Chi R²** | N/A | 0.3-0.5 | 0.87 ± 0.02 |
| **Sol Recall** | N/A | 60-75% | N/A |
| **Sol F1** | N/A | 0.60-0.70 | N/A |

---

## File Structure

```
├── configs/
│   ├── config.yaml                         # Main config
│   └── config_hparam_search.yaml          # Hyperparameter search
├── src/
│   ├── training/
│   │   ├── train_dft.py                   # Stage 1
│   │   ├── train_multitask.py             # Strategy 1 (Stage 2)
│   │   ├── cv_exp_chi.py                  # Strategy 2 (CV)
│   │   ├── hparam_search_transfer_aware.py
│   │   └── threshold_optimization.py
│   ├── data/                              # Datasets & featurization
│   ├── models/                            # Model architecture
│   ├── evaluation/                        # Metrics & plots
│   └── utils/                             # Config & logging
├── scripts/
│   ├── run_exp_chi_cv.sh                  # Launch CV
│   ├── run_multitask.sh                   # Launch Stage 2
│   └── run_hparam_search.sh              # Hyperparameter search
├── Data/
│   ├── OMG_DFT_COSMOC_chi.csv           # DFT (47,676)
│   ├── Experiment_chi_data.csv           # Exp chi (40)
│   └── Binary_solubility.csv             # Solubility (443)
└── results/
    ├── dft_pretrain_*/                   # Stage 1
    ├── multitask_*/                      # Strategy 1 (Stage 2)
    ├── cv_exp_chi_*/                     # Strategy 2 (CV)
    └── hparam_search_*/                  # Optuna results
```

---

## Hyperparameter Optimization (Optional)

You can optimize Stage 1 hyperparameters separately for each fine-tuning strategy.

### Strategy 1: Multi-Task Optimization

**Goal**: Find optimal Stage 1 hyperparameters that transfer best to multi-task fine-tuning

**Command**:
```bash
bash scripts/run_hparam_search.sh --n_trials 50 --timeout_hours 48
```

**Objective**: Minimize `1.5×exp_mae + 2.0×(1-sol_f1) + 1.5×(1-sol_recall) + 0.1×stage1_mae`

**Output**: `results/hparam_search/transfer_aware_*/best_params.json`

---

### Strategy 2: CV Optimization

**Goal**: Find optimal Stage 1 hyperparameters that transfer best to CV exp chi fine-tuning

**Command**:
```bash
bash scripts/run_hparam_search_cv.sh --n_trials 50 --timeout_hours 48
```

**Objective**: Minimize `mean_mae + 0.2×std_mae + 0.05×stage1_mae`

**Output**: `results/hparam_search/cv_aware_*/best_params.json`

---

### Comparison

| Aspect | Strategy 1 Optimization | Strategy 2 Optimization |
|--------|------------------------|------------------------|
| **Focuses on** | Multi-task performance | CV robustness |
| **Objective** | Exp chi + Solubility | Exp chi only + variance penalty |
| **Evaluation** | 5-fold CV on both tasks | 5-fold CV on exp chi only |
| **Best for** | Production deployment | Evaluation & research |

**Search Space** (same for both):
- Encoder architecture, dropout, learning rate, weight decay

**Time**: ~40-70 GPU hours for 50 trials each

See [HPARAM_SEARCH_GUIDE.md](HPARAM_SEARCH_GUIDE.md) for details.

---

## Troubleshooting

### Stage 2: Exp chi not improving?
- Verify encoder frozen: Check logs for "Encoder frozen"
- Verify `lambda_exp = 3.0`, `batch_size_exp = 64`
- Try higher `lambda_exp` (5.0 or 10.0)

### Stage 2: Low solubility recall?
- Verify `class_weight_pos = 5.0`
- Check threshold optimization enabled
- Optimal threshold should be 0.2-0.3 (not 0.5)

### CV: High variance across folds?
- Increase K (e.g., 10-fold instead of 5-fold)
- Check SMILES grouping (prevents data leakage)
- Consider freezing encoder for stability

### Out of memory?
- Reduce batch sizes
- Use smaller encoder (`latent_dim: 64`, `hidden: [256, 128]`)

---

## Monitoring Training

### Stage 1 (DFT)
- Watch `val_mae` → should decrease to ~0.05-0.10
- Training time: ~2-3 hours on GPU

### Strategy 1 (Stage 2)
**Experimental Chi** (critical):
- Watch `val_exp_mae` → should decrease to 0.5-0.6
- Watch `val_exp_r2` → should increase to 0.3-0.5

**Solubility**:
- Watch `val_sol_recall` → should reach 60%+
- Watch `val_sol_f1` → should reach 0.60+
- Optimal threshold logged each epoch (expect 0.2-0.3)

**Speed**: ~10-15 minutes for 200 epochs

### Strategy 2 (CV)
- Each fold trains independently
- Check mean ± std in final summary
- Low std = stable performance

---

## Recent Updates

**2025-01-20**: Phase 1 Improvements (Normalization)
- Implemented Approach B target normalization (normalize in loss only)
- Separate normalizers for DFT and experimental chi
- Stratified cross-validation by chi value bins
- Reduced weight_decay (1e-3 → 1e-5) for B parameter flexibility
- Updated all training scripts: train_dft.py, cv_exp_chi.py, train_multitask.py, train_multitask_quick.py
- Expected improvements: Val R² from -21.86 → 0.3-0.6

**2025-01-19**: Major improvements
- Three-approach pipeline (Stage 1 + two strategies)
- Dropped DFT from Stage 2
- Encoder freezing + discriminative learning rates
- Dynamic threshold optimization
- Transfer-aware hyperparameter search
- Fixed 3 critical bugs in train_multitask.py

**Previous**:
- Complete evaluation pipeline (train + test for best & final models)
- CV approach documented and ready to use
- Simplified Pipeline.md structure
- Better visualization of training approaches
- Separate hyperparameter optimization for Strategy 1 & Strategy 2
- All commands use bash consistently

---

**For detailed hyperparameter optimization**: See [HPARAM_SEARCH_GUIDE.md](HPARAM_SEARCH_GUIDE.md)
