# Direct Learning for Chi Parameter Prediction

Train directly on experimental data (no transfer learning) to compare with transfer learning approach.

## Quick Start

Run from parent directory `transfer_learning_chi/`:

### 1. Hyperparameter Optimization (Optional)
```bash
bash run_direct_learning_optimization.sh
```

### 2. Training
```bash
bash run_direct_learning.sh
```

### 3. Results
- Metrics: `direct_learning/Results/metrics.json`
- Plots: `direct_learning/Figures/`

## Direct Learning vs Transfer Learning

| | Direct Learning | Transfer Learning |
|---|---|---|
| **Data** | ~40 exp samples | ~47k DFT + ~40 exp |
| **Training** | Single-stage | DFT pretrain → exp finetune |
| **Initialization** | Random weights | Pretrained weights |

## Files

- `config.yaml` - Configuration (no pretrain section)
- `scripts/train_direct.py` - Training from scratch
- `scripts/optuna_objective.py` - Hyperparameter search
- `scripts/run_direct_learning.py` - Main workflow

Reuses `data_utils.py`, `features.py`, `model.py`, `plotting.py` from parent folder.
