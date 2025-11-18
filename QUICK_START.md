# Quick Start Guide

Get up and running with the Polymer χ(T) + Solubility ML framework in minutes.

## 1. Environment Setup (5 minutes)

```bash
# Clone/navigate to repository
cd polymer_water_interaction

# Create conda environment
conda env create -f env.yml

# Activate environment
conda activate polymer_chi_ml

# Verify installation
python -c "import torch; import rdkit; print('✓ Environment ready!')"
```

## 2. Check Your Data (1 minute)

Ensure these files exist in the `Data/` directory:
- `OMG_DFT_COSMOC_chi.csv` (~47,676 rows)
- `Experiment_chi_data.csv` (~40 measurements)
- `Binary_solubility.csv` (430 polymers)

Expected columns:
- DFT: `SMILES`, `chi`, `temp`
- Exp: `SMILES`, `chi`, `temp`
- Sol: `SMILES`, `soluble` (0 or 1)

## 3. Configure Settings (2 minutes)

Edit `configs/config.yaml` to adjust key settings:

```yaml
# Essential settings to review:
paths:
  dft_chi_csv: "Data/OMG_DFT_COSMOC_chi.csv"
  exp_chi_csv: "Data/Experiment_chi_data.csv"
  solubility_csv: "Data/Binary_solubility.csv"

training:
  device: "cuda"  # Change to "cpu" if no GPU
  num_epochs_pretrain: 200
  num_epochs_finetune: 200

model:
  encoder_latent_dim: 128
  T_ref_K: 298.0  # Reference temperature
```

## 4. Run Training Pipeline

### Option A: Full Pipeline (Recommended)

```bash
# Step 1: DFT Pretraining (~1-2 hours on GPU)
bash scripts/run_pretrain_dft.sh

# Step 2: Multi-Task Fine-Tuning (~1-2 hours)
# Replace <timestamp> with actual timestamp from Step 1
bash scripts/run_multitask.sh \
    configs/config.yaml \
    results/dft_pretrain_<timestamp>/checkpoints/best_model.pt

# Step 3: K-Fold Cross-Validation (Optional, ~2-3 hours)
bash scripts/run_exp_chi_cv.sh
```

### Option B: Quick Test Run

Edit `configs/config.yaml` to reduce epochs:

```yaml
training:
  num_epochs_pretrain: 10
  num_epochs_finetune: 10
```

Then run the pipeline as above.

## 5. Check Results

Results are saved in timestamped directories under `results/`:

```
results/
├── dft_pretrain_<timestamp>/
│   ├── checkpoints/best_model.pt
│   ├── metrics_summary.json
│   ├── predictions_dft_test.csv
│   └── figures/dft_parity.png
│
└── multitask_<timestamp>/
    ├── checkpoints/best_model.pt
    ├── metrics_summary.json
    ├── predictions_polymer_test.csv
    └── figures/
        ├── exp_parity.png
        ├── sol_roc_curve.png
        ├── sol_confusion_matrix.png
        └── chi_rt_vs_solubility.png
```

### Quick Metrics Check

```bash
# View DFT pretraining results
cat results/dft_pretrain_*/metrics_summary.json

# View multi-task results
cat results/multitask_*/metrics_summary.json
```

## 6. Common Issues & Solutions

### Issue: CUDA out of memory

**Solution:** Reduce batch size in `configs/config.yaml`:

```yaml
training:
  batch_size_dft: 128  # Reduce from 256
  batch_size_sol: 32   # Reduce from 64
```

### Issue: Invalid SMILES

**Solution:** Check logs for invalid SMILES. The code will skip them automatically and report which ones failed.

```bash
# Check for SMILES errors in logs
grep "invalid" results/*/train.log
```

### Issue: Model not improving

**Solution:** Adjust learning rate or check data quality:

```yaml
training:
  lr_pretrain: 5e-4  # Reduce from 1e-3
  lr_finetune: 1e-4  # Reduce from 3e-4
```

### Issue: Pretrained model path not found

**Solution:** Use tab completion or wildcard:

```bash
# List available pretrained models
ls results/dft_pretrain_*/checkpoints/best_model.pt

# Use wildcard in script
bash scripts/run_multitask.sh \
    configs/config.yaml \
    results/dft_pretrain_*/checkpoints/best_model.pt
```

## 7. Advanced Usage

### Hyperparameter Optimization

```bash
# Run 50 trials (takes several hours)
bash scripts/run_hparam_search.sh configs/config.yaml 50

# Use best config for full training
cp results/hparam_search_*/best_config.yaml configs/config_optimized.yaml
bash scripts/run_pretrain_dft.sh configs/config_optimized.yaml
```

### Custom Analysis

```python
# Load results and perform custom analysis
import pandas as pd
import json

# Load metrics
with open('results/multitask_*/metrics_summary.json', 'r') as f:
    metrics = json.load(f)

# Load predictions
preds = pd.read_csv('results/multitask_*/predictions_polymer_test.csv')

# Custom analysis
high_uncertainty = preds[preds['chi_RT_std'] > 0.5]
print(f"High uncertainty polymers: {len(high_uncertainty)}")
```

### Prediction on New Data

```python
from src.models import MultiTaskChiSolubilityModel
from src.data.featurization import PolymerFeaturizer
from src.utils.config import load_config
import torch

# Load config and model
config = load_config('configs/config.yaml')
model = MultiTaskChiSolubilityModel.load_from_checkpoint(
    'results/multitask_*/checkpoints/best_model.pt',
    config
)

# Featurize new SMILES
featurizer = PolymerFeaturizer(config)
new_smiles = ['*CC(C)C*', '*CC(=O)OCC*']
features = featurizer.featurize(new_smiles)

# Predict
x = torch.tensor(features, dtype=torch.float32)
T = torch.tensor([298.0] * len(new_smiles))

model.eval()
with torch.no_grad():
    outputs = model(x, T, predict_solubility=True)

print("Predictions:")
for i, smi in enumerate(new_smiles):
    chi_RT = outputs['chi_pred'][i].item()
    p_soluble = outputs['p_soluble'][i].item()
    print(f"{smi}: χ_RT={chi_RT:.3f}, P(soluble)={p_soluble:.3f}")
```

## 8. Performance Benchmarks

**Expected performance on default config:**

| Dataset | Metric | Typical Value |
|---------|--------|---------------|
| DFT χ | R² | 0.85-0.95 |
| DFT χ | RMSE | 0.05-0.15 |
| Exp χ | R² | 0.60-0.80 |
| Exp χ | MAE | 0.10-0.20 |
| Solubility | ROC-AUC | 0.75-0.85 |
| Solubility | F1 | 0.70-0.80 |

**Training time (on single GPU):**
- Stage 1 (DFT pretraining): 1-2 hours
- Stage 2 (Multi-task): 1-2 hours
- K-fold CV (5 folds): 2-3 hours
- HPO (100 trials): 12-24 hours

## 9. Getting Help

### Documentation
- [README.md](README.md) - Full documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [prompt.md](prompt.md) - Original specification

### Debugging Tips

1. **Check logs**: All runs save detailed logs in `results/*/train.log`
2. **Validate data**: Ensure CSV files have correct columns
3. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
4. **Reduce complexity**: Start with fewer epochs for testing

### Common Commands

```bash
# Monitor training in real-time
tail -f results/*/train.log

# Check GPU usage
watch nvidia-smi

# Find latest results
ls -lt results/ | head

# Clean old results (be careful!)
rm -rf results/dft_pretrain_OLD_TIMESTAMP/
```

## 10. Next Steps

Once you have successful results:

1. **Analyze Results**: Review figures in `results/*/figures/`
2. **Iterate**: Adjust hyperparameters and retrain
3. **Optimize**: Run hyperparameter search for best performance
4. **Publish**: Use generated figures and metrics in your paper

---

## Quick Reference Card

```bash
# Setup
conda env create -f env.yml
conda activate polymer_chi_ml

# Train
bash scripts/run_pretrain_dft.sh
bash scripts/run_multitask.sh configs/config.yaml results/dft_*/checkpoints/best_model.pt

# Evaluate
bash scripts/run_exp_chi_cv.sh

# Optimize
bash scripts/run_hparam_search.sh configs/config.yaml 100

# Results
cat results/*/metrics_summary.json
ls results/*/figures/
```

---

**Ready to start? Run the first command and you'll have results in ~2-4 hours!**

```bash
bash scripts/run_pretrain_dft.sh
```
