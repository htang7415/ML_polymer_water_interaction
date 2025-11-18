# Project Prompt: Polymer–Water χ(T) + Solubility ML Repo

You are an expert ML engineer and software architect.

Your task: **design and implement a clean, reusable PyTorch repository** for predicting polymer–water interaction parameters and solubility, with uncertainty, based on polymer repeat-unit SMILES.

The repo must be:

- Well-structured and modular.
- Easy to read, extend, and reuse.
- Driven entirely by a **single YAML config file** for all hyperparameters.
- Able to run experiments, hyperparameter optimization, and produce **paper-quality figures and tables**.

---

## 1. Problem Overview

We want a model that takes **polymer repeat-unit SMILES** (with two `*` for connection points) and predicts:

1. **DFT–COSMO-SAC χ(p–water, T_ref)** from ~47,676 data points (room temperature or a fixed reference T).
2. **Experimental χ(p–water, T)** at various temperatures (~40 points), modeled as:
   \[
   \chi(p,T) = \frac{A(p)}{T} + B(p)
   \]
3. **Binary water solubility at room temperature** (430 polymers, 1 = soluble, 0 = insoluble).
4. **Uncertainty** for χ_RT and P(soluble) using **MC Dropout**.

Core design decisions (please implement them):

- **Features**: Morgan fingerprints + a small set of RDKit descriptors from repeat-unit SMILES.
- **Model**:
  - Shared encoder → latent vector `z(p)`.
  - χ(T) head outputs `A(p)` and `B(p)` so χ(T) = A/T + B.
  - Solubility head takes `[z(p), χ_RT(p)]` as input and outputs P(soluble).
  - MC Dropout used at inference for epistemic uncertainty.
- **Training strategy**:
  - Stage 1: Pretrain encoder + χ(T) head on DFT χ.
  - Stage 2: Fine-tune multi-task on DFT χ + experimental χ + solubility.
  - Experimental χ evaluated with **k-fold cross-validation** (~40 data points).
  - Solubility and DFT χ use **80 / 10 / 10** splits.

---

## 2. Repository Structure

Create a repo structure like:

```text
polymer_chi_solubility/
  README.md
  pyproject.toml or setup.cfg
  .gitignore
  env.yml                    # conda env (PyTorch, RDKit, etc.)
  configs/
    config.yaml              # single source of truth for all hyperparameters
  data/
    raw/
      dft_chi.csv
      exp_chi.csv
      solubility.csv
    processed/
      ...                    # cached features, indices, etc.
  src/
    __init__.py
    data/
      __init__.py
      featurization.py       # SMILES → features (Morgan + descriptors)
      datasets.py            # PyTorch Datasets + collate functions
      splits.py              # DFT splits, solubility splits, χ_exp k-fold
    models/
      __init__.py
      encoder.py             # shared encoder MLP
      multitask_model.py     # χ(T) head, solubility head, forward logic
    training/
      __init__.py
      losses.py              # multi-task losses
      train_dft.py           # Stage 1: DFT χ pretraining
      train_multitask.py     # Stage 2: multi-task fine-tuning
      cv_exp_chi.py          # k-fold CV for experimental χ
    evaluation/
      __init__.py
      metrics.py             # regression + classification metrics
      plots.py               # publication-quality figures
      uncertainty.py         # MC Dropout utilities
      analysis.py            # χ_RT vs solubility, A-sign analysis, etc.
    utils/
      __init__.py
      config.py              # YAML loading, validation, config object
      logging_utils.py       # logging, run dirs, saving configs & metrics
      seed_utils.py          # global seeding for reproducibility
  scripts/
    run_pretrain_dft.sh
    run_multitask.sh
    run_exp_chi_cv.sh
    run_hparam_search.sh
    hparam_opt.py            # hyperparameter optimization driver
  results/
    ...                      # auto-created per-run directories


## 3. Data Format Assumptions

Assume three main CSV files in `Data`:

- `OMG_DFT_COSMOC_chi.csv` – COSMO-SAC / DFT χ data
- `Experiment_chi_data.csv` – experimental χ(T) data
- `Binary_solubility.csv.csv` – binary water solubility labels

All use **polymer repeat-unit SMILES with two `*`** as the structural identifier.

---

### 3.1 DFT χ (data/raw/dft_chi.csv)

Each row corresponds to one polymer:

- `SMILES` *(string)*  
  - Polymer repeat-unit SMILES with two `*` indicating connection points.  
  - Example: `*CC(=O)OCC*`

- `chi` *(float)*  
  - COSMO-SAC / DFT-computed χ for the polymer–water system at a reference temperature.

- `temp` *(float, optional if constant)*  
  - Temperature in Kelvin  

**Assumptions:**

- One row per polymer with T
- If multiple temperatures ever appear, the χ(T) head will still use χ(T) = A/T + B, but primary use is at reference T.

---

### 3.2 Experimental χ 

Each row is a single experimental measurement χ(p, T_i):

- `SMILES` *(string)*  
  - Polymer repeat-unit SMILES with two `*`.

- `chi` *(float)*  
  - Experimental χ for polymer–water at the given temperature.

- `temp` *(float)*  
  - Actual measurement temperature in Kelvin

## 4. Featurization

Implement featurization in `src/data/featurization.py`.

Goal: convert **polymer repeat-unit SMILES with two `*`** into a fixed-length numeric feature vector using:

- Morgan fingerprints, and  
- A small set of RDKit descriptors.

### 4.1 SMILES handling

- Input SMILES contain two `*` connection points.
- For feature generation:
  - Replace `*` with a consistent dummy atom (e.g., `"C"`), and document this choice clearly in code and comments.
  - Parse the modified SMILES into an RDKit `Mol`.
- If parsing fails:
  - Log a clear warning or error with the offending SMILES.
  - Optionally skip or collect problematic entries into a separate report.

### 4.2 Morgan fingerprints

- Use RDKit to compute circular fingerprints (Morgan fingerprints):

  - `radius` (configurable, e.g., 2 or 3),
  - `nBits` (configurable, e.g., 1024 or 2048),
  - Output as a binary vector, then cast to `float32`.

- Hyperparameters (`radius`, `nBits`) are defined in `configs/config.yaml` under a `chem` section.

### 4.3 RDKit descriptors

- Compute a configurable set of global molecular descriptors (10–30 descriptors), for example:

  - Molecular weight (MolWt)
  - LogP (Crippen/MolLogP)
  - Topological Polar Surface Area (TPSA)
  - NumHDonors
  - NumHAcceptors
  - FractionCSP3
  - NumAromaticRings
  - NumAliphaticRings
  - NumRotatableBonds
  - FormalCharge
  - (and other simple, robust descriptors)

- The **list of descriptors** should be specified in the YAML config (e.g., `chem.descriptor_list`), so it is easy to adjust.

### 4.4 Feature vector

- Concatenate:

  - Morgan fingerprint (binary vector as float), and
  - Descriptor vector (`float32`)

into a single 1D feature vector `x(p)` for each polymer.

- The resulting `input_dim` is determined at runtime and passed into the model via config or inferred during setup.

### 4.5 Caching

- Implement a caching mechanism in `data/processed/`:

  - After first-time featurization, serialize features (e.g., via pickle, joblib, or NumPy arrays) along with an index mapping from **SMILES → feature row**.
  - On subsequent runs, load from cache instead of recomputing, unless a “force recompute” flag is set.

- Cache should be keyed by:
  - The featurization parameters (Morgan radius, nBits, descriptors used),
  - And possibly a hash of the raw input file(s), to avoid mismatch.

---

## 5. Datasets and Splits

Implement logic in `src/data/datasets.py` and `src/data/splits.py`.

All grouping should be done **by SMILES string** (no separate polymer ID field). Wherever “polymer-level” grouping is mentioned, it means “group by identical `smiles` string”.

### 5.1 Dataset classes

Create PyTorch `Dataset` classes that operate on the featurized data.

#### 5.1.1 DFTChiDataset

- Represents rows from `dft_chi.csv`.
- Each item provides:

  - `x`: feature vector for the polymer (from its `smiles`).
  - `chi_dft`: DFT χ(p–water, T_ref).
  - `temperature_K` or `T_ref`: scalar temperature for χ_dft.
  - `smiles`: the original SMILES string (for grouping/analysis if needed).

#### 5.1.2 ExpChiDataset

- Represents rows from `exp_chi.csv`.
- Each item provides:

  - `x`: feature vector for the polymer.
  - `chi_exp`: experimental χ(p–water, T_i).
  - `temperature_K`: measurement temperature (T_i).
  - `smiles`: the original SMILES string.

- Note: a given `smiles` can appear multiple times at different temperatures; grouping for CV must be done by `smiles`.

#### 5.1.3 SolubilityDataset

- Represents rows from `solubility.csv`.
- Each item provides:

  - `x`: feature vector for the polymer.
  - `soluble`: label (0 or 1).
  - `smiles`: the original SMILES string.

### 5.2 Split logic

Implement in `src/data/splits.py`.

#### 5.2.1 DFT χ splits

- Randomly split the DFT dataset into:

  - `train`: 80%
  - `val`:   10%
  - `test`:  10%

- Use a configurable random seed from the YAML config.
- Store/return indices or masks that map to the underlying dataset.

#### 5.2.2 Solubility splits (80 / 10 / 10)

- Create a **polymer-level** split for solubility based on `smiles`:

  - `train_smiles_sol`: 80% of unique SMILES
  - `val_smiles_sol`:   10% of unique SMILES
  - `test_smiles_sol`:  10% of unique SMILES

- Requirements:

  - All rows with the same `smiles` go into the **same** split.
  - Prefer **stratified splitting** by `soluble` label (based on unique SMILES) to preserve class balance in each split.

- Then map these SMILES sets back to dataset indices.

#### 5.2.3 Experimental χ k-fold CV

- For polymers that appear in `exp_chi.csv`:

  - Construct **k-fold (e.g., k = 5) cross-validation** at the SMILES level.
  - Steps:
    - Get the list of unique `smiles` that have χ_exp data.
    - Perform k-fold split on this list (optionally stratified by some property like average χ or temp range).
    - For each fold, determine which rows (indices) belong to train vs validation based on their `smiles`.

- For each CV fold, return:

  - `train_smiles_exp` and `val_smiles_exp`, and/or
  - index lists for χ_exp rows belonging to train vs validation.

- This CV is used **only for evaluating** the χ(T) head on the small experimental dataset; there is no separate fixed 80/10/10 split for exp χ.

---

## 6. Model Architecture

Implement model components in `src/models/encoder.py` and `src/models/multitask_model.py`.

### 6.1 Shared encoder

- Input: feature vector `x(p)` of dimension `input_dim`.
- Output: latent representation `z(p)` of dimension `latent_dim`.

- Architecture (default, configurable via YAML):

  - Linear(`input_dim` → 512)  
    → BatchNorm  
    → ReLU  
    → Dropout(p = 0.2)

  - Linear(512 → 256)  
    → BatchNorm  
    → ReLU  
    → Dropout(p = 0.2)

  - Linear(256 → `latent_dim`)  
    → ReLU

- `latent_dim` (default 128) and `hidden_dims`, activations, and dropout probabilities should be configurable via `config.yaml`.

- Implement a clean `Encoder` class with:

  - Type hints,
  - Docstrings,
  - A method `forward(x) -> z`.

### 6.2 χ(T) head: χ(T) = A/T + B

- Implement a χ head that takes `z(p)` and outputs scalars `A(p)` and `B(p)`.

- Example architecture (configurable):

  - Linear(`latent_dim` → 64)  
    → ReLU  
    → Dropout(p = 0.1)

  - Linear(64 → 2) → `[A, B]`

- χ prediction at temperature T (Kelvin):

  - χ_pred(p, T) = A(p) / T + B(p)

- No explicit constraints on A (so A can be positive or negative), allowing different T-dependence (LCST-like or UCST-like behavior).

### 6.3 Solubility head using χ_RT explicitly

- Let the reference temperature `T_ref` (Kelvin) be defined in config (e.g., 298.0).

- For each polymer:

  - Compute `A(p)` and `B(p)` from the χ head.
  - Compute χ_RT(p) = A(p) / T_ref + B(p).

- Solubility head input is the concatenation:

  - `[z(p), χ_RT(p)]`  
  - Dimension: `latent_dim + 1`.

- Example architecture (configurable):

  - Linear(`latent_dim + 1` → 64)  
    → ReLU  
    → Dropout(p = 0.1)

  - Linear(64 → 1) → sigmoid → P_soluble(p) ∈ (0,1)

- This explicitly exposes χ_RT to the classifier, improving interpretability (low χ_RT → likely soluble).

### 6.4 Combined multi-task model

Create a `MultiTaskChiSolubilityModel` class that:

- Owns:

  - The encoder,
  - The χ(T) head,
  - The solubility head.

- Provides clear methods, for example:

  - `encode(x) -> z`
  - `predict_AB(x) -> (A, B)`
  - `predict_chi(x, T) -> chi_pred`
  - `predict_solubility(x, T_ref) -> P_soluble`
  - A unified `forward` that can output requested quantities (e.g., via flags or a structured return).

- Design it so training code can easily:

  - Call the appropriate methods for DFT χ, exp χ, and solubility tasks,
  - Share the encoder and χ head across tasks.

---

## 7. Losses and Multi-Task Training

Implement multi-task loss functions and utilities in `src/training/losses.py`.

### 7.1 Individual loss terms

Define the following per-sample losses:

#### 7.1.1 DFT χ loss `L_DFT` (MSE)

For DFT χ at T_ref:

- Given χ_dft (target), A(p), B(p), and T_ref:

  - χ_pred_dft = A / T_ref + B  
  - L_DFT = (χ_pred_dft − χ_dft)²

Use mean over the DFT batch.

#### 7.1.2 Experimental χ loss `L_exp` (MSE)

For experimental χ at T_i:

- Given χ_exp (target), A(p), B(p), and T_i:

  - χ_pred_exp = A / T_i + B  
  - L_exp = (χ_pred_exp − χ_exp)²

Use mean over the exp χ batch.

#### 7.1.3 Solubility loss `L_sol` (weighted BCE)

For binary solubility labels:

- P_soluble = model output (after sigmoid).
- y ∈ {0,1}.

Use a **weighted binary cross-entropy**:

- Positive class (soluble) and negative class (insoluble) weights come from config (to handle class imbalance).

### 7.2 Combined multi-task loss

For each batch, the model may see different types of samples:

- Some with DFT χ only,
- Some with exp χ only,
- Some with solubility only,
- Some with multiple labels.

Design masking logic so that:

- For each sample, only compute loss terms for labels that exist.
- Aggregate across the batch to compute:

  - `mean L_DFT` over samples with DFT χ,
  - `mean L_exp` over samples with exp χ,
  - `mean L_sol` over samples with solubility labels.

Then compute total loss:

- `L_total = λ₁ * L_DFT + λ₂ * L_exp + λ₃ * L_sol`

where λ₁, λ₂, λ₃ are multi-task weights defined in `config.yaml` under a section like `loss_weights`.

### 7.3 Training strategy (high-level)

- **Stage 1** (pretraining):

  - Train encoder + χ(T) head using only DFT χ data.
  - Use `L_DFT` as the objective.
  - Early stopping on DFT validation set (Val_DFT).

- **Stage 2** (multi-task fine-tuning):

  - Initialize encoder + χ head from Stage 1.
  - Add solubility head.
  - Train on:

    - DFT χ (Train_DFT),
    - Experimental χ (training folds or training subset),
    - Solubility (training SMILES from solubility split),

  using the combined loss `L_total`.

  - Use a smaller learning rate for fine-tuning.
  - Apply early stopping based primarily on polymer-level validation metrics (e.g., solubility performance on Val_smiles_sol and χ_exp performance on validation folds).

- Ensure the training loops are modular and reuse the same loss functions without duplicating logic.

## 8. Training Scripts

Implement training scripts in `src/training/` and small shell wrappers in `scripts/`. All scripts should:

- Load the YAML config.
- Set seeds and logging.
- Create/run a results directory.
- Save metrics, predictions, and figures in a structured way.

### 8.1 Stage 1: DFT χ pretraining (`train_dft.py`)

Purpose: train the encoder + χ(T) head **only** on DFT χ data.

**Inputs:**

- Config file path.
- DFT dataset (train / val / test split created in `splits.py`).
- Features from `featurization.py`.

**Procedure:**

1. Load config.
2. Initialize logging and seeds.
3. Load `dft_chi.csv`, featurize SMILES, build `DFTChiDataset` and DataLoaders.
4. Build model with:
   - Encoder,
   - χ(T) head.
5. Define optimizer, LR scheduler, and `L_DFT` loss using config settings.
6. Train loop:
   - For each epoch:
     - Train on Train_DFT using `L_DFT`.
     - Evaluate on Val_DFT: compute χ metrics (MAE, RMSE, R², Spearman).
     - Log metrics per epoch.
     - Early stopping based on Val_DFT (e.g., lowest RMSE or highest R²).
7. After training:
   - Evaluate on Test_DFT.
   - Save:
     - Model checkpoint (encoder + χ head).
     - Metrics summary (`metrics_summary.json`, `metrics_summary.csv`).
     - Per-sample predictions on Test_DFT (`predictions_dft_test.csv`).
     - A high-quality parity plot (χ_pred vs χ_true) as PNG and PDF.

Wrapper script: `scripts/run_pretrain_dft.sh`  
- Calls `python -m src.training.train_dft --config configs/config.yaml` or similar.

---

### 8.2 Stage 2: Multi-task fine-tuning (`train_multitask.py`)

Purpose: fine-tune encoder + χ(T) head and train solubility head using:

- DFT χ (Train_DFT),
- Experimental χ data (training portion, if used),
- Solubility data (train SMILES subset).

**Inputs:**

- Config file path.
- Paths to:
  - Pretrained encoder + χ head checkpoint (from Stage 1),
  - Solubility and exp χ CSVs.

**Procedure:**

1. Load config, set seeds, initialize logging.
2. Load and featurize SMILES for DFT, exp χ, and solubility datasets.
3. Build DataLoaders for:
   - DFT (Train_DFT, maybe Val_DFT),
   - Solubility (train / val / test SMILES),
   - Exp χ (optional non-CV usage, or small training subset).
4. Build model:
   - Load encoder + χ head weights from Stage 1.
   - Add and initialize solubility head.
5. Define multi-task optimizer, scheduler, and combined loss `L_total`:
   - `L_total = λ₁ * L_DFT + λ₂ * L_exp + λ₃ * L_sol`.
6. Training loop:
   - Sample batches from DFT, exp χ, and solubility (e.g., via separate loaders or a combined iterator).
   - Compute appropriate losses per batch (masking missing labels).
   - Backpropagate `L_total`.
   - At regular intervals, evaluate on:
     - Val solubility SMILES (`val_smiles_sol`),
     - Optional held-out exp χ subset (if not purely CV).
   - Use early stopping based on validation metrics (e.g., solubility ROC-AUC + χ_exp MAE).
7. After training:
   - Evaluate solubility on Test_SMILES (test split).
   - Optionally evaluate exp χ on a held-out set.
   - Save:
     - Full model checkpoint.
     - Metrics summary for solubility and χ (JSON + CSV).
     - Per-SMILES predictions (`predictions_polymer_test.csv`).
     - Plots: ROC, PR, calibration, confusion matrix, χ_RT vs solubility, etc.

Wrapper script: `scripts/run_multitask.sh`.

---

### 8.3 Experimental χ k-fold CV (`cv_exp_chi.py`)

Purpose: robustly evaluate χ(T) head on the small experimental χ dataset via **SMILES-level k-fold CV**.

**Procedure:**

1. Load config and set seeds.
2. Load and featurize exp χ and solubility data.
3. Construct k folds over unique SMILES with χ_exp using `splits.py`.
4. For each fold `j = 1..k`:
   - Define which SMILES belong to training vs validation for exp χ.
   - Start from the same pretrained encoder + χ head (Stage 1 weights).
   - Optionally attach the solubility head and use Train_SMILES_sol in training.
   - Train multi-task (DFT + solubility + exp χ(train folds)), using `L_total`.
   - Evaluate χ_exp metrics (MAE, RMSE, R², Spearman) on exp χ(val fold).
   - Optionally collect solubility metrics for the subset of polymers with χ_exp.
5. Aggregate metrics across folds:
   - Compute mean ± std for each χ metric.
6. Save:
   - Per-fold metrics (CSV/JSON).
   - Aggregated metrics summary.
   - Optional combined parity plot for χ_exp across folds.

Wrapper script: `scripts/run_exp_chi_cv.sh`.

---

## 9. Hyperparameter Configuration (Single YAML)

All hyperparameters and paths should be centralized in `configs/config.yaml`. No hyperparameters should be hard-coded in Python files.

**Sections to include:**

```yaml
seed: 42

paths:
  dft_chi_csv: data/raw/dft_chi.csv
  exp_chi_csv: data/raw/exp_chi.csv
  solubility_csv: data/raw/solubility.csv
  processed_dir: data/processed
  results_dir: results
  pretrained_dft_checkpoint: results/dft_pretrain/best_model.pt

chem:
  morgan_radius: 2
  morgan_n_bits: 2048
  descriptor_list: [MolWt, LogP, TPSA, NumHDonors, NumHAcceptors]
  smiles_dummy_replacement: "C"

training:
  device: "cuda"
  batch_size_dft: 256
  batch_size_poly: 64
  num_epochs_pretrain: 200
  num_epochs_finetune: 200
  optimizer: "adamw"
  lr_pretrain: 1e-3
  lr_finetune: 3e-4
  weight_decay: 1e-4
  early_stopping_patience: 20
  num_workers: 4

model:
  encoder_hidden_dims: [512, 256]
  encoder_latent_dim: 128
  encoder_dropout: 0.2
  chi_head_hidden_dim: 64
  chi_head_dropout: 0.1
  sol_head_hidden_dim: 64
  sol_head_dropout: 0.1
  T_ref_K: 298.0

loss_weights:
  lambda_dft: 0.5
  lambda_exp: 2.0
  lambda_sol: 1.0

solubility:
  class_weight_pos: 2.0      # soluble
  class_weight_neg: 1.0      # insoluble
  decision_threshold: 0.5

cv:
  exp_chi_k_folds: 5
  exp_chi_shuffle_seed: 42

uncertainty:
  mc_dropout_samples: 50

## 10. Hyperparameter Optimization

Implement hyperparameter optimization in `scripts/hparam_opt.py` and a shell wrapper `scripts/run_hparam_search.sh`.

**Core requirements:**

- Use a library like **Optuna** (preferred) or a simple random search if external dependencies should be minimized.
- Start from a base `config.yaml` and modify selected hyperparameters per trial (in memory).

**Search space examples:**

- Model:
  - `encoder_latent_dim` (e.g., 64, 128, 256)
  - `encoder_hidden_dims` (e.g., [512, 256], [256, 256])
  - `encoder_dropout`, `chi_head_dropout`, `sol_head_dropout` (e.g., 0.1–0.4)
- Optimization:
  - `lr_pretrain`, `lr_finetune` in log scale (e.g., 1e-4–1e-2)
  - `weight_decay` (e.g., 1e-5–1e-3)
- Multi-task:
  - `lambda_dft`, `lambda_exp`, `lambda_sol`
- Solubility:
  - `class_weight_pos` (e.g., 1.0–5.0)
  - `decision_threshold` (e.g., 0.3–0.7)

**Objective:**

- Define a scalar objective that combines:

  - χ_exp performance (e.g., MAE from k-fold CV), and  
  - Solubility performance (e.g., ROC-AUC on validation split).

- Example:
  - `objective = roc_auc_val - alpha * mae_exp_cv`  
    where `alpha` is a constant controlling the trade-off between regression and classification performance.

**Workflow per trial:**

1. Sample hyperparameters from the search space.
2. Construct an in-memory config object based on the base YAML plus trial modifications.
3. Run a shortened training pipeline:
   - Option 1: Use a fixed pretrained encoder and only fine-tune (faster).
   - Option 2: Include a shorter DFT pretraining phase for each trial.
4. Evaluate:
   - χ_exp CV metrics (possibly with fewer folds or fewer epochs for speed).
   - Solubility metrics on validation SMILES.
5. Compute the objective and report it to the optimizer.
6. Log all trial information (hyperparameters + metrics).

**Outputs:**

- A CSV/JSON logging all trials with:

  - Trial ID,  
  - Hyperparameters,  
  - Objective,  
  - Key metrics (χ_exp MAE/RMSE/R²/Spearman, solubility ROC-AUC/F1/MCC, etc.).

- The best hyperparameter set written to:

  - `results/hparam_search/best_config.yaml`.

---

## 11. Evaluation, Metrics, and Publication-Quality Plots

Implement evaluation in `src/evaluation/metrics.py` and `src/evaluation/plots.py`.

### 11.1 Metrics

**For DFT χ and experimental χ:**

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- R² (coefficient of determination).
- Spearman rank correlation.

**For solubility classification:**

- ROC-AUC.
- PR-AUC (for soluble class).
- Accuracy.
- Balanced accuracy.
- Precision, recall, and F1 for the soluble class.
- MCC (Matthews correlation coefficient).
- Brier score.
- Confusion matrix at chosen decision threshold.

Provide reusable functions, for example:

- `regression_metrics(y_true, y_pred) -> dict`
- `classification_metrics(y_true, y_prob, threshold) -> dict`

### 11.2 Figures (journal style)

Implement high-quality plots in `plots.py`:

1. **DFT χ parity plot**  
   - x-axis: χ_true (Test_DFT).  
   - y-axis: χ_pred.  
   - Draw y = x diagonal.  
   - Annotate MAE, RMSE, and R² in the figure or caption.

2. **Experimental χ parity plot**  
   - Combine predictions from all CV folds.  
   - x-axis: χ_true (exp).  
   - y-axis: χ_pred.  
   - Color points by temperature (e.g., colormap over `temperature_K`).  
   - Include y = x diagonal.

3. **Residual vs temperature plot for exp χ**  
   - x-axis: T (K).  
   - y-axis: residual = χ_pred − χ_true.  
   - Horizontal line at 0 for reference.

4. **Solubility ROC curve**  
   - ROC on the solubility test split.  
   - Include ROC-AUC in legend or text.

5. **Solubility PR curve**  
   - Precision–Recall for soluble as positive class.  
   - Include PR-AUC.

6. **Calibration / reliability plot**  
   - Bin predicted probabilities (e.g., 10 bins).  
   - Plot predicted mean vs empirical fraction soluble.  
   - Add diagonal line for perfect calibration.

7. **Confusion matrix heatmap**  
   - Show counts or normalized values.  
   - Label axes with “True” vs “Predicted”, and class names.

8. **χ_RT vs solubility**  
   - Compare predicted χ_RT distributions for soluble vs insoluble polymers.  
   - Box plots or violin plots by class.

9. **Uncertainty vs error plots**  
   - For χ_exp:
     - Bin by χ_RT_std (from MC Dropout).
     - Plot mean |χ_error| per bin.
   - For solubility:
     - Bin by P_std (or entropy).
     - Plot misclassification rate per bin.

**Styling:**

- Clear axis labels with units (T in K, χ dimensionless).
- Consistent font sizes and line widths suitable for journal figures.
- Tight layout and minimal clutter.
- Save both **PNG and PDF** versions under `figures/`.

### 11.3 Results directory structure

Each experiment (run) should create a directory under `results/`, for example:

```text
results/
  chi_solubility_<timestamp>/
    config_used.yaml
    metrics_summary.json
    metrics_summary.csv
    predictions_dft_test.csv
    predictions_polymer_test.csv
    exp_chi_cv_metrics.csv
    figures/
      dft_parity.png
      exp_parity.png
      exp_residual_vs_T.png
      sol_roc_curve.png
      sol_pr_curve.png
      sol_calibration.png
      sol_confusion_matrix.png
      chi_rt_vs_solubility.png
      uncertainty_vs_error_chi.png
      uncertainty_vs_error_sol.png

## 12. Uncertainty via MC Dropout

Implement uncertainty utilities in `src/evaluation/uncertainty.py`.

### 12.1 MC Dropout behavior

- During training:
  - Use Dropout in encoder and heads for regularization.
- During inference:
  - Keep Dropout layers **active** to sample from a quasi-posterior over weights (MC Dropout).

### 12.2 API

Provide helper functions such as:

- `enable_mc_dropout(model)`:
  - Puts the model in eval mode, but forces Dropout layers to behave in training mode (so they still random-drop units).

- `mc_predict(model, x_tensor, T_ref, n_samples)`:
  - Runs the model `n_samples` times on the same input features with Dropout active.
  - From each pass:
    - Extract χ_RT (A/T_ref + B),
    - Extract P_soluble.
  - Returns:
    - `chi_mean`, `chi_std`,
    - `p_mean`, `p_std`.

Use the number of samples from `config.uncertainty.mc_dropout_samples`.

### 12.3 Using uncertainty in evaluation

- For χ_exp:
  - Compare χ_std to |χ_pred − χ_true|.
  - Higher std should correspond to higher error if uncertainty is meaningful.

- For solubility:
  - Compare P_std (or predictive entropy) to misclassification.
  - Check if high-uncertainty predictions are more often wrong.

### 12.4 Using uncertainty in screening (optional, but plan-ready)

Although screening code may be out of scope, design everything so that:

- For each hypothetical polymer, one can retrieve:
  - `chi_mean`, `chi_std`, `p_mean`, `p_std` at T_ref.
- Then apply rules, such as:
  - High-confidence soluble:
    - `p_mean > 0.8` and `p_std < 0.1` and `chi_mean < chi_threshold`.
  - Uncertain but promising:
    - `0.6 < p_mean < 0.8` and `p_std` large.

---

## 13. Logging, Seeds, and Reproducibility

Implement utilities in `src/utils/logging_utils.py` and `src/utils/seed_utils.py`.

### 13.1 Seeding

- Provide a function:

  - `set_seed(seed: int)` that sets:
    - Python `random.seed(seed)`,
    - NumPy `np.random.seed(seed)`,
    - PyTorch CPU and CUDA seeds,
    - Any additional backend settings (e.g., `torch.backends.cudnn.deterministic` if desired).

- Call this at the start of every script using `config.seed`.

### 13.2 Logging

- Configure a logger that:
  - Logs to console (stdout),
  - Logs to a file in each run’s result directory (e.g., `train.log`).
- Log at least:
  - Start and end of each run,
  - Config used,
  - Model size (number of parameters),
  - Data split sizes,
  - Key metrics per epoch (train + val),
  - Final test metrics.

### 13.3 Saving config and metadata

- For each run directory:
  - Save the exact `config_used.yaml` (resolved config).
  - Optionally save:
    - Git commit hash (if repo is in Git) to a file like `git_info.txt`.
    - Machine/device info.

---

## 14. README

Create a clear `README.md` at the repo root. It should include:

1. **Project description:**
   - Short overview of the goal:
     - Predict polymer–water χ(T) and solubility from repeat-unit SMILES,
     - Use multi-task learning and MC Dropout for uncertainty,
     - Enable screening for water-soluble polymers.

2. **Data requirements:**
   - Description of `dft_chi.csv`, `exp_chi.csv`, and `solubility.csv` formats.
   - Example rows or schemas.

3. **Environment setup:**
   - How to create and activate the conda environment from `env.yml`.
   - Any additional installation notes (e.g., RDKit).

4. **Config file:**
   - Explain that `configs/config.yaml` controls all hyperparameters and paths.
   - Show how to modify key fields (paths, training hyperparameters, model sizes).

5. **How to run:**
   - DFT pretraining:
     - `bash scripts/run_pretrain_dft.sh`
   - Multi-task training:
     - `bash scripts/run_multitask.sh`
   - Experimental χ k-fold CV:
     - `bash scripts/run_exp_chi_cv.sh`
   - Hyperparameter optimization:
     - `bash scripts/run_hparam_search.sh`

6. **Outputs:**
   - Explain where results are stored.
   - Brief description of main outputs:
     - Metrics files,
     - Predictions,
     - Figures,
     - Best configs and checkpoints.

7. **Extensibility:**
   - Note that the encoder can be swapped (e.g., to a GNN) without changing training code significantly.
   - Mention that additional tasks (properties) can be plugged into the multi-task framework.

---

## 15. Code Quality Guidelines

To ensure the repo is high quality and reusable:

- **Type hints and docstrings:**
  - Use Python type hints for all functions and methods.
  - Add docstrings explaining:
    - Inputs,
    - Outputs,
    - Any assumptions or side effects.

- **PEP8 and style:**
  - Follow PEP8-style conventions for naming and formatting.
  - Avoid overly long functions; break logic into smaller, testable units.

- **No hard-coded magic numbers:**
  - All hyperparameters and important constants should come from `config.yaml`.
  - No hidden assumptions about paths or dataset sizes.

- **Modular design:**
  - Keep:
    - Featurization separate from datasets,
    - Models separate from training loops,
    - Metrics separate from plotting,
    - Config and logging in dedicated utility modules.
  - This makes it easy to:
    - Swap encoders,
    - Add new heads (e.g., for other properties),
    - Change featurization.

- **Clear error handling:**
  - For SMILES parsing failures or missing values, provide informative logs.
  - Fail fast for critical errors, but allow optional graceful skipping for individual samples when appropriate.

- **Reproducibility:**
  - Always log the seed and config used.
  - Ensure that running the same script with the same config reproduces the same splits and results (within expected randomness for MC Dropout).

- **Documentation in code:**
  - When a design choice is non-obvious (e.g., replacing `*` with `C`, using χ(T)=A/T+B), explain briefly in comments.

The final repo should be ready for:

- Running multiple experiments with different configs,
- Producing metrics and figures suitable for direct use in a journal paper,
- Serving as a base for future polymer ML projects (e.g., swapping to a graph encoder or adding other solvent systems).
