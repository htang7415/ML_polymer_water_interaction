"""
Quick Stage 2 multitask fine-tuning for hyperparameter search.

This is a lightweight version of train_multitask.py designed for:
- Rapid hyperparameter evaluation (20 epochs instead of 200)
- Cross-validation support
- Minimal logging and checkpointing
- Returns metrics directly for Optuna optimization
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.datasets import (
    ExpChiDataset,
    SolubilityDataset,
    collate_exp_chi,
    collate_solubility,
)
from src.data.featurization import PolymerFeaturizer
from src.data.splits import create_exp_chi_splits, create_solubility_splits
from src.models.multitask_model import MultiTaskChiSolubilityModel
from src.training.losses import (
    multitask_loss,
    compute_metrics_regression,
    compute_metrics_classification,
)
from src.training.threshold_optimization import find_optimal_threshold
from src.utils.config import Config
from src.utils.seed_utils import set_seed


def train_epoch_quick(
    model: nn.Module,
    train_loaders: dict,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
) -> float:
    """Quick training epoch (minimal logging)."""
    model.train()
    total_loss = 0.0

    exp_iter = iter(train_loaders["exp"])
    sol_iter = iter(train_loaders["sol"])
    max_batches = max(len(train_loaders["exp"]), len(train_loaders["sol"]))

    for _ in range(max_batches):
        batch_loss = 0.0
        n_tasks = 0

        # Exp chi batch
        try:
            batch_exp = next(exp_iter)
        except StopIteration:
            exp_iter = iter(train_loaders["exp"])
            batch_exp = next(exp_iter)

        x_exp = batch_exp["x"].to(device)
        chi_exp = batch_exp["chi_exp"].to(device)
        temp_exp = batch_exp["temperature"].to(device)

        outputs = model(x_exp, temperature=temp_exp)
        loss_exp, _ = multitask_loss(
            outputs, chi_exp_true=chi_exp, temperature_exp=temp_exp, config=config
        )
        batch_loss += loss_exp
        n_tasks += 1

        # Solubility batch
        try:
            batch_sol = next(sol_iter)
        except StopIteration:
            sol_iter = iter(train_loaders["sol"])
            batch_sol = next(sol_iter)

        x_sol = batch_sol["x"].to(device)
        soluble = batch_sol["soluble"].to(device)

        outputs = model(x_sol, predict_solubility=True)
        loss_sol, _ = multitask_loss(outputs, soluble_true=soluble, config=config)
        batch_loss += loss_sol
        n_tasks += 1

        # Backward
        batch_loss = batch_loss / n_tasks
        optimizer.zero_grad()
        batch_loss.backward()

        if config.training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )

        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / max_batches


def validate_quick(
    model: nn.Module,
    val_loaders: dict,
    config: Config,
    device: torch.device,
) -> Dict[str, float]:
    """Quick validation (minimal computation)."""
    model.eval()

    # Exp chi validation
    exp_preds, exp_targets = [], []
    with torch.no_grad():
        for batch in val_loaders["exp"]:
            x = batch["x"].to(device)
            chi_true = batch["chi_exp"].to(device)
            temp = batch["temperature"].to(device)

            outputs = model(x, temperature=temp)
            chi_pred = outputs["chi"]

            exp_preds.append(chi_pred.cpu())
            exp_targets.append(chi_true.cpu())

    exp_preds = torch.cat(exp_preds)
    exp_targets = torch.cat(exp_targets)
    exp_metrics = compute_metrics_regression(exp_preds, exp_targets)

    # Solubility validation
    sol_preds, sol_targets = [], []
    with torch.no_grad():
        for batch in val_loaders["sol"]:
            x = batch["x"].to(device)
            soluble = batch["soluble"].to(device)

            outputs = model(x, predict_solubility=True)
            p_soluble = outputs["p_soluble"]

            sol_preds.append(p_soluble.cpu())
            sol_targets.append(soluble.cpu())

    sol_preds = torch.cat(sol_preds)
    sol_targets = torch.cat(sol_targets)

    # Optimize threshold
    threshold_to_use = config.solubility.decision_threshold
    if hasattr(config.solubility, 'optimize_threshold') and config.solubility.optimize_threshold:
        optimal_threshold, _ = find_optimal_threshold(
            y_true=sol_targets.numpy(),
            y_prob=sol_preds.numpy(),
            metric=getattr(config.solubility, 'threshold_metric', 'f1'),
        )
        threshold_to_use = optimal_threshold

    sol_metrics = compute_metrics_classification(sol_preds, sol_targets, threshold=threshold_to_use)

    return {
        "exp_mae": exp_metrics["mae"],
        "exp_r2": exp_metrics["r2"],
        "sol_f1": sol_metrics["f1"],
        "sol_recall": sol_metrics["recall"],
        "sol_precision": sol_metrics["precision"],
        "sol_roc_auc": sol_metrics["roc_auc"],
    }


def train_multitask_quick(
    config: Config,
    pretrained_path: Path,
    cv_fold: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Quick multitask training for hyperparameter search.

    Args:
        config: Configuration object
        pretrained_path: Path to pretrained Stage 1 model
        cv_fold: Optional CV fold index (for cross-validation)
        verbose: If True, show progress bars

    Returns:
        Dictionary of test metrics
    """
    # Set seed
    set_seed(config.seed + (cv_fold if cv_fold is not None else 0))

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    # Load data
    featurizer = PolymerFeaturizer(config)
    features, smiles_to_idx = featurizer.featurize_all_datasets()
    feature_dim = features.shape[1]

    # Exp chi data
    exp_chi_df = featurizer.load_exp_chi_data()
    train_exp_df, val_exp_df, test_exp_df = create_exp_chi_splits(
        exp_chi_df, config, fold=cv_fold
    )

    train_exp = ExpChiDataset(train_exp_df, features, smiles_to_idx)
    val_exp = ExpChiDataset(val_exp_df, features, smiles_to_idx)
    test_exp = ExpChiDataset(test_exp_df, features, smiles_to_idx)

    # Solubility data
    sol_df = featurizer.load_solubility_data()
    train_sol_df, val_sol_df, test_sol_df = create_solubility_splits(sol_df, config)

    train_sol = SolubilityDataset(train_sol_df, features, smiles_to_idx)
    val_sol = SolubilityDataset(val_sol_df, features, smiles_to_idx)
    test_sol = SolubilityDataset(test_sol_df, features, smiles_to_idx)

    # Create dataloaders
    train_loaders = {
        "exp": DataLoader(train_exp, batch_size=config.training.batch_size_exp,
                         shuffle=True, collate_fn=collate_exp_chi),
        "sol": DataLoader(train_sol, batch_size=config.training.batch_size_sol,
                         shuffle=True, collate_fn=collate_solubility),
    }

    val_loaders = {
        "exp": DataLoader(val_exp, batch_size=config.training.batch_size_exp,
                         shuffle=False, collate_fn=collate_exp_chi),
        "sol": DataLoader(val_sol, batch_size=config.training.batch_size_sol,
                         shuffle=False, collate_fn=collate_solubility),
    }

    test_loaders = {
        "exp": DataLoader(test_exp, batch_size=config.training.batch_size_exp,
                         shuffle=False, collate_fn=collate_exp_chi),
        "sol": DataLoader(test_sol, batch_size=config.training.batch_size_sol,
                         shuffle=False, collate_fn=collate_solubility),
    }

    # Build model
    model = MultiTaskChiSolubilityModel(feature_dim, config)
    model.load_encoder_and_chi_head(pretrained_path)

    if config.training.get("freeze_encoder_stage2", False):
        model.freeze_encoder()

    model = model.to(device)

    # Setup optimizer
    lr = config.training.lr_finetune
    weight_decay = config.training.weight_decay

    if config.training.get("use_discriminative_lr", False):
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': config.training.get("lr_encoder", 1e-5)},
            {'params': model.chi_head.parameters(), 'lr': lr},
            {'params': model.solubility_head.parameters(), 'lr': lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Quick training loop (20 epochs, early stopping at 10)
    num_epochs = config.training.get("num_epochs_quick", 20)
    patience = config.training.get("quick_patience", 10)
    best_val_loss = float('inf')
    patience_counter = 0

    iterator = tqdm(range(num_epochs), desc="Quick training") if verbose else range(num_epochs)

    for epoch in iterator:
        train_loss = train_epoch_quick(model, train_loaders, optimizer, config, device)
        val_metrics = validate_quick(model, val_loaders, config, device)

        val_loss = val_metrics["exp_mae"] + (1.0 - val_metrics["sol_f1"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    model = model.to(device)
    test_metrics = validate_quick(model, test_loaders, config, device)

    return test_metrics
