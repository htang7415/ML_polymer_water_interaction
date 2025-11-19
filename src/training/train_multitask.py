"""
Stage 2: Multi-task fine-tuning script.

Fine-tunes pretrained encoder + chi head and adds solubility head.
Trains on DFT chi + experimental chi + solubility simultaneously.

Usage:
    python -m src.training.train_multitask --config configs/config.yaml \
        --pretrained results/dft_pretrain_*/checkpoints/best_model.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.datasets import (
    DFTChiDataset,
    ExpChiDataset,
    SolubilityDataset,
    collate_dft_chi,
    collate_exp_chi,
    collate_solubility,
)
from src.data.featurization import PolymerFeaturizer
from src.data.splits import create_dft_splits, create_solubility_splits, create_exp_chi_splits
from src.evaluation.analysis import (
    save_detailed_predictions,
    save_classification_predictions,
    analyze_chi_solubility_relationship,
    analyze_A_sign_distribution,
)
from src.evaluation.plots import (
    plot_training_history,
    plot_parity_with_temperature,
    plot_parity_with_uncertainty,
    plot_residual_vs_temperature,
    plot_calibration,
    plot_confusion_matrix,
    plot_chi_rt_vs_solubility,
    plot_error_vs_uncertainty,
    plot_uncertainty_calibration,
)
from src.evaluation.uncertainty import mc_predict_batch, enable_mc_dropout
from src.evaluation.metrics import compute_confusion_matrix
from src.models.multitask_model import MultiTaskChiSolubilityModel
from src.training.losses import (
    multitask_loss,
    compute_metrics_regression,
    compute_metrics_classification,
)
from src.utils.config import Config, load_config, save_config
from src.utils.logging_utils import create_run_directory, get_logger, setup_logging, MetricsLogger
from src.utils.seed_utils import set_seed, worker_init_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Multi-task fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained DFT model checkpoint from Stage 1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, uses config value.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. If not specified, uses timestamped directory.",
    )
    return parser.parse_args()


def load_and_prepare_data(
    config: Config,
    logger,
) -> Tuple[dict, dict, dict, int]:
    """
    Load all datasets (DFT, exp chi, solubility), featurize, and create dataloaders.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        train_loaders: Dict of training dataloaders
        val_loaders: Dict of validation dataloaders
        test_loaders: Dict of test dataloaders
        feature_dim: Feature dimensionality
    """
    logger.info("=" * 80)
    logger.info("Loading and preparing multi-task data")
    logger.info("=" * 80)

    # Collect all unique SMILES from all datasets
    all_smiles = set()

    # Load DFT data
    dft_csv_path = Path(config.paths.dft_chi_csv)
    logger.info(f"Loading DFT data from: {dft_csv_path}")
    df_dft = pd.read_csv(dft_csv_path)
    all_smiles.update(df_dft["SMILES"].unique())
    logger.info(f"DFT: {len(df_dft)} measurements, {df_dft['SMILES'].nunique()} unique polymers")

    # Load experimental chi data
    exp_csv_path = Path(config.paths.exp_chi_csv)
    logger.info(f"Loading experimental chi data from: {exp_csv_path}")
    df_exp = pd.read_csv(exp_csv_path)
    all_smiles.update(df_exp["SMILES"].unique())
    logger.info(f"Exp chi: {len(df_exp)} measurements, {df_exp['SMILES'].nunique()} unique polymers")

    # Load solubility data
    sol_csv_path = Path(config.paths.solubility_csv)
    logger.info(f"Loading solubility data from: {sol_csv_path}")
    df_sol = pd.read_csv(sol_csv_path)
    # Rename water_soluble to soluble for consistency
    if "water_soluble" in df_sol.columns:
        df_sol = df_sol.rename(columns={"water_soluble": "soluble"})
    all_smiles.update(df_sol["SMILES"].unique())
    logger.info(f"Solubility: {len(df_sol)} samples, {df_sol['SMILES'].nunique()} unique polymers")

    # Featurize all unique SMILES
    all_smiles = list(all_smiles)
    logger.info(f"\nFeaturizing {len(all_smiles)} total unique polymers...")
    featurizer = PolymerFeaturizer(config)
    features, smiles_to_idx = featurizer.featurize(all_smiles)
    feature_dim = featurizer.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")

    # Create splits for each dataset
    logger.info("\nCreating data splits...")

    # DFT splits
    train_dft_df, val_dft_df, test_dft_df = create_dft_splits(df_dft, config)

    # Exp chi splits (SMILES-grouped to prevent data leakage)
    # NOTE: Using create_exp_chi_splits ensures all measurements of same polymer
    # stay in same split (train/val/test), preventing data leakage
    train_exp_df, val_exp_df, test_exp_df = create_exp_chi_splits(df_exp, config)

    # Solubility splits (stratified)
    train_sol_df, val_sol_df, test_sol_df = create_solubility_splits(df_sol, config)

    # Create datasets
    logger.info("\nCreating datasets...")

    # DFT datasets
    train_dft = DFTChiDataset(train_dft_df, features, smiles_to_idx)
    val_dft = DFTChiDataset(val_dft_df, features, smiles_to_idx)
    test_dft = DFTChiDataset(test_dft_df, features, smiles_to_idx)

    # Exp chi datasets
    train_exp = ExpChiDataset(train_exp_df, features, smiles_to_idx)
    val_exp = ExpChiDataset(val_exp_df, features, smiles_to_idx)
    test_exp = ExpChiDataset(test_exp_df, features, smiles_to_idx)

    # Solubility datasets
    train_sol = SolubilityDataset(train_sol_df, features, smiles_to_idx)
    val_sol = SolubilityDataset(val_sol_df, features, smiles_to_idx)
    test_sol = SolubilityDataset(test_sol_df, features, smiles_to_idx)

    # Create dataloaders
    logger.info("\nCreating dataloaders...")

    num_workers = config.training.num_workers
    pin_memory = config.training.pin_memory

    # Train loaders
    train_dft_loader = DataLoader(
        train_dft,
        batch_size=config.training.batch_size_dft,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
        worker_init_fn=lambda wid: worker_init_fn(wid, config.seed),
    )

    train_exp_loader = DataLoader(
        train_exp,
        batch_size=config.training.batch_size_exp,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_exp_chi,
        worker_init_fn=lambda wid: worker_init_fn(wid, config.seed),
    )

    train_sol_loader = DataLoader(
        train_sol,
        batch_size=config.training.batch_size_sol,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_solubility,
        worker_init_fn=lambda wid: worker_init_fn(wid, config.seed),
    )

    # Val loaders
    val_dft_loader = DataLoader(
        val_dft,
        batch_size=config.training.batch_size_dft,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
    )

    val_exp_loader = DataLoader(
        val_exp,
        batch_size=config.training.batch_size_exp,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_exp_chi,
    )

    val_sol_loader = DataLoader(
        val_sol,
        batch_size=config.training.batch_size_sol,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_solubility,
    )

    # Test loaders
    test_dft_loader = DataLoader(
        test_dft,
        batch_size=config.training.batch_size_dft,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
    )

    test_exp_loader = DataLoader(
        test_exp,
        batch_size=config.training.batch_size_exp,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_exp_chi,
    )

    test_sol_loader = DataLoader(
        test_sol,
        batch_size=config.training.batch_size_sol,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_solubility,
    )

    train_loaders = {
        "dft": train_dft_loader,
        "exp": train_exp_loader,
        "sol": train_sol_loader,
    }

    val_loaders = {
        "dft": val_dft_loader,
        "exp": val_exp_loader,
        "sol": val_sol_loader,
    }

    test_loaders = {
        "dft": test_dft_loader,
        "exp": test_exp_loader,
        "sol": test_sol_loader,
    }

    logger.info(f"Train batches - DFT: {len(train_dft_loader)}, "
                f"Exp: {len(train_exp_loader)}, Sol: {len(train_sol_loader)}")
    logger.info(f"Val batches - DFT: {len(val_dft_loader)}, "
                f"Exp: {len(val_exp_loader)}, Sol: {len(val_sol_loader)}")
    logger.info(f"Test batches - DFT: {len(test_dft_loader)}, "
                f"Exp: {len(test_exp_loader)}, Sol: {len(test_sol_loader)}")

    return train_loaders, val_loaders, test_loaders, feature_dim


def train_epoch(
    model: nn.Module,
    train_loaders: dict,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    logger,
) -> Dict[str, float]:
    """
    Train for one epoch on all tasks.

    Args:
        model: Model to train
        train_loaders: Dictionary of training dataloaders
        optimizer: Optimizer
        config: Configuration
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary of epoch metrics
    """
    model.train()

    total_loss = 0.0
    task_losses = {"dft": 0.0, "exp": 0.0, "sol": 0.0}
    task_counts = {"dft": 0, "exp": 0, "sol": 0}

    # Cycle through all dataloaders
    # Simple approach: iterate through longest dataloader, cycling shorter ones
    dft_iter = iter(train_loaders["dft"])
    exp_iter = iter(train_loaders["exp"])
    sol_iter = iter(train_loaders["sol"])

    max_batches = max(len(train_loaders["dft"]), len(train_loaders["exp"]), len(train_loaders["sol"]))

    pbar = tqdm(range(max_batches), desc="Training", leave=False)

    for batch_idx in pbar:
        batch_loss = 0.0
        n_tasks = 0

        # DFT batch
        try:
            batch_dft = next(dft_iter)
        except StopIteration:
            dft_iter = iter(train_loaders["dft"])
            batch_dft = next(dft_iter)

        x_dft = batch_dft["x"].to(device)
        chi_dft = batch_dft["chi_dft"].to(device)
        temp_dft = batch_dft["temperature"].to(device)  # Use actual temperatures

        outputs = model(x_dft, temperature=temp_dft)
        loss_dft, _ = multitask_loss(outputs, chi_dft_true=chi_dft, config=config)
        batch_loss += loss_dft
        task_losses["dft"] += loss_dft.item()
        task_counts["dft"] += 1
        n_tasks += 1

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
            outputs,
            chi_exp_true=chi_exp,
            temperature_exp=temp_exp,
            config=config
        )
        batch_loss += loss_exp
        task_losses["exp"] += loss_exp.item()
        task_counts["exp"] += 1
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
        task_losses["sol"] += loss_sol.item()
        task_counts["sol"] += 1
        n_tasks += 1

        # Average batch loss across tasks
        batch_loss = batch_loss / n_tasks

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping
        if config.training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )

        optimizer.step()

        total_loss += batch_loss.item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / max_batches
    avg_task_losses = {
        task: loss / count if count > 0 else 0.0
        for task, (loss, count) in [(t, (task_losses[t], task_counts[t])) for t in task_losses]
    }

    metrics = {
        "loss": avg_loss,
        "loss_dft": avg_task_losses["dft"],
        "loss_exp": avg_task_losses["exp"],
        "loss_sol": avg_task_losses["sol"],
    }

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loaders: dict,
    config: Config,
    device: torch.device,
) -> Tuple[Dict[str, float], dict]:
    """
    Validate model on all tasks.

    Args:
        model: Model to validate
        val_loaders: Dictionary of validation dataloaders
        config: Configuration
        device: Device to use

    Returns:
        metrics: Dictionary of validation metrics
        predictions: Dictionary of predictions for each task
    """
    model.eval()

    # DFT chi validation
    dft_preds, dft_targets = [], []
    for batch in val_loaders["dft"]:
        x = batch["x"].to(device)
        chi_true = batch["chi_dft"].to(device)
        temp_dft = batch["temperature"].to(device)  # Use actual temperatures

        outputs = model(x, temperature=temp_dft)
        chi_pred = outputs["chi"]

        dft_preds.append(chi_pred.cpu())
        dft_targets.append(chi_true.cpu())

    dft_preds = torch.cat(dft_preds)
    dft_targets = torch.cat(dft_targets)
    dft_metrics = compute_metrics_regression(dft_preds, dft_targets)

    # Exp chi validation
    exp_preds, exp_targets = [], []
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
    for batch in val_loaders["sol"]:
        x = batch["x"].to(device)
        soluble = batch["soluble"].to(device)

        outputs = model(x, predict_solubility=True)
        p_soluble = outputs["p_soluble"]

        sol_preds.append(p_soluble.cpu())
        sol_targets.append(soluble.cpu())

    sol_preds = torch.cat(sol_preds)
    sol_targets = torch.cat(sol_targets)
    sol_metrics = compute_metrics_classification(
        sol_preds, sol_targets, threshold=config.solubility.decision_threshold
    )

    # Combine metrics
    metrics = {
        "dft_mae": dft_metrics["mae"],
        "dft_rmse": dft_metrics["rmse"],
        "dft_r2": dft_metrics["r2"],
        "exp_mae": exp_metrics["mae"],
        "exp_rmse": exp_metrics["rmse"],
        "exp_r2": exp_metrics["r2"],
        "sol_accuracy": sol_metrics["accuracy"],
        "sol_precision": sol_metrics["precision"],
        "sol_recall": sol_metrics["recall"],
        "sol_f1": sol_metrics["f1"],
        "sol_roc_auc": sol_metrics["roc_auc"],
    }

    predictions = {
        "dft": (dft_preds.numpy(), dft_targets.numpy()),
        "exp": (exp_preds.numpy(), exp_targets.numpy()),
        "sol": (sol_preds.numpy(), sol_targets.numpy()),
    }

    return metrics, predictions


def plot_multitask_results(
    predictions: dict,
    save_dir: Path,
    config: Config,
    title_prefix: str = "Validation",
    dpi: int = 300,
    uncertainties: dict = None,
):
    """
    Create plots for all tasks with optional uncertainty visualization.

    Args:
        predictions: Dictionary of predictions for each task
        save_dir: Directory to save figures
        config: Configuration object
        title_prefix: Prefix for plot titles
        dpi: Figure DPI
        uncertainties: Optional dictionary of uncertainties {"dft_std": array, "exp_std": array}
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # DFT chi parity plot with uncertainty
    dft_preds, dft_targets = predictions["dft"]
    if uncertainties is not None and "dft_std" in uncertainties:
        # Use uncertainty-aware parity plot
        try:
            plot_parity_with_uncertainty(
                y_true=dft_targets,
                y_pred_mean=dft_preds,
                y_pred_std=uncertainties["dft_std"],
                save_path=save_dir / "parity_dft_chi",
                config=config,
                title=f"{title_prefix} - DFT Chi with Uncertainty",
                error_bar_type="color",
            )
        except Exception as e:
            # Fallback to basic parity plot
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(dft_targets, dft_preds, alpha=0.5, s=20, edgecolors='none')
            min_val = min(dft_targets.min(), dft_preds.min())
            max_val = max(dft_targets.max(), dft_preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
            ax.set_xlabel("True Chi (DFT)", fontsize=12)
            ax.set_ylabel("Predicted Chi", fontsize=12)
            ax.set_title(f"{title_prefix} - DFT Chi", fontsize=14)
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(save_dir / "parity_dft_chi.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        # Basic parity plot without uncertainty
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(dft_targets, dft_preds, alpha=0.5, s=20, edgecolors='none')
        min_val = min(dft_targets.min(), dft_preds.min())
        max_val = max(dft_targets.max(), dft_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
        ax.set_xlabel("True Chi (DFT)", fontsize=12)
        ax.set_ylabel("Predicted Chi", fontsize=12)
        ax.set_title(f"{title_prefix} - DFT Chi", fontsize=14)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(save_dir / "parity_dft_chi.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    # Exp chi parity plot with uncertainty
    exp_preds, exp_targets = predictions["exp"]
    if uncertainties is not None and "exp_std" in uncertainties:
        # Use uncertainty-aware parity plot
        try:
            plot_parity_with_uncertainty(
                y_true=exp_targets,
                y_pred_mean=exp_preds,
                y_pred_std=uncertainties["exp_std"],
                save_path=save_dir / "parity_exp_chi",
                config=config,
                title=f"{title_prefix} - Experimental Chi with Uncertainty",
                error_bar_type="bars",  # Use error bars for smaller dataset
            )
        except Exception as e:
            # Fallback to basic parity plot
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(exp_targets, exp_preds, alpha=0.5, s=20, edgecolors='none')
            min_val = min(exp_targets.min(), exp_preds.min())
            max_val = max(exp_targets.max(), exp_preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
            ax.set_xlabel("True Chi (Experimental)", fontsize=12)
            ax.set_ylabel("Predicted Chi", fontsize=12)
            ax.set_title(f"{title_prefix} - Experimental Chi", fontsize=14)
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(save_dir / "parity_exp_chi.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        # Basic parity plot without uncertainty
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(exp_targets, exp_preds, alpha=0.5, s=20, edgecolors='none')
        min_val = min(exp_targets.min(), exp_preds.min())
        max_val = max(exp_targets.max(), exp_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
        ax.set_xlabel("True Chi (Experimental)", fontsize=12)
        ax.set_ylabel("Predicted Chi", fontsize=12)
        ax.set_title(f"{title_prefix} - Experimental Chi", fontsize=14)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(save_dir / "parity_exp_chi.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    # Solubility ROC curve (if sklearn available)
    try:
        from sklearn.metrics import roc_curve, auc

        sol_preds, sol_targets = predictions["sol"]
        fpr, tpr, _ = roc_curve(sol_targets, sol_preds)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"{title_prefix} - Solubility ROC", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "roc_solubility.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    except ImportError:
        pass


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override device if specified
    if args.device is not None:
        config.training.device = args.device

    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    # Set random seed
    set_seed(config.seed, deterministic=False)

    # Create run directory
    if args.output_dir is not None:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(
            Path(config.paths.results_dir),
            "multitask",
        )

    # Setup logging
    logger = setup_logging(
        log_dir=run_dir,
        log_file="train.log",
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
    )

    logger.info("=" * 80)
    logger.info("Stage 2: Multi-task Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.seed}")
    logger.info(f"Pretrained checkpoint: {args.pretrained}")

    # Save config
    save_config(config, run_dir / "config.yaml")

    # Load and prepare data
    train_loaders, val_loaders, test_loaders, feature_dim = load_and_prepare_data(config, logger)

    # Build model
    logger.info("=" * 80)
    logger.info("Building model")
    logger.info("=" * 80)

    model = MultiTaskChiSolubilityModel(feature_dim, config)

    # Load pretrained weights
    logger.info(f"Loading pretrained weights from {args.pretrained}")
    model.load_encoder_and_chi_head(args.pretrained)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {n_params:,} trainable parameters")

    # Setup optimizer
    optimizer_name = config.training.optimizer.lower()
    lr = config.training.lr_finetune
    weight_decay = config.training.weight_decay

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")

    # Setup scheduler
    scheduler = None
    if config.training.use_scheduler:
        scheduler_type = config.training.scheduler_type

        if scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience,
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.scheduler_step_size,
                gamma=config.training.scheduler_factor,
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs_finetune,
            )

        if scheduler is not None:
            logger.info(f"Scheduler: {scheduler_type}")

    # Setup metrics logger
    metrics_logger = MetricsLogger(run_dir / "metrics.csv")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)

    num_epochs = config.training.num_epochs_finetune
    best_val_metric = float('inf')  # Using exp_mae as primary metric
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = config.training.early_stopping_patience

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 80)

        # Train
        train_metrics = train_epoch(model, train_loaders, optimizer, config, device, logger)

        logger.info(
            f"Train - Total Loss: {train_metrics['loss']:.4f}, "
            f"DFT: {train_metrics['loss_dft']:.4f}, "
            f"Exp: {train_metrics['loss_exp']:.4f}, "
            f"Sol: {train_metrics['loss_sol']:.4f}"
        )

        # Validate
        val_metrics, val_predictions = validate(model, val_loaders, config, device)

        logger.info(
            f"Val DFT   - MAE: {val_metrics['dft_mae']:.4f}, "
            f"RMSE: {val_metrics['dft_rmse']:.4f}, "
            f"R²: {val_metrics['dft_r2']:.4f}"
        )
        logger.info(
            f"Val Exp   - MAE: {val_metrics['exp_mae']:.4f}, "
            f"RMSE: {val_metrics['exp_rmse']:.4f}, "
            f"R²: {val_metrics['exp_r2']:.4f}"
        )
        logger.info(
            f"Val Sol   - Acc: {val_metrics['sol_accuracy']:.4f}, "
            f"F1: {val_metrics['sol_f1']:.4f}, "
            f"ROC-AUC: {val_metrics['sol_roc_auc']:.4f}"
        )

        # Log metrics
        metrics_logger.log({
            "train_loss": train_metrics["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["exp_mae"])
            else:
                scheduler.step()

        # Check for best model (using exp_mae as primary metric)
        if val_metrics["exp_mae"] < best_val_metric:
            best_val_metric = val_metrics["exp_mae"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config.to_dict(),
            }, checkpoint_path)

            logger.info(f"Saved best model to {checkpoint_path}")

            # Plot results (without uncertainty during training for speed)
            plot_multitask_results(
                val_predictions,
                run_dir / "figures" / "best",
                config,
                title_prefix=f"Validation (Epoch {epoch})",
                dpi=config.plotting.dpi,
            )

        else:
            patience_counter += 1

        # Early stopping
        if config.training.early_stopping and patience_counter >= early_stopping_patience:
            logger.info(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(best epoch: {best_epoch}, best exp MAE: {best_val_metric:.4f})"
            )
            break

    # Training complete
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best experimental chi MAE: {best_val_metric:.4f}")

    # Save final model (last epoch)
    final_checkpoint_path = run_dir / "checkpoints" / "final_model.pt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "config": config.to_dict(),
    }, final_checkpoint_path)

    logger.info(f"Saved final model (epoch {epoch}) to {final_checkpoint_path}")

    # Load best model and evaluate on test set
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(run_dir / "checkpoints" / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get basic metrics for logging
    test_metrics, test_predictions = validate(model, test_loaders, config, device)

    logger.info(
        f"Test DFT  - MAE: {test_metrics['dft_mae']:.4f}, "
        f"RMSE: {test_metrics['dft_rmse']:.4f}, "
        f"R²: {test_metrics['dft_r2']:.4f}"
    )
    logger.info(
        f"Test Exp  - MAE: {test_metrics['exp_mae']:.4f}, "
        f"RMSE: {test_metrics['exp_rmse']:.4f}, "
        f"R²: {test_metrics['exp_r2']:.4f}"
    )
    logger.info(
        f"Test Sol  - Acc: {test_metrics['sol_accuracy']:.4f}, "
        f"F1: {test_metrics['sol_f1']:.4f}, "
        f"ROC-AUC: {test_metrics['sol_roc_auc']:.4f}"
    )

    # Get MC dropout uncertainty estimates for chi predictions
    logger.info("Computing test predictions with MC dropout uncertainty...")
    enable_mc_dropout(model)

    # DFT chi uncertainty
    test_uncertainties = {}
    if test_loaders["dft"] is not None:
        dft_mc_results = mc_predict_batch(
            model=model,
            dataloader=test_loaders["dft"],
            T_ref=config.model.T_ref_K,
            n_samples=config.uncertainty.mc_dropout_samples,
            device=device,
            predict_solubility=False,
        )
        test_uncertainties["dft_std"] = dft_mc_results["chi_std"]
        # Update predictions with MC dropout means
        test_predictions["dft"] = (dft_mc_results["chi_mean"], dft_mc_results["chi_true"])

    # Experimental chi uncertainty
    if test_loaders["exp"] is not None:
        exp_mc_results = mc_predict_batch(
            model=model,
            dataloader=test_loaders["exp"],
            T_ref=config.model.T_ref_K,
            n_samples=config.uncertainty.mc_dropout_samples,
            device=device,
            predict_solubility=False,
        )
        test_uncertainties["exp_std"] = exp_mc_results["chi_std"]
        # Update predictions with MC dropout means
        test_predictions["exp"] = (exp_mc_results["chi_mean"], exp_mc_results["chi_true"])

    # Plot test results with uncertainty
    plot_multitask_results(
        test_predictions,
        run_dir / "figures" / "test",
        config,
        title_prefix="Test Set",
        dpi=config.plotting.dpi,
        uncertainties=test_uncertainties,
    )

    # Plot uncertainty calibration for experimental chi (most important)
    if "exp_std" in test_uncertainties:
        exp_preds, exp_targets = test_predictions["exp"]
        try:
            plot_error_vs_uncertainty(
                y_true=exp_targets,
                y_pred_mean=exp_preds,
                y_pred_std=test_uncertainties["exp_std"],
                save_path=run_dir / "figures" / "test" / "error_vs_uncertainty_exp",
                config=config,
                title="Experimental Chi: Error vs Uncertainty",
            )
            logger.info("Saved error vs uncertainty plot for experimental chi")
        except Exception as e:
            logger.error(f"Failed to generate error vs uncertainty plot: {e}")

        try:
            plot_uncertainty_calibration(
                y_true=exp_targets,
                y_pred_mean=exp_preds,
                y_pred_std=test_uncertainties["exp_std"],
                save_path=run_dir / "figures" / "test" / "uncertainty_calibration_exp",
                config=config,
                title="Experimental Chi: Uncertainty Calibration",
            )
            logger.info("Saved uncertainty calibration plot for experimental chi")
        except Exception as e:
            logger.error(f"Failed to generate uncertainty calibration plot: {e}")

    # Save detailed predictions to CSV with uncertainty
    # DFT chi predictions
    dft_preds, dft_targets = test_predictions["dft"]
    save_detailed_predictions(
        predictions=dft_preds,
        targets=dft_targets,
        save_path=run_dir / "test_predictions_dft_chi.csv",
        uncertainties=test_uncertainties.get("dft_std"),
    )
    logger.info(f"DFT chi predictions saved to test_predictions_dft_chi.csv")

    # Experimental chi predictions
    exp_preds, exp_targets = test_predictions["exp"]
    save_detailed_predictions(
        predictions=exp_preds,
        targets=exp_targets,
        save_path=run_dir / "test_predictions_exp_chi.csv",
        uncertainties=test_uncertainties.get("exp_std"),
    )
    logger.info(f"Experimental chi predictions saved to test_predictions_exp_chi.csv")

    # Solubility predictions
    sol_preds, sol_targets = test_predictions["sol"]
    save_classification_predictions(
        predictions_prob=sol_preds,
        targets=sol_targets,
        save_path=run_dir / "test_predictions_solubility.csv",
        threshold=config.solubility.decision_threshold,
    )
    logger.info(f"Solubility predictions saved to test_predictions_solubility.csv")

    # Generate training history plots
    logger.info("\nGenerating training history plots...")
    try:
        plot_training_history(
            metrics_csv_path=run_dir / "metrics.csv",
            save_path=run_dir / "figures" / "training_curves",
            config=config,
            title="Multi-task Training History",
            is_multitask=True,
        )
        logger.info(f"Saved training curves to {run_dir / 'figures' / 'training_curves.png'}")
    except Exception as e:
        logger.error(f"Failed to generate training history plot: {e}")

    # Generate additional diagnostic plots
    logger.info("Generating diagnostic plots...")

    # Solubility calibration plot
    try:
        plot_calibration(
            y_true=sol_targets,
            y_prob=sol_preds,
            save_path=run_dir / "figures" / "test" / "calibration_solubility",
            config=config,
            title="Solubility Prediction Calibration (Test Set)",
        )
        logger.info(f"Saved calibration plot")
    except Exception as e:
        logger.error(f"Failed to generate calibration plot: {e}")

    # Confusion matrix for solubility
    try:
        threshold = config.solubility.decision_threshold
        cm, counts = compute_confusion_matrix(sol_targets, sol_preds, threshold=threshold)
        plot_confusion_matrix(
            cm=counts,
            save_path=run_dir / "figures" / "test" / "confusion_matrix_solubility",
            config=config,
            title=f"Solubility Confusion Matrix (Test Set, threshold={threshold})",
        )
        logger.info(f"Saved confusion matrix")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")

    # Get detailed predictions with chi_RT for chi-solubility relationship analysis
    logger.info("\nRunning chi-solubility relationship analysis...")
    try:
        # Compute chi_RT for test solubility data
        model.eval()
        with torch.no_grad():
            chi_rt_list = []
            sol_labels_list = []
            for batch in test_loaders["sol"]:
                x = batch["x"].to(device)
                temp_rt = torch.full((x.size(0),), config.solubility.T_ref).to(device)
                outputs = model(x, temperature=temp_rt)
                chi_rt = outputs["chi"]
                chi_rt_list.append(chi_rt.cpu().numpy())
                sol_labels_list.append(batch["solubility"].cpu().numpy())

            chi_rt_array = np.concatenate(chi_rt_list)
            sol_labels_array = np.concatenate(sol_labels_list)

        # Plot chi_RT distribution by solubility class
        plot_chi_rt_vs_solubility(
            chi_rt=chi_rt_array,
            solubility_labels=sol_labels_array,
            save_path=run_dir / "figures" / "analysis" / "chi_rt_vs_solubility",
            config=config,
            plot_type="box",
            title=r"$\chi_{RT}$ Distribution by Solubility Class",
        )
        logger.info(f"Saved chi_RT vs solubility plot")

        # Run chi-solubility relationship analysis
        analysis_results = analyze_chi_solubility_relationship(
            chi_rt=chi_rt_array,
            solubility_labels=sol_labels_array,
            T_ref=config.solubility.T_ref,
        )

        # Save analysis results
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / "chi_solubility_analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2)
        logger.info(f"Saved chi-solubility analysis to {analysis_dir / 'chi_solubility_analysis.json'}")

    except Exception as e:
        logger.error(f"Failed to run chi-solubility analysis: {e}")

    # Analyze A-parameter distribution
    logger.info("\nAnalyzing A-parameter distribution (UCST/LCST behavior)...")
    try:
        # Get A parameters for experimental chi test set
        with torch.no_grad():
            A_params_list = []
            for batch in test_loaders["exp"]:
                x = batch["x"].to(device)
                temp = batch["temperature"].to(device)
                outputs = model(x, temperature=temp)
                A_params_list.append(outputs["A"].cpu().numpy())

            A_params_array = np.concatenate(A_params_list)

        # Analyze A-parameter distribution
        A_analysis = analyze_A_sign_distribution(A_params_array)

        # Save A-parameter analysis
        with open(analysis_dir / "A_parameter_analysis.json", "w") as f:
            json.dump(A_analysis, f, indent=2)
        logger.info(f"Saved A-parameter analysis")
        logger.info(f"  UCST (A>0): {A_analysis['n_positive']} ({A_analysis['percent_positive']:.1f}%)")
        logger.info(f"  LCST (A<0): {A_analysis['n_negative']} ({A_analysis['percent_negative']:.1f}%)")

    except Exception as e:
        logger.error(f"Failed to analyze A-parameter distribution: {e}")

    # Save final summary
    summary = {
        "best_epoch": best_epoch,
        "best_exp_mae": best_val_metric,
        "test_metrics": test_metrics,
        "run_dir": str(run_dir),
        "best_model_checkpoint": str(run_dir / "checkpoints" / "best_model.pt"),
        "final_model_checkpoint": str(run_dir / "checkpoints" / "final_model.pt"),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to {run_dir / 'summary.json'}")
    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
