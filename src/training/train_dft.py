"""
Stage 1: DFT chi pretraining script.

Trains encoder + chi head on large DFT dataset to learn polymer representations
and chi(T) = A/T + B prediction.

Usage:
    python -m src.training.train_dft --config configs/config.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.datasets import DFTChiDataset, collate_dft_chi
from src.data.featurization import PolymerFeaturizer
from src.data.splits import create_dft_splits
from src.evaluation.analysis import save_detailed_predictions
from src.evaluation.plots import (
    plot_training_history,
    plot_parity_with_temperature,
    plot_parity_with_uncertainty,
    plot_residual_vs_temperature,
    plot_error_vs_uncertainty,
    plot_uncertainty_calibration,
)
from src.evaluation.uncertainty import mc_predict_batch, enable_mc_dropout
from src.models.multitask_model import MultiTaskChiSolubilityModel
from src.training.losses import chi_dft_loss, compute_metrics_regression
from src.utils.config import Config, load_config, save_config
from src.utils.logging_utils import create_run_directory, get_logger, setup_logging, MetricsLogger
from src.utils.seed_utils import set_seed, worker_init_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 1: DFT chi pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file",
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
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load DFT data, featurize, and create train/val/test dataloaders.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        feature_dim: Feature dimensionality
    """
    logger.info("=" * 80)
    logger.info("Loading and preparing DFT chi data")
    logger.info("=" * 80)

    # Load DFT data
    dft_csv_path = Path(config.paths.dft_chi_csv)
    logger.info(f"Loading DFT data from: {dft_csv_path}")

    df_dft = pd.read_csv(dft_csv_path)
    logger.info(f"Loaded {len(df_dft)} DFT measurements")

    # Get unique SMILES
    unique_smiles = df_dft["SMILES"].unique().tolist()
    logger.info(f"Found {len(unique_smiles)} unique polymers")

    # Featurize
    logger.info("Featurizing polymers...")
    featurizer = PolymerFeaturizer(config)
    features, smiles_to_idx = featurizer.featurize(unique_smiles)
    feature_dim = featurizer.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")

    # Create splits
    logger.info("Creating train/val/test splits...")
    train_df, val_df, test_df = create_dft_splits(df_dft, config)

    # Create datasets
    train_dataset = DFTChiDataset(train_df, features, smiles_to_idx)
    val_dataset = DFTChiDataset(val_df, features, smiles_to_idx)
    test_dataset = DFTChiDataset(test_df, features, smiles_to_idx)

    # Create dataloaders
    batch_size = config.training.batch_size_dft
    num_workers = config.training.num_workers
    pin_memory = config.training.pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
        worker_init_fn=lambda wid: worker_init_fn(wid, config.seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
                f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, feature_dim


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    logger,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training dataloader
        optimizer: Optimizer
        config: Configuration
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary of epoch metrics
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        x = batch["x"].to(device)
        chi_true = batch["chi_dft"].to(device)
        # Use actual temperatures from dataset instead of hard-coded T_ref
        temperature = batch["temperature"].to(device)

        # Forward pass with actual temperatures
        outputs = model(x, temperature=temperature)
        A, B = outputs["A"], outputs["B"]

        # Compute loss with actual temperatures
        loss = chi_dft_loss(A, B, chi_true, temperature=temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        chi_pred = outputs["chi"]
        all_preds.append(chi_pred.detach().cpu())
        all_targets.append(chi_true.detach().cpu())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_metrics_regression(all_preds, all_targets)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    return_detailed: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Validate model.

    Args:
        model: Model to validate
        val_loader: Validation dataloader
        config: Configuration
        device: Device to use
        return_detailed: If True, return additional data (A, B, T, SMILES)

    Returns:
        If return_detailed=False:
            metrics: Dictionary of validation metrics
            predictions: Array of predictions
            targets: Array of true values
        If return_detailed=True:
            metrics: Dictionary of validation metrics
            predictions: Array of predictions
            targets: Array of true values
            temperatures: Array of temperatures
            A_params: Array of A parameters
            B_params: Array of B parameters
            smiles_list: List of SMILES strings
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_temps = []
    all_A = []
    all_B = []
    all_smiles = []

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        # Move batch to device
        x = batch["x"].to(device)
        chi_true = batch["chi_dft"].to(device)
        # Use actual temperatures from dataset
        temperature = batch["temperature"].to(device)

        # Forward pass with actual temperatures
        outputs = model(x, temperature=temperature)
        A, B = outputs["A"], outputs["B"]

        # Compute loss with actual temperatures
        loss = chi_dft_loss(A, B, chi_true, temperature=temperature)

        # Track metrics
        total_loss += loss.item()
        chi_pred = outputs["chi"]
        all_preds.append(chi_pred.cpu())
        all_targets.append(chi_true.cpu())

        if return_detailed:
            all_temps.append(temperature.cpu())
            all_A.append(A.cpu())
            all_B.append(B.cpu())
            all_smiles.extend(batch["smiles"])

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_metrics_regression(all_preds, all_targets)
    metrics["loss"] = avg_loss

    if return_detailed:
        all_temps = torch.cat(all_temps)
        all_A = torch.cat(all_A)
        all_B = torch.cat(all_B)
        return (
            metrics,
            all_preds.numpy(),
            all_targets.numpy(),
            all_temps.numpy(),
            all_A.numpy(),
            all_B.numpy(),
            all_smiles,
        )
    else:
        return metrics, all_preds.numpy(), all_targets.numpy()


def plot_parity(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    title: str = "DFT Chi Parity Plot",
    dpi: int = 300,
):
    """
    Create parity plot (predicted vs true).

    Args:
        predictions: Predicted values
        targets: True values
        save_path: Path to save figure
        title: Plot title
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20, edgecolors='none')

    # Diagonal line (perfect prediction)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel("True Chi", fontsize=12)
    ax.set_ylabel("Predicted Chi", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


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
            "dft_pretrain",
        )

    # Setup logging
    logger = setup_logging(
        log_dir=run_dir,
        log_file="train.log",
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
    )

    logger.info("=" * 80)
    logger.info("Stage 1: DFT Chi Pretraining")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.seed}")

    # Save config
    save_config(config, run_dir / "config.yaml")
    logger.info(f"Saved configuration to {run_dir / 'config.yaml'}")

    # Load and prepare data
    train_loader, val_loader, test_loader, feature_dim = load_and_prepare_data(config, logger)

    # Build model
    logger.info("=" * 80)
    logger.info("Building model")
    logger.info("=" * 80)

    model = MultiTaskChiSolubilityModel(feature_dim, config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {n_params:,} trainable parameters")

    # Setup optimizer
    optimizer_name = config.training.optimizer.lower()
    lr = config.training.lr_pretrain
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
                T_max=config.training.num_epochs_pretrain,
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, disabling scheduler")

        if scheduler is not None:
            logger.info(f"Scheduler: {scheduler_type}")

    # Setup metrics logger
    metrics_logger = MetricsLogger(run_dir / "metrics.csv")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)

    num_epochs = config.training.num_epochs_pretrain
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = config.training.early_stopping_patience

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 80)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, device, logger)

        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"MAE: {train_metrics['mae']:.4f}, "
            f"RMSE: {train_metrics['rmse']:.4f}, "
            f"R²: {train_metrics['r2']:.4f}"
        )

        # Validate
        val_metrics, val_preds, val_targets = validate(model, val_loader, config, device)

        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"RMSE: {val_metrics['rmse']:.4f}, "
            f"R²: {val_metrics['r2']:.4f}"
        )

        # Log metrics
        metrics_logger.log({
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Check for best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "config": config.to_dict(),
            }, checkpoint_path)

            logger.info(f"Saved best model to {checkpoint_path}")

        else:
            patience_counter += 1

        # Early stopping
        if config.training.early_stopping and patience_counter >= early_stopping_patience:
            logger.info(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})"
            )
            break

    # Training complete
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model (last epoch)
    final_checkpoint_path = run_dir / "checkpoints" / "final_model.pt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "config": config.to_dict(),
    }, final_checkpoint_path)

    logger.info(f"Saved final model (epoch {epoch}) to {final_checkpoint_path}")

    # Load best model and evaluate on train and test sets with MC dropout
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(run_dir / "checkpoints" / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # ========================================================================
    # Evaluate on training set
    # ========================================================================
    logger.info("\nEvaluating on training set...")

    # Get basic metrics for logging
    train_metrics, _, _ = validate(
        model, train_loader, config, device, return_detailed=False
    )

    logger.info(
        f"Train - Loss: {train_metrics['loss']:.4f}, "
        f"MAE: {train_metrics['mae']:.4f}, "
        f"RMSE: {train_metrics['rmse']:.4f}, "
        f"R²: {train_metrics['r2']:.4f}"
    )

    # Get detailed train predictions with MC dropout uncertainty
    logger.info("Computing train predictions with MC dropout uncertainty...")
    enable_mc_dropout(model)
    train_mc_results = mc_predict_batch(
        model=model,
        dataloader=train_loader,
        T_ref=config.model.T_ref_K,
        n_samples=config.uncertainty.mc_dropout_samples,
        device=device,
        predict_solubility=False,
    )

    train_preds = train_mc_results["chi_mean"]
    train_preds_std = train_mc_results["chi_std"]
    train_targets = train_mc_results["chi_true"]
    train_temps = train_mc_results["temperatures"]
    train_A = train_mc_results["A_mean"]
    train_B = train_mc_results["B_mean"]
    train_smiles = train_mc_results["smiles"]

    # Save detailed train predictions with uncertainty
    save_detailed_predictions(
        predictions=train_preds,
        targets=train_targets,
        save_path=run_dir / "train_predictions.csv",
        smiles=train_smiles,
        temperatures=train_temps,
        A_params=train_A,
        B_params=train_B,
        uncertainties=train_preds_std,
    )
    logger.info(f"Saved train predictions with uncertainty to train_predictions.csv")

    # Plot train parity with uncertainty
    logger.info("Generating train set uncertainty-aware parity plots...")
    try:
        plot_parity_with_uncertainty(
            y_true=train_targets,
            y_pred_mean=train_preds,
            y_pred_std=train_preds_std,
            save_path=run_dir / "figures" / "parity_with_uncertainty_train",
            config=config,
            title="DFT Chi Parity Plot with Uncertainty (Train Set)",
            error_bar_type="color",
        )
        logger.info("Saved train uncertainty parity plot")
    except Exception as e:
        logger.error(f"Failed to generate train uncertainty parity plot: {e}")

    # ========================================================================
    # Evaluate on test set
    # ========================================================================
    logger.info("\nEvaluating on test set...")

    # Get basic metrics for logging
    test_metrics, _, _ = validate(
        model, test_loader, config, device, return_detailed=False
    )

    logger.info(
        f"Test  - Loss: {test_metrics['loss']:.4f}, "
        f"MAE: {test_metrics['mae']:.4f}, "
        f"RMSE: {test_metrics['rmse']:.4f}, "
        f"R²: {test_metrics['r2']:.4f}"
    )

    # Get detailed test predictions with MC dropout uncertainty
    logger.info("Computing test predictions with MC dropout uncertainty...")
    enable_mc_dropout(model)
    test_mc_results = mc_predict_batch(
        model=model,
        dataloader=test_loader,
        T_ref=config.model.T_ref_K,
        n_samples=config.uncertainty.mc_dropout_samples,
        device=device,
        predict_solubility=False,
    )

    test_preds = test_mc_results["chi_mean"]
    test_preds_std = test_mc_results["chi_std"]
    test_targets = test_mc_results["chi_true"]
    test_temps = test_mc_results["temperatures"]
    test_A = test_mc_results["A_mean"]
    test_B = test_mc_results["B_mean"]
    test_smiles = test_mc_results["smiles"]

    # Save detailed test predictions with uncertainty
    save_detailed_predictions(
        predictions=test_preds,
        targets=test_targets,
        save_path=run_dir / "test_predictions.csv",
        smiles=test_smiles,
        temperatures=test_temps,
        A_params=test_A,
        B_params=test_B,
        uncertainties=test_preds_std,
    )

    # Plot parity with uncertainty
    logger.info("Generating uncertainty-aware parity plots...")
    try:
        plot_parity_with_uncertainty(
            y_true=test_targets,
            y_pred_mean=test_preds,
            y_pred_std=test_preds_std,
            save_path=run_dir / "figures" / "parity_with_uncertainty_test",
            config=config,
            title="DFT Chi Parity Plot with Uncertainty (Test Set)",
            error_bar_type="color",  # Use color-coding for many points
        )
        logger.info("Saved uncertainty parity plot")
    except Exception as e:
        logger.error(f"Failed to generate uncertainty parity plot: {e}")

    # Plot uncertainty calibration
    try:
        plot_error_vs_uncertainty(
            y_true=test_targets,
            y_pred_mean=test_preds,
            y_pred_std=test_preds_std,
            save_path=run_dir / "figures" / "error_vs_uncertainty_test",
            config=config,
        )
        logger.info("Saved error vs uncertainty plot")
    except Exception as e:
        logger.error(f"Failed to generate error vs uncertainty plot: {e}")

    try:
        plot_uncertainty_calibration(
            y_true=test_targets,
            y_pred_mean=test_preds,
            y_pred_std=test_preds_std,
            save_path=run_dir / "figures" / "uncertainty_calibration_test",
            config=config,
        )
        logger.info("Saved uncertainty calibration plot")
    except Exception as e:
        logger.error(f"Failed to generate uncertainty calibration plot: {e}")

    # Generate training history plots
    logger.info("\nGenerating training history plots...")
    try:
        plot_training_history(
            metrics_csv_path=run_dir / "metrics.csv",
            save_path=run_dir / "figures" / "training_curves",
            config=config,
            title="DFT Training History",
            is_multitask=False,
        )
        logger.info(f"Saved training curves to {run_dir / 'figures' / 'training_curves.png'}")
    except Exception as e:
        logger.error(f"Failed to generate training history plot: {e}")

    # Generate additional diagnostic plots
    logger.info("Generating diagnostic plots...")

    # Temperature-colored parity plot for test set
    try:
        plot_parity_with_temperature(
            y_true=test_targets,
            y_pred=test_preds,
            temperatures=test_temps,
            save_path=run_dir / "figures" / "parity_plot_test_temperature",
            config=config,
            title="DFT Chi Parity Plot - Colored by Temperature (Test Set)",
        )
        logger.info(f"Saved temperature-colored parity plot")
    except Exception as e:
        logger.error(f"Failed to generate temperature-colored parity plot: {e}")

    # Residual vs temperature plot
    try:
        plot_residual_vs_temperature(
            y_true=test_targets,
            y_pred=test_preds,
            temperatures=test_temps,
            save_path=run_dir / "figures" / "residual_vs_temperature_test",
            config=config,
            title="Residuals vs Temperature (Test Set)",
        )
        logger.info(f"Saved residual vs temperature plot")
    except Exception as e:
        logger.error(f"Failed to generate residual plot: {e}")

    # Save final summary
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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
