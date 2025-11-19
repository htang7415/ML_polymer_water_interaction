"""
Enhanced checkpoint management utilities.

Provides functions for saving and loading model checkpoints with rich metadata,
periodic checkpoint management, and checkpoint resumption support.
"""

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("polymer_chi_ml.checkpoint_utils")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    config,
    save_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    best_val_metric: Optional[float] = None,
    patience_counter: int = 0,
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint with comprehensive metadata.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        config: Configuration object
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        best_val_metric: Best validation metric value (optional)
        patience_counter: Early stopping patience counter
        is_best: Whether this is the best checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        # Model and optimizer state
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        # Metrics
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_metric": best_val_metric,

        # Training state
        "patience_counter": patience_counter,

        # Configuration
        "config": config.to_dict(),

        # Metadata
        "timestamp": datetime.now().isoformat(),
        "is_best": is_best,

        # Random states for reproducibility
        "random_state": {
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
    }

    # Add scheduler state if available
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
    restore_random_state: bool = False,
) -> Dict:
    """
    Load model checkpoint and optionally restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        restore_random_state: Whether to restore random number generator states

    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore random states for exact reproducibility
    if restore_random_state and "random_state" in checkpoint:
        random_states = checkpoint["random_state"]

        if "torch_rng_state" in random_states:
            torch.set_rng_state(random_states["torch_rng_state"])

        if "torch_cuda_rng_state" in random_states and random_states["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(random_states["torch_cuda_rng_state"])

        if "numpy_rng_state" in random_states:
            np.random.set_state(random_states["numpy_rng_state"])

        if "python_rng_state" in random_states:
            random.setstate(random_states["python_rng_state"])

        logger.info("Restored random number generator states")

    # Return metadata for resuming training
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "train_metrics": checkpoint.get("train_metrics", {}),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "best_val_metric": checkpoint.get("best_val_metric"),
        "patience_counter": checkpoint.get("patience_counter", 0),
        "timestamp": checkpoint.get("timestamp", "unknown"),
        "is_best": checkpoint.get("is_best", False),
    }

    logger.info(f"Loaded checkpoint from epoch {metadata['epoch']}")
    return metadata


def manage_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int = 3,
    keep_best: bool = True,
) -> None:
    """
    Clean up old checkpoints, keeping only the most recent N and the best.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to always keep best_model.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    # Get all epoch checkpoints (not best or last)
    epoch_checkpoints = sorted(
        [p for p in checkpoint_dir.glob("epoch_*.pt")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Files to always keep
    keep_files = set()
    if keep_best:
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            keep_files.add(best_path)

    last_path = checkpoint_dir / "last_model.pt"
    if last_path.exists():
        keep_files.add(last_path)

    # Keep the N most recent epoch checkpoints
    for ckpt in epoch_checkpoints[:keep_last_n]:
        keep_files.add(ckpt)

    # Delete old checkpoints
    for ckpt in epoch_checkpoints[keep_last_n:]:
        if ckpt not in keep_files:
            ckpt.unlink()
            logger.info(f"Removed old checkpoint: {ckpt.name}")


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Look for checkpoints in order of preference
    candidates = [
        checkpoint_dir / "last_model.pt",
        checkpoint_dir / "best_model.pt",
    ]

    # Add epoch checkpoints
    epoch_checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(epoch_checkpoints)

    # Return first existing checkpoint
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt

    return None
