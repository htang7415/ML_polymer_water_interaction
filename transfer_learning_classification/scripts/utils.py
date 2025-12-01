"""
Common utilities for training, evaluation, and metrics.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve
)


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get available device (CUDA or CPU).

    Returns:
        torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Handle case where only one class is present
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = np.nan

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def evaluate_mc_dropout(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 100,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model using MC Dropout for uncertainty estimation.

    Args:
        model: PyTorch model
        X: Input features
        y: True targets
        n_samples: Number of MC samples
        device: Device to use

    Returns:
        Tuple of (mean_predictions, std_predictions, y_true, metrics_dict)
    """
    if device is None:
        device = get_device()

    model.train()  # Keep dropout enabled

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            y_pred = model(X.to(device))
            predictions.append(y_pred.cpu().numpy())

    predictions = np.array(predictions)  # (n_samples, batch_size, 1)

    # Compute mean and std
    mean_pred = np.mean(predictions, axis=0).squeeze()  # (batch_size,)
    std_pred = np.std(predictions, axis=0).squeeze()  # (batch_size,)

    y_true = y.cpu().numpy().squeeze()

    # Compute metrics using mean predictions
    metrics = compute_regression_metrics(y_true, mean_pred)

    return mean_pred, std_pred, y_true, metrics


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaN/Inf in loss
        if not torch.isfinite(loss):
            print(f"ERROR: Loss became {loss.item()} at batch {n_batches}")
            print(f"  Batch X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
            print(f"  Batch y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
            raise ValueError("Loss became NaN or Inf")

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average_loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return avg_loss, predictions, targets


def evaluate_classification_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Evaluate classification model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average_loss, probabilities, targets, accuracy)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Get probabilities
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            n_batches += 1

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    probabilities = np.concatenate(all_probs, axis=0).squeeze()
    targets = np.concatenate(all_targets, axis=0).squeeze()

    # Compute accuracy
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(targets, predictions)

    return avg_loss, probabilities, targets, accuracy


def compute_calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration bins for uncertainty vs error analysis.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predicted uncertainties (standard deviations)
        n_bins: Number of bins

    Returns:
        Tuple of (bin_centers, bin_errors, bin_counts)
    """
    # Compute absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Create bins based on uncertainty quantiles
    bin_edges = np.percentile(y_std, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # Ensure last bin includes maximum

    bin_centers = []
    bin_errors = []
    bin_counts = []

    for i in range(n_bins):
        # Find samples in this bin
        mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])

        if mask.sum() > 0:
            bin_centers.append(np.mean(y_std[mask]))
            bin_errors.append(np.mean(abs_errors[mask]))
            bin_counts.append(mask.sum())
        else:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_errors.append(0)
            bin_counts.append(0)

    return np.array(bin_centers), np.array(bin_errors), np.array(bin_counts)
