"""
Training logic for direct learning: training directly on experimental data without transfer learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Import from parent folder - use absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_scripts_dir = os.path.join(script_dir, '../../scripts')
sys.path.insert(0, os.path.abspath(parent_scripts_dir))

from model import ChiMLP, evaluate_mc_dropout


def train_epoch(
    model: ChiMLP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train model for one epoch.

    Args:
        model: ChiMLP model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze(-1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def eval_epoch(
    model: ChiMLP,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Evaluate model on validation set.

    Args:
        model: ChiMLP model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch).squeeze(-1)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_on_exp_direct(
    exp_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    hyperparams: Dict,
    config: Dict,
    device: str = 'cpu'
) -> Dict:
    """
    Train model directly on experimental data with 5-fold CV (no transfer learning).

    Args:
        exp_folds: List of (X_train, y_train, X_val, y_val) for each fold
        hyperparams: Hyperparameter dictionary
        config: Configuration dictionary
        device: Device to train on

    Returns:
        results: Dictionary with per-fold results and aggregated metrics
    """
    print("\n=== Direct training on experimental data (5-fold CV) ===")

    # Extract hyperparameters
    n_layers = hyperparams['n_layers']
    hidden_dims = hyperparams['hidden_dims']
    dropout_rate = hyperparams['dropout_rate']
    lr = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']
    epochs = hyperparams['epochs']
    batch_size = hyperparams['batch_size']

    n_mc_samples = config['features']['mc_dropout_samples']

    fold_results = []
    val_r2_scores = []
    train_r2_scores = []

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(exp_folds):
        print(f"\nFold {fold_idx + 1}/5")

        # Get input dimension from data
        input_dim = X_train.shape[1]

        # Create new model from scratch (random initialization)
        model = ChiMLP(input_dim, n_layers, hidden_dims, dropout_rate).to(device)
        print(f"  Model: {sum(p.numel() for p in model.parameters())} parameters")

        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = eval_epoch(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Evaluate with MC Dropout
        mu_train, sigma_train, metrics_train = evaluate_mc_dropout(model, X_train, y_train, n_mc_samples, device)
        mu_val, sigma_val, metrics_val = evaluate_mc_dropout(model, X_val, y_val, n_mc_samples, device)

        print(f"  Train: R²={metrics_train['r2']:.4f}, MAE={metrics_train['mae']:.4f}, RMSE={metrics_train['rmse']:.4f}")
        print(f"  Val:   R²={metrics_val['r2']:.4f}, MAE={metrics_val['mae']:.4f}, RMSE={metrics_val['rmse']:.4f}")

        val_r2_scores.append(metrics_val['r2'])
        train_r2_scores.append(metrics_train['r2'])

        fold_results.append({
            'fold': fold_idx,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': metrics_train,
            'val_metrics': metrics_val,
            'train_predictions': {'mu': mu_train, 'sigma': sigma_train, 'y_true': y_train},
            'val_predictions': {'mu': mu_val, 'sigma': sigma_val, 'y_true': y_val},
            'model': model,
        })

    # Compute aggregated statistics
    r2_val_mean = np.mean(val_r2_scores)
    r2_val_std = np.std(val_r2_scores)
    r2_train_mean = np.mean(train_r2_scores)
    r2_train_std = np.std(train_r2_scores)

    print(f"\n5-fold CV results:")
    print(f"  Train R²: {r2_train_mean:.4f} ± {r2_train_std:.4f}")
    print(f"  Val R²:   {r2_val_mean:.4f} ± {r2_val_std:.4f}")

    results = {
        'fold_results': fold_results,
        'r2_val_mean': r2_val_mean,
        'r2_val_std': r2_val_std,
        'r2_train_mean': r2_train_mean,
        'r2_train_std': r2_train_std,
    }

    return results
