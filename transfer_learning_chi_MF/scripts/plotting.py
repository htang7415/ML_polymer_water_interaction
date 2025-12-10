"""
Plotting utilities for transfer learning results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


def setup_plot_style(config: Dict):
    """
    Set up global plot style from configuration.

    Args:
        config: Configuration dictionary
    """
    plt.rcParams['font.size'] = config['plotting']['font_size']
    plt.rcParams['figure.dpi'] = config['plotting']['dpi']


def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: str, config: Dict):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure
        config: Configuration dictionary
    """
    figsize = tuple(config['plotting']['figure_size'])
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train', linewidth=2)
    ax.plot(epochs, val_losses, label='Val', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_training_curves_folds(fold_results: List[Dict], save_path: str, config: Dict):
    """
    Plot training curves for all CV folds on a single figure.

    Args:
        fold_results: List of fold result dictionaries containing train_losses and val_losses
        save_path: Path to save figure
        config: Configuration dictionary
    """
    figsize = tuple(config['plotting']['figure_size'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    # Plot training losses
    for fold_idx, fold_data in enumerate(fold_results):
        epochs = np.arange(1, len(fold_data['train_losses']) + 1)
        ax1.plot(epochs, fold_data['train_losses'], color=colors[fold_idx],
                linewidth=1.5, alpha=0.7, label=f'Fold {fold_idx + 1}')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss (All Folds)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot validation losses
    for fold_idx, fold_data in enumerate(fold_results):
        epochs = np.arange(1, len(fold_data['val_losses']) + 1)
        ax2.plot(epochs, fold_data['val_losses'], color=colors[fold_idx],
                linewidth=1.5, alpha=0.7, label=f'Fold {fold_idx + 1}')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Validation Loss (All Folds)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved fold training curves to {save_path}")


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    metrics: Dict,
    title: str,
    save_path: str,
    config: Dict,
    figsize: tuple = None
):
    """
    Plot parity plot with uncertainty as color.

    Args:
        y_true: True values
        y_pred: Predicted mean values
        sigma: Predicted standard deviations
        metrics: Dictionary with R², MAE, RMSE
        title: Plot title
        save_path: Path to save figure
        config: Configuration dictionary
        figsize: Figure size as (width, height) tuple. If None, uses config default.
    """
    if figsize is None:
        figsize = tuple(config['plotting']['figure_size'])
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot with uncertainty as color
    scatter = ax.scatter(y_true, y_pred, c=sigma, cmap='viridis', alpha=0.6, s=20)
    cbar = plt.colorbar(scatter, ax=ax, label='Predictive σ')

    # y = x diagonal
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

    # Add metrics text
    text_str = f"R² = {metrics['r2']:.4f}\nMAE = {metrics['mae']:.4f}\nRMSE = {metrics['rmse']:.4f}"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('True χ')
    ax.set_ylabel('Predicted χ')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved parity plot to {save_path}")


def plot_parity_folds(
    fold_results: List[Dict],
    mode: str,
    save_path: str,
    config: Dict
):
    """
    Plot combined parity plot for all folds with different colors.

    Args:
        fold_results: List of fold result dictionaries
        mode: 'train' or 'val'
        save_path: Path to save figure
        config: Configuration dictionary
    """
    figsize = tuple(config['plotting']['figure_size'])
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    all_true = []
    all_pred = []

    # Collect all metrics
    all_r2 = []
    all_mae = []
    all_rmse = []

    for fold_idx, fold_data in enumerate(fold_results):
        if mode == 'train':
            pred_data = fold_data['train_predictions']
            metrics = fold_data['train_metrics']
        else:
            pred_data = fold_data['val_predictions']
            metrics = fold_data['val_metrics']

        y_true = pred_data['y_true']
        y_pred = pred_data['mu']

        all_true.append(y_true)
        all_pred.append(y_pred)

        all_r2.append(metrics['r2'])
        all_mae.append(metrics['mae'])
        all_rmse.append(metrics['rmse'])

        # Plot fold
        ax.scatter(y_true, y_pred, c=colors[fold_idx], alpha=0.6, s=20, label=f'Fold {fold_idx + 1}')

    # y = x diagonal
    all_true_concat = np.concatenate(all_true)
    all_pred_concat = np.concatenate(all_pred)
    min_val = min(all_true_concat.min(), all_pred_concat.min())
    max_val = max(all_true_concat.max(), all_pred_concat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

    # Add aggregated metrics text
    text_str = (f"R² = {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}\n"
                f"MAE = {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}\n"
                f"RMSE = {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('True χ')
    ax.set_ylabel('Predicted χ')
    title = f"Experimental {'Train' if mode == 'train' else 'Val'} (5-fold CV)"
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved fold parity plot to {save_path}")


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    title: str,
    save_path: str,
    config: Dict,
    n_bins: int = 10
):
    """
    Plot uncertainty calibration: predicted σ vs actual error.

    Args:
        y_true: True values
        y_pred: Predicted mean values
        sigma: Predicted standard deviations
        title: Plot title
        save_path: Path to save figure
        config: Configuration dictionary
        n_bins: Number of bins for calibration
    """
    figsize = tuple(config['plotting']['figure_size'])
    fig, ax = plt.subplots(figsize=figsize)

    # Compute absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Create bins based on sigma quantiles
    bin_edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-10  # Ensure last bin includes maximum

    bin_sigma = []
    bin_mae = []

    for i in range(n_bins):
        mask = (sigma >= bin_edges[i]) & (sigma < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_sigma.append(sigma[mask].mean())
            bin_mae.append(abs_errors[mask].mean())

    # Plot
    if len(bin_sigma) > 0:
        ax.plot(bin_sigma, bin_mae, 'o-', linewidth=2, markersize=8)

        # Add diagonal for perfect calibration
        min_val = min(min(bin_sigma), min(bin_mae))
        max_val = max(max(bin_sigma), max(bin_mae))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted σ (per bin)')
    ax.set_ylabel('Mean Absolute Error (per bin)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved calibration plot to {save_path}")


def plot_calibration_folds(
    fold_results: List[Dict],
    mode: str,
    save_path: str,
    config: Dict
):
    """
    Plot combined calibration for all folds.

    Args:
        fold_results: List of fold result dictionaries
        mode: 'train' or 'val'
        save_path: Path to save figure
        config: Configuration dictionary
    """
    # Concatenate all predictions
    all_true = []
    all_pred = []
    all_sigma = []

    for fold_data in fold_results:
        if mode == 'train':
            pred_data = fold_data['train_predictions']
        else:
            pred_data = fold_data['val_predictions']

        all_true.append(pred_data['y_true'])
        all_pred.append(pred_data['mu'])
        all_sigma.append(pred_data['sigma'])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    sigma = np.concatenate(all_sigma)

    title = f"Experimental {'Train' if mode == 'train' else 'Val'} Calibration (5-fold)"
    n_bins = config['plotting']['n_bins_exp']

    plot_calibration(y_true, y_pred, sigma, title, save_path, config, n_bins)
