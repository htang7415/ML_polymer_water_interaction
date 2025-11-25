"""
Plotting utilities for polymer-water χ prediction.

Handles:
- Training curves (loss vs epoch)
- Parity plots with uncertainty coloring
- Uncertainty calibration plots
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os


# Global plot settings
plt.rcParams['font.size'] = 12


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves"
):
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {save_path}")


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Parity Plot",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot parity plot with uncertainty coloring.

    Args:
        y_true: True chi values
        y_pred: Predicted chi values (mean)
        y_std: Predicted chi std (uncertainty)
        metrics: Dictionary with 'r2', 'mae', 'rmse'
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot colored by uncertainty
    scatter = ax.scatter(
        y_true,
        y_pred,
        c=y_std,
        cmap='viridis',
        s=20,
        alpha=0.6,
        edgecolors='none'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Predicted σ')

    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)

    # Add metrics text
    text_str = f"R² = {metrics['r2']:.3f}\n"
    text_str += f"MAE = {metrics['mae']:.4f}\n"
    text_str += f"RMSE = {metrics['rmse']:.4f}"

    ax.text(
        0.05, 0.95,
        text_str,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )

    ax.set_xlabel('True χ')
    ax.set_ylabel('Predicted χ')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved parity plot to {save_path}")


def plot_parity_multicolor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_labels: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Out-of-Fold Parity Plot",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot parity plot with different colors for different folds.

    Args:
        y_true: True chi values
        y_pred: Predicted chi values
        fold_labels: Fold labels for each point
        metrics: Dictionary with 'r2_mean', 'r2_std', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std'
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique folds
    unique_folds = np.unique(fold_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_folds)))

    # Plot each fold with different color
    for i, fold_id in enumerate(unique_folds):
        mask = fold_labels == fold_id
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            c=[colors[i]],
            s=20,
            alpha=0.6,
            edgecolors='none',
            label=f'Fold {fold_id}'
        )

    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)

    # Add metrics text
    text_str = f"R² = {metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}\n"
    text_str += f"MAE = {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}\n"
    text_str += f"RMSE = {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}"

    ax.text(
        0.05, 0.95,
        text_str,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )

    ax.set_xlabel('True χ')
    ax.set_ylabel('Predicted χ')
    ax.set_title(f'{title} (colors = folds)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved multi-fold parity plot to {save_path}")


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    save_path: str,
    n_bins: int = 10,
    title: str = "Uncertainty Calibration"
):
    """
    Plot uncertainty calibration: predicted std vs actual error.

    Args:
        y_true: True values
        y_pred: Predicted values (mean)
        y_std: Predicted std (uncertainty)
        save_path: Path to save the plot
        n_bins: Number of bins for calibration
        title: Plot title
    """
    # Compute absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Create quantile bins based on predicted std
    bin_edges = np.percentile(y_std, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # Ensure last bin includes maximum

    # Compute mean std and mean error in each bin
    mean_stds = []
    mean_errors = []

    for i in range(n_bins):
        mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_stds.append(y_std[mask].mean())
            mean_errors.append(abs_errors[mask].mean())

    mean_stds = np.array(mean_stds)
    mean_errors = np.array(mean_errors)

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(mean_stds, mean_errors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add diagonal line (perfect calibration)
    max_val = max(mean_stds.max(), mean_errors.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted σ')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved calibration plot to {save_path}")


def plot_dft_results(
    train_results: Dict,
    val_results: Dict,
    test_results: Dict,
    history: Dict,
    output_dir: str
):
    """
    Plot all DFT pretraining results.

    Args:
        train_results: Training set MC dropout results
        val_results: Validation set MC dropout results
        test_results: Test set MC dropout results
        history: Training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Training curves
    plot_training_curves(
        history=history,
        save_path=os.path.join(output_dir, 'dft_training_curves.png'),
        title='DFT Pretraining'
    )

    # Parity plots for each split (use 5.5x4.5 for pretrain as specified)
    for split_name, results in [('train', train_results), ('val', val_results), ('test', test_results)]:
        plot_parity(
            y_true=results['y_true'],
            y_pred=results['chi_mean'],
            y_std=results['chi_std'],
            metrics={'r2': results['r2'], 'mae': results['mae'], 'rmse': results['rmse']},
            save_path=os.path.join(output_dir, f'dft_{split_name}_parity.png'),
            title=f'DFT {split_name.capitalize()}',
            figsize=(5.5, 4.5)
        )

    # Calibration plots
    for split_name, results in [('train', train_results), ('val', val_results), ('test', test_results)]:
        plot_calibration(
            y_true=results['y_true'],
            y_pred=results['chi_mean'],
            y_std=results['chi_std'],
            save_path=os.path.join(output_dir, f'dft_{split_name}_calibration.png'),
            n_bins=10,
            title=f'DFT {split_name.capitalize()} Calibration'
        )


def plot_cv_results(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    all_y_std: np.ndarray,
    all_fold_labels: np.ndarray,
    fold_metrics: List[Dict[str, float]],
    output_dir: str,
    prefix: str = 'exp'
):
    """
    Plot cross-validation results.

    Args:
        all_y_true: All true values (concatenated from all folds)
        all_y_pred: All predicted values
        all_y_std: All predicted stds
        all_fold_labels: Fold label for each point
        fold_metrics: List of metrics for each fold
        output_dir: Directory to save plots
        prefix: Prefix for saved files ('exp' for experimental)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute aggregate metrics
    r2_values = [m['r2'] for m in fold_metrics]
    mae_values = [m['mae'] for m in fold_metrics]
    rmse_values = [m['rmse'] for m in fold_metrics]

    agg_metrics = {
        'r2_mean': np.mean(r2_values),
        'r2_std': np.std(r2_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values)
    }

    # Multi-color parity plot
    plot_parity_multicolor(
        y_true=all_y_true,
        y_pred=all_y_pred,
        fold_labels=all_fold_labels,
        metrics=agg_metrics,
        save_path=os.path.join(output_dir, f'{prefix}_val_parity.png'),
        title='Out-of-Fold Validation'
    )

    # Calibration plot (use fewer bins for experimental data)
    plot_calibration(
        y_true=all_y_true,
        y_pred=all_y_pred,
        y_std=all_y_std,
        save_path=os.path.join(output_dir, f'{prefix}_val_calibration.png'),
        n_bins=5,
        title='Experimental Validation Calibration'
    )


if __name__ == '__main__':
    print("Testing plotting utilities...")

    # Create dummy data
    n_samples = 200

    # Test training curves
    history = {
        'train_loss': [1.0 - 0.01 * i for i in range(50)],
        'val_loss': [1.1 - 0.01 * i for i in range(50)]
    }

    plot_training_curves(
        history=history,
        save_path='test_training_curves.png',
        title='Test Training'
    )

    # Test parity plot
    y_true = np.random.randn(n_samples)
    y_pred = y_true + np.random.randn(n_samples) * 0.3
    y_std = np.abs(np.random.randn(n_samples) * 0.2 + 0.1)

    metrics = {
        'r2': 0.85,
        'mae': 0.25,
        'rmse': 0.35
    }

    plot_parity(
        y_true=y_true,
        y_pred=y_pred,
        y_std=y_std,
        metrics=metrics,
        save_path='test_parity.png',
        title='Test Parity'
    )

    # Test multi-color parity
    fold_labels = np.random.randint(0, 5, n_samples)
    agg_metrics = {
        'r2_mean': 0.80,
        'r2_std': 0.05,
        'mae_mean': 0.30,
        'mae_std': 0.05,
        'rmse_mean': 0.40,
        'rmse_std': 0.06
    }

    plot_parity_multicolor(
        y_true=y_true,
        y_pred=y_pred,
        fold_labels=fold_labels,
        metrics=agg_metrics,
        save_path='test_multicolor_parity.png',
        title='Test Multi-Fold'
    )

    # Test calibration plot
    plot_calibration(
        y_true=y_true,
        y_pred=y_pred,
        y_std=y_std,
        save_path='test_calibration.png',
        title='Test Calibration'
    )

    print("\nPlotting utilities test completed successfully!")
    print("Generated test plots: test_training_curves.png, test_parity.png, test_multicolor_parity.png, test_calibration.png")
