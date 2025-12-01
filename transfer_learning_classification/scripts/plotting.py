"""
Plotting utilities for transfer learning project.
All plots follow consistent styling (size, font, etc.).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import roc_curve, auc, confusion_matrix


# Global plot settings
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Training Curves",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray],
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Parity Plot",
    figsize: Tuple[float, float] = (5.5, 4.5)
):
    """
    Plot parity plot with uncertainty coloring.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predicted uncertainties (standard deviations), or None
        metrics: Dictionary of metrics (mae, rmse, r2)
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x')

    # Scatter plot with uncertainty coloring if available
    if y_std is not None:
        scatter = ax.scatter(y_true, y_pred, c=y_std, cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted σ')
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')

    ax.set_xlabel('True χ')
    ax.set_ylabel('Predicted χ')
    ax.set_title(title)

    # Add metrics text
    metrics_text = f"MAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}\nR²: {metrics['r2']:.4f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parity plot to {save_path}")


def plot_calibration(
    bin_centers: np.ndarray,
    bin_errors: np.ndarray,
    save_path: str,
    title: str = "Uncertainty Calibration",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot uncertainty calibration (predicted uncertainty vs actual error).

    Args:
        bin_centers: Mean predicted uncertainty in each bin
        bin_errors: Mean absolute error in each bin
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(bin_centers, bin_errors, 'bo-', linewidth=2, markersize=8)

    # Add diagonal line for perfect calibration
    max_val = max(bin_centers.max(), bin_errors.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted σ')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str,
    label: str = "ROC Curve",
    title: str = "ROC Curve",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
        label: Label for the ROC curve
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_multiple_roc_curves(
    y_true_list: List[np.ndarray],
    y_proba_list: List[np.ndarray],
    labels: List[str],
    save_path: str,
    title: str = "ROC Curves",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot multiple ROC curves on the same plot.

    Args:
        y_true_list: List of true labels for each curve
        y_proba_list: List of predicted probabilities for each curve
        labels: List of labels for each curve
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(y_true_list)))

    for i, (y_true, y_proba, label) in enumerate(zip(y_true_list, y_proba_list, labels)):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multiple ROC curves to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: List[str] = ['Insoluble', 'Soluble'],
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_probability_histogram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str,
    title: str = "Predicted Probability Distribution",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot histogram of predicted probabilities for each class.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Separate probabilities by true class
    proba_class0 = y_proba[y_true == 0]
    proba_class1 = y_proba[y_true == 1]

    ax.hist(proba_class0, bins=30, alpha=0.6, label='Insoluble (True)', color='blue')
    ax.hist(proba_class1, bins=30, alpha=0.6, label='Soluble (True)', color='red')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved probability histogram to {save_path}")


def plot_aggregate_classification_curves(
    train_losses_list: List[List[float]],
    val_losses_list: List[List[float]],
    save_path: str,
    title: str = "Aggregate Classification Training Curves",
    figsize: Tuple[float, float] = (4.5, 4.5)
):
    """
    Plot aggregate view of training curves across folds.

    Args:
        train_losses_list: List of training loss curves (one per fold)
        val_losses_list: List of validation loss curves (one per fold)
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to arrays for easier manipulation
    train_losses_array = np.array(train_losses_list)  # (n_folds, n_epochs)
    val_losses_array = np.array(val_losses_list)

    # Compute mean and std
    train_mean = np.mean(train_losses_array, axis=0)
    train_std = np.std(train_losses_array, axis=0)
    val_mean = np.mean(val_losses_array, axis=0)
    val_std = np.std(val_losses_array, axis=0)

    epochs = range(1, len(train_mean) + 1)

    # Plot mean with shaded std
    ax.plot(epochs, train_mean, 'b-', label='Train', linewidth=2)
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')

    ax.plot(epochs, val_mean, 'r-', label='Validation', linewidth=2)
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregate classification curves to {save_path}")
