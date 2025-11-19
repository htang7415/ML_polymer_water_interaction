"""
Data split visualization and analysis.

Provides functions to visualize and verify the quality of train/val/test splits,
including distribution comparisons, feature space coverage, and overlap analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger("polymer_chi_ml.split_visualization")


def plot_split_sizes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Union[str, Path],
    dataset_name: str = "Dataset",
    dpi: int = 300,
) -> None:
    """
    Plot bar chart showing sample counts per split.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        save_path: Path to save figure
        dataset_name: Name of dataset for title
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    splits = ["Train", "Val", "Test"]
    counts = [len(train_df), len(val_df), len(test_df)]
    percentages = [c / sum(counts) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(splits, counts, alpha=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c"])

    # Add count and percentage labels
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(f"{dataset_name} Split Sizes", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved split sizes plot to {save_path}")


def plot_property_distributions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    property_col: str,
    save_path: Union[str, Path],
    xlabel: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Plot overlaid histograms showing property distributions across splits.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        property_col: Column name containing property to plot
        save_path: Path to save figure
        xlabel: X-axis label (defaults to property_col)
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if property_col not in train_df.columns:
        logger.warning(f"Column '{property_col}' not found in data")
        return

    xlabel = xlabel or property_col.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms with transparency
    ax.hist(
        train_df[property_col].dropna(),
        bins=30,
        alpha=0.5,
        label=f"Train (n={len(train_df)})",
        color="#1f77b4",
        density=True,
    )
    ax.hist(
        val_df[property_col].dropna(),
        bins=30,
        alpha=0.5,
        label=f"Val (n={len(val_df)})",
        color="#ff7f0e",
        density=True,
    )
    ax.hist(
        test_df[property_col].dropna(),
        bins=30,
        alpha=0.5,
        label=f"Test (n={len(test_df)})",
        color="#2ca02c",
        density=True,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{xlabel} Distribution Across Splits", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved property distribution plot to {save_path}")


def plot_feature_space_coverage(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    save_path: Union[str, Path],
    method: str = "pca",
    n_samples: int = 1000,
    dpi: int = 300,
) -> None:
    """
    Plot 2D projection of feature space colored by split.

    Args:
        train_features: Training features, shape (n_train, n_features)
        val_features: Validation features, shape (n_val, n_features)
        test_features: Test features, shape (n_test, n_features)
        save_path: Path to save figure
        method: Dimensionality reduction method ("pca" or "tsne")
        n_samples: Max samples to plot (for performance)
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Subsample if needed
    if len(train_features) > n_samples:
        indices = np.random.choice(len(train_features), n_samples, replace=False)
        train_features = train_features[indices]

    if len(val_features) > n_samples // 5:
        indices = np.random.choice(len(val_features), n_samples // 5, replace=False)
        val_features = val_features[indices]

    if len(test_features) > n_samples // 5:
        indices = np.random.choice(len(test_features), n_samples // 5, replace=False)
        test_features = test_features[indices]

    # Combine features
    all_features = np.vstack([train_features, val_features, test_features])
    labels = (
        ["Train"] * len(train_features)
        + ["Val"] * len(val_features)
        + ["Test"] * len(test_features)
    )

    # Dimensionality reduction
    if method.lower() == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(all_features)
        variance = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({variance[0]:.1%} variance)"
        ylabel = f"PC2 ({variance[1]:.1%} variance)"
        title = "PCA Projection of Feature Space"

    elif method.lower() == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) // 5))
        embedded = reducer.fit_transform(all_features)
        xlabel = "t-SNE Dimension 1"
        ylabel = "t-SNE Dimension 2"
        title = "t-SNE Projection of Feature Space"

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"Train": "#1f77b4", "Val": "#ff7f0e", "Test": "#2ca02c"}
    for split_name in ["Train", "Val", "Test"]:
        mask = np.array(labels) == split_name
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=colors[split_name],
            label=split_name,
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature space coverage plot to {save_path}")


def plot_temperature_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Union[str, Path],
    temp_col: str = "temperature_K",
    dpi: int = 300,
) -> None:
    """
    Plot temperature range coverage across splits.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        save_path: Path to save figure
        temp_col: Name of temperature column
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if temp_col not in train_df.columns:
        logger.warning(f"Temperature column '{temp_col}' not found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Overlaid histograms
    ax = axes[0]
    ax.hist(
        train_df[temp_col].dropna(),
        bins=30,
        alpha=0.5,
        label="Train",
        color="#1f77b4",
        density=True,
    )
    ax.hist(
        val_df[temp_col].dropna(),
        bins=30,
        alpha=0.5,
        label="Val",
        color="#ff7f0e",
        density=True,
    )
    ax.hist(
        test_df[temp_col].dropna(),
        bins=30,
        alpha=0.5,
        label="Test",
        color="#2ca02c",
        density=True,
    )
    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Temperature Distribution Across Splits", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Box plots
    ax = axes[1]
    data_to_plot = [
        train_df[temp_col].dropna(),
        val_df[temp_col].dropna(),
        test_df[temp_col].dropna(),
    ]
    bp = ax.boxplot(
        data_to_plot,
        labels=["Train", "Val", "Test"],
        patch_artist=True,
        widths=0.6,
    )

    # Color boxes
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.set_title("Temperature Range Coverage", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved temperature coverage plot to {save_path}")


def analyze_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    property_cols: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Perform statistical tests to compare distributions across splits.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        property_cols: List of columns to analyze
        save_path: Optional path to save JSON results

    Returns:
        Dictionary with statistical test results
    """
    results = {}

    for col in property_cols:
        if col not in train_df.columns:
            continue

        train_vals = train_df[col].dropna().values
        val_vals = val_df[col].dropna().values
        test_vals = test_df[col].dropna().values

        # Kolmogorov-Smirnov test (train vs val, train vs test)
        ks_train_val = stats.ks_2samp(train_vals, val_vals)
        ks_train_test = stats.ks_2samp(train_vals, test_vals)

        # Basic statistics
        results[col] = {
            "train": {
                "mean": float(np.mean(train_vals)),
                "std": float(np.std(train_vals)),
                "min": float(np.min(train_vals)),
                "max": float(np.max(train_vals)),
                "n": int(len(train_vals)),
            },
            "val": {
                "mean": float(np.mean(val_vals)),
                "std": float(np.std(val_vals)),
                "min": float(np.min(val_vals)),
                "max": float(np.max(val_vals)),
                "n": int(len(val_vals)),
            },
            "test": {
                "mean": float(np.mean(test_vals)),
                "std": float(np.std(test_vals)),
                "min": float(np.min(test_vals)),
                "max": float(np.max(test_vals)),
                "n": int(len(test_vals)),
            },
            "ks_test": {
                "train_vs_val": {
                    "statistic": float(ks_train_val.statistic),
                    "pvalue": float(ks_train_val.pvalue),
                },
                "train_vs_test": {
                    "statistic": float(ks_train_test.statistic),
                    "pvalue": float(ks_train_test.pvalue),
                },
            },
        }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path.with_suffix(".json"), "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved split statistics to {save_path}")

    return results


def plot_class_balance(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_col: str,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    dpi: int = 300,
) -> None:
    """
    Plot class balance for classification tasks.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        class_col: Column name containing class labels
        save_path: Path to save figure
        class_names: Optional list of class names
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if class_col not in train_df.columns:
        logger.warning(f"Class column '{class_col}' not found")
        return

    # Get class counts
    train_counts = train_df[class_col].value_counts().sort_index()
    val_counts = val_df[class_col].value_counts().sort_index()
    test_counts = test_df[class_col].value_counts().sort_index()

    # Ensure all splits have same classes
    all_classes = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))
    train_counts = train_counts.reindex(all_classes, fill_value=0)
    val_counts = val_counts.reindex(all_classes, fill_value=0)
    test_counts = test_counts.reindex(all_classes, fill_value=0)

    # Class names
    if class_names is None:
        class_names = [f"Class {c}" for c in all_classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked bar chart
    ax = axes[0]
    x = np.arange(len(all_classes))
    width = 0.25

    ax.bar(x - width, train_counts, width, label="Train", alpha=0.7, color="#1f77b4")
    ax.bar(x, val_counts, width, label="Val", alpha=0.7, color="#ff7f0e")
    ax.bar(x + width, test_counts, width, label="Test", alpha=0.7, color="#2ca02c")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution Across Splits", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Percentage bar chart
    ax = axes[1]
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100

    ax.bar(x - width, train_pct, width, label="Train", alpha=0.7, color="#1f77b4")
    ax.bar(x, val_pct, width, label="Val", alpha=0.7, color="#ff7f0e")
    ax.bar(x + width, test_pct, width, label="Test", alpha=0.7, color="#2ca02c")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Class Balance Across Splits", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved class balance plot to {save_path}")
