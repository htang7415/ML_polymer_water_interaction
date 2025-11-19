"""
Cross-validation analysis utilities.

Provides functions for analyzing cross-validation results, including
statistical tests, confidence intervals, and fold variance analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("polymer_chi_ml.cv_analysis")


def compute_cv_confidence_intervals(
    fold_metrics: pd.DataFrame,
    metrics: List[str],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for CV metrics.

    Args:
        fold_metrics: DataFrame with fold results (rows=folds, cols=metrics)
        metrics: List of metric names to analyze
        confidence: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary mapping metric names to {mean, lower, upper, std}
    """
    results = {}

    for metric in metrics:
        if metric not in fold_metrics.columns:
            logger.warning(f"Metric '{metric}' not found in fold_metrics")
            continue

        values = fold_metrics[metric].values

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        results[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "lower": float(np.percentile(bootstrap_means, lower_percentile)),
            "upper": float(np.percentile(bootstrap_means, upper_percentile)),
            "confidence": confidence,
        }

        logger.info(
            f"{metric}: {results[metric]['mean']:.4f} "
            f"[{results[metric]['lower']:.4f}, {results[metric]['upper']:.4f}] "
            f"({confidence*100:.0f}% CI)"
        )

    return results


def analyze_fold_variance(
    fold_predictions_list: List[pd.DataFrame],
    smiles_col: str = "smiles",
    pred_col: str = "predicted_value",
) -> pd.DataFrame:
    """
    Analyze which samples have high variance across folds.

    Args:
        fold_predictions_list: List of prediction DataFrames (one per fold)
        smiles_col: Column name for polymer SMILES
        pred_col: Column name for predictions

    Returns:
        DataFrame with per-sample variance analysis:
            - smiles
            - mean_prediction
            - std_prediction
            - cv_coefficient (std/mean)
            - n_folds
    """
    # Combine all fold predictions
    all_preds = []
    for fold_idx, fold_df in enumerate(fold_predictions_list):
        df_copy = fold_df[[smiles_col, pred_col]].copy()
        df_copy["fold"] = fold_idx
        all_preds.append(df_copy)

    combined = pd.concat(all_preds, ignore_index=True)

    # Aggregate by SMILES
    variance_analysis = combined.groupby(smiles_col).agg({
        pred_col: ["mean", "std", "count"]
    })

    variance_analysis.columns = ["mean_prediction", "std_prediction", "n_folds"]
    variance_analysis = variance_analysis.reset_index()

    # Compute coefficient of variation
    variance_analysis["cv_coefficient"] = (
        variance_analysis["std_prediction"] / variance_analysis["mean_prediction"].abs()
    )

    # Sort by std descending
    variance_analysis = variance_analysis.sort_values("std_prediction", ascending=False)

    logger.info(
        f"Analyzed fold variance for {len(variance_analysis)} unique samples"
    )

    return variance_analysis


def perform_statistical_tests(
    fold_metrics: pd.DataFrame,
    metric_col: str = "mae",
) -> Dict[str, any]:
    """
    Perform statistical tests on fold metrics.

    Args:
        fold_metrics: DataFrame with fold results
        metric_col: Metric column to analyze

    Returns:
        Dictionary with test results:
            - shapiro_test: Normality test
            - mean, std, median
            - cv_coefficient
    """
    from scipy import stats as scipy_stats

    if metric_col not in fold_metrics.columns:
        raise ValueError(f"Metric '{metric_col}' not found in fold_metrics")

    values = fold_metrics[metric_col].values

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = scipy_stats.shapiro(values)

    # Basic statistics
    results = {
        "metric": metric_col,
        "n_folds": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "cv_coefficient": float(np.std(values) / np.mean(values)),
        "shapiro_test": {
            "statistic": float(shapiro_stat),
            "pvalue": float(shapiro_p),
            "is_normal": bool(shapiro_p > 0.05),
        },
    }

    logger.info(f"Statistical tests for {metric_col}:")
    logger.info(f"  Mean ± Std: {results['mean']:.4f} ± {results['std']:.4f}")
    logger.info(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")

    return results


def plot_cv_metrics_with_error_bars(
    fold_metrics: pd.DataFrame,
    metrics: List[str],
    save_path: Union[str, Path],
    metric_labels: Optional[Dict[str, str]] = None,
    dpi: int = 300,
) -> None:
    """
    Plot bar chart of CV metrics with error bars.

    Args:
        fold_metrics: DataFrame with fold results
        metrics: List of metrics to plot
        save_path: Path to save figure
        metric_labels: Optional mapping of metric names to display labels
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if metric_labels is None:
        metric_labels = {m: m.upper() for m in metrics}

    # Compute means and stds
    means = [fold_metrics[m].mean() for m in metrics]
    stds = [fold_metrics[m].std() for m in metrics]
    labels = [metric_labels.get(m, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(metrics))
    bars = ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, color="steelblue")

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.4f}±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Cross-Validation Metrics (Mean ± Std)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved CV metrics plot to {save_path}")


def plot_fold_boxplots(
    fold_metrics: pd.DataFrame,
    metrics: List[str],
    save_path: Union[str, Path],
    metric_labels: Optional[Dict[str, str]] = None,
    dpi: int = 300,
) -> None:
    """
    Plot box plots showing metric distributions across folds.

    Args:
        fold_metrics: DataFrame with fold results
        metrics: List of metrics to plot
        save_path: Path to save figure
        metric_labels: Optional mapping of metric names to display labels
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if metric_labels is None:
        metric_labels = {m: m.upper() for m in metrics}

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric not in fold_metrics.columns:
            ax.text(0.5, 0.5, f"Metric '{metric}'\nnot found", ha="center", va="center")
            ax.set_title(metric_labels.get(metric, metric))
            continue

        values = fold_metrics[metric].values

        bp = ax.boxplot(
            [values],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
        )

        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11)
        ax.set_xticklabels([""])
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Mean: {mean_val:.4f}")
        ax.legend(fontsize=9)

    fig.suptitle("Cross-Validation Metric Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved fold boxplots to {save_path}")


def save_cv_summary(
    fold_metrics: pd.DataFrame,
    confidence_intervals: Dict,
    statistical_tests: Dict,
    save_path: Union[str, Path],
) -> None:
    """
    Save comprehensive CV summary to JSON.

    Args:
        fold_metrics: DataFrame with fold results
        confidence_intervals: Dict from compute_cv_confidence_intervals()
        statistical_tests: Dict from perform_statistical_tests()
        save_path: Path to save JSON
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_folds": int(len(fold_metrics)),
        "confidence_intervals": confidence_intervals,
        "statistical_tests": statistical_tests,
        "fold_results": fold_metrics.to_dict(orient="records"),
    }

    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved CV summary to {save_path}")
