"""
Per-polymer error analysis for identifying problematic predictions.

Provides functions to aggregate errors by polymer (SMILES) and analyze
which polymer structures are most difficult to predict.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("polymer_chi_ml.polymer_analysis")


def aggregate_polymer_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate prediction errors by polymer (SMILES).

    Args:
        predictions_df: DataFrame with columns:
            - smiles
            - true_value
            - predicted_value
            - abs_error
            - temperature_K (optional)
            - A_parameter (optional)
            - B_parameter (optional)

    Returns:
        DataFrame with per-polymer metrics:
            - smiles
            - n_measurements: Number of measurements for this polymer
            - mean_abs_error: Mean absolute error across temperatures
            - std_abs_error: Standard deviation of absolute errors
            - max_abs_error: Maximum absolute error
            - mean_error: Mean signed error (bias)
            - mean_temperature: Average temperature of measurements
            - temperature_range: Range of temperatures measured
            - A_parameter_mean: Mean A parameter (if available)
            - B_parameter_mean: Mean B parameter (if available)
    """
    if "smiles" not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'smiles' column")

    if "abs_error" not in predictions_df.columns:
        if "error" not in predictions_df.columns:
            if "true_value" in predictions_df.columns and "predicted_value" in predictions_df.columns:
                predictions_df["error"] = predictions_df["predicted_value"] - predictions_df["true_value"]
                predictions_df["abs_error"] = predictions_df["error"].abs()
            else:
                raise ValueError("Cannot compute errors from available columns")
        else:
            predictions_df["abs_error"] = predictions_df["error"].abs()

    # Aggregation functions
    agg_dict = {
        "abs_error": ["mean", "std", "max"],
        "error": "mean",  # Signed error for bias detection
    }

    # Add temperature aggregations if available
    if "temperature_K" in predictions_df.columns:
        agg_dict["temperature_K"] = ["mean", lambda x: x.max() - x.min()]

    # Add parameter aggregations if available
    if "A_parameter" in predictions_df.columns:
        agg_dict["A_parameter"] = "mean"
    if "B_parameter" in predictions_df.columns:
        agg_dict["B_parameter"] = "mean"

    # Count measurements
    agg_dict["true_value"] = "count"

    # Group by SMILES and aggregate
    polymer_metrics = predictions_df.groupby("smiles").agg(agg_dict)

    # Flatten multi-level column names
    polymer_metrics.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in polymer_metrics.columns
    ]

    # Rename columns for clarity
    rename_map = {
        "abs_error_mean": "mean_abs_error",
        "abs_error_std": "std_abs_error",
        "abs_error_max": "max_abs_error",
        "error_mean": "mean_error",
        "true_value_count": "n_measurements",
        "temperature_K_mean": "mean_temperature",
        "temperature_K_<lambda_0>": "temperature_range",
        "A_parameter_mean": "A_parameter_mean",
        "B_parameter_mean": "B_parameter_mean",
    }

    polymer_metrics = polymer_metrics.rename(columns=rename_map)
    polymer_metrics = polymer_metrics.reset_index()

    logger.info(f"Aggregated metrics for {len(polymer_metrics)} unique polymers")
    return polymer_metrics


def identify_problematic_polymers(
    polymer_metrics_df: pd.DataFrame,
    threshold: float = 0.2,
    metric: str = "mean_abs_error",
) -> pd.DataFrame:
    """
    Identify polymers with high prediction errors.

    Args:
        polymer_metrics_df: DataFrame from aggregate_polymer_metrics()
        threshold: Error threshold for identifying problematic polymers
        metric: Metric to use for threshold (default: "mean_abs_error")

    Returns:
        DataFrame of problematic polymers sorted by error
    """
    if metric not in polymer_metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in polymer_metrics_df")

    problematic = polymer_metrics_df[polymer_metrics_df[metric] > threshold].copy()
    problematic = problematic.sort_values(metric, ascending=False)

    logger.info(
        f"Identified {len(problematic)} problematic polymers "
        f"(>{threshold} {metric})"
    )

    return problematic


def plot_polymer_error_distribution(
    polymer_metrics_df: pd.DataFrame,
    save_path: Union[str, Path],
    metric: str = "mean_abs_error",
    dpi: int = 300,
) -> None:
    """
    Plot distribution of per-polymer errors.

    Args:
        polymer_metrics_df: DataFrame from aggregate_polymer_metrics()
        save_path: Path to save figure
        metric: Error metric to plot
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(polymer_metrics_df[metric], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Number of Polymers", fontsize=12)
    ax.set_title("Distribution of Per-Polymer Errors", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    mean_val = polymer_metrics_df[metric].mean()
    median_val = polymer_metrics_df[metric].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.4f}")
    ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.4f}")
    ax.legend()

    # Box plot
    ax = axes[1]
    ax.boxplot(polymer_metrics_df[metric], vert=True)
    ax.set_ylabel(metric.replace("_", " ").Title(), fontsize=12)
    ax.set_title("Per-Polymer Error Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved polymer error distribution plot to {save_path}")


def plot_top_worst_polymers(
    polymer_metrics_df: pd.DataFrame,
    save_path: Union[str, Path],
    n: int = 10,
    metric: str = "mean_abs_error",
    dpi: int = 300,
) -> None:
    """
    Plot bar chart of worst-performing polymers.

    Args:
        polymer_metrics_df: DataFrame from aggregate_polymer_metrics()
        save_path: Path to save figure
        n: Number of top worst polymers to show
        metric: Error metric to plot
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Get top N worst polymers
    worst_polymers = polymer_metrics_df.nlargest(n, metric)

    # Create truncated SMILES labels for readability
    labels = []
    for smiles in worst_polymers["smiles"]:
        if len(smiles) > 30:
            labels.append(smiles[:27] + "...")
        else:
            labels.append(smiles)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, worst_polymers[metric], align="center", alpha=0.7, color="coral")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Worst at top
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Top {n} Worst-Performing Polymers", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    # Add n_measurements annotation
    if "n_measurements" in worst_polymers.columns:
        for i, (idx, row) in enumerate(worst_polymers.iterrows()):
            ax.text(
                row[metric] + 0.01,
                i,
                f"(n={int(row['n_measurements'])})",
                va="center",
                fontsize=9,
                color="gray",
            )

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved top worst polymers plot to {save_path}")


def plot_error_vs_measurements(
    polymer_metrics_df: pd.DataFrame,
    save_path: Union[str, Path],
    metric: str = "mean_abs_error",
    dpi: int = 300,
) -> None:
    """
    Plot error vs number of measurements to check if data quantity affects accuracy.

    Args:
        polymer_metrics_df: DataFrame from aggregate_polymer_metrics()
        save_path: Path to save figure
        metric: Error metric to plot
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if "n_measurements" not in polymer_metrics_df.columns:
        logger.warning("n_measurements not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with transparency
    ax.scatter(
        polymer_metrics_df["n_measurements"],
        polymer_metrics_df[metric],
        alpha=0.5,
        s=50,
        edgecolors="none",
    )

    # Add trend line if scipy available
    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            polymer_metrics_df["n_measurements"],
            polymer_metrics_df[metric],
        )
        x_line = np.array([polymer_metrics_df["n_measurements"].min(), polymer_metrics_df["n_measurements"].max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r--", linewidth=2, label=f"r={r_value:.3f}, p={p_value:.3f}")
        ax.legend()
    except ImportError:
        pass

    ax.set_xlabel("Number of Measurements", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Error vs Number of Measurements", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved error vs measurements plot to {save_path}")


def save_polymer_metrics(
    polymer_metrics_df: pd.DataFrame,
    save_path: Union[str, Path],
) -> None:
    """
    Save per-polymer metrics to CSV.

    Args:
        polymer_metrics_df: DataFrame from aggregate_polymer_metrics()
        save_path: Path to save CSV
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    polymer_metrics_df.to_csv(save_path.with_suffix(".csv"), index=False)
    logger.info(f"Saved polymer metrics to {save_path}")
