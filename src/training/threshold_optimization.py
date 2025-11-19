"""
Threshold optimization for binary classification tasks.

This module provides functions to find the optimal decision threshold
for binary classification based on various metrics (F1, recall, precision, etc.).
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Tuple, Dict, Optional


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    recall_target: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities (0.0-1.0)
        metric: Metric to optimize. Options:
            - "f1": F1 score (harmonic mean of precision and recall)
            - "recall": Recall (sensitivity, true positive rate)
            - "precision": Precision (positive predictive value)
            - "youden": Youden's J statistic (recall + specificity - 1)
        recall_target: If set, only consider thresholds that achieve at least
                      this recall value. Useful for ensuring minimum sensitivity.

    Returns:
        optimal_threshold: Best threshold value (float)
        metrics: Dictionary of metrics at the optimal threshold:
            - "threshold": The optimal threshold
            - "f1": F1 score
            - "recall": Recall (TPR)
            - "precision": Precision
            - "youden": Youden's J statistic
            - "specificity": Specificity (TNR)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
        >>> threshold, metrics = find_optimal_threshold(y_true, y_prob, metric="f1")
        >>> print(f"Optimal threshold: {threshold:.3f}, F1: {metrics['f1']:.3f}")
    """
    # Try thresholds from 0.01 to 0.99 in steps of 0.01
    thresholds = np.linspace(0.01, 0.99, 99)

    best_score = -np.inf
    best_threshold = 0.5
    best_metrics = {}

    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred = (y_prob >= threshold).astype(int)

        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Compute specificity (true negative rate)
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Youden's J statistic: recall + specificity - 1
        # Maximizes the difference between TPR and FPR
        youden = recall + specificity - 1

        # Select metric to optimize
        if metric == "f1":
            score = f1
        elif metric == "recall":
            score = recall
        elif metric == "precision":
            score = precision
        elif metric == "youden":
            score = youden
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'recall', 'precision', or 'youden'")

        # Check recall target constraint (if specified)
        if recall_target is not None and recall < recall_target:
            continue  # Skip this threshold if it doesn't meet recall requirement

        # Update best threshold if this one is better
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                "threshold": threshold,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "youden": youden,
                "specificity": specificity,
            }

    return best_threshold, best_metrics


def analyze_threshold_range(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Analyze classification performance across a range of thresholds.

    Useful for understanding the trade-off between precision and recall,
    and for visualizing ROC curves or precision-recall curves.

    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities (0.0-1.0)
        thresholds: Array of thresholds to evaluate. If None, uses
                   np.linspace(0.01, 0.99, 99).

    Returns:
        Dictionary with arrays for each metric at each threshold:
            - "thresholds": The threshold values
            - "f1": F1 scores
            - "recall": Recall values
            - "precision": Precision values
            - "specificity": Specificity values

    Example:
        >>> results = analyze_threshold_range(y_true, y_prob)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(results["recall"], results["precision"])
        >>> plt.xlabel("Recall")
        >>> plt.ylabel("Precision")
        >>> plt.title("Precision-Recall Curve")
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    f1_scores = []
    recalls = []
    precisions = []
    specificities = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        specificities.append(specificity)

    return {
        "thresholds": thresholds,
        "f1": np.array(f1_scores),
        "recall": np.array(recalls),
        "precision": np.array(precisions),
        "specificity": np.array(specificities),
    }
