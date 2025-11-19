"""
SHAP-based feature importance analysis for model interpretability.

Provides functions to compute and visualize SHAP values for understanding
which molecular features contribute most to predictions.

Note: Requires `pip install shap` to use these functions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger("polymer_chi_ml.shap_analysis")


def compute_shap_values(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    background_samples: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for test samples using DeepExplainer.

    Args:
        model: Trained PyTorch model
        X_test: Test features, shape (n_samples, n_features)
        background_samples: Number of background samples for SHAP
        device: Device to run computations on

    Returns:
        shap_values: SHAP values, shape (n_samples, n_features)
        base_values: Base values (expected model output)

    Raises:
        ImportError: If shap package is not installed
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP package is required for feature importance analysis. "
            "Install it with: pip install shap"
        )

    logger.info("Computing SHAP values...")
    model.eval()
    model.to(device)
    X_test = X_test.to(device)

    # Select background samples (use first N samples)
    n_background = min(background_samples, X_test.size(0))
    X_background = X_test[:n_background]

    # Create SHAP explainer for PyTorch model
    # We need to wrap the model to output only chi predictions
    class ChiWrapper(torch.nn.Module):
        def __init__(self, base_model, temperature=298.15):
            super().__init__()
            self.model = base_model
            self.temperature = temperature

        def forward(self, x):
            temp = torch.full((x.size(0),), self.temperature, device=x.device)
            outputs = self.model(x, temperature=temp)
            return outputs["chi"].unsqueeze(1)  # Shape: (batch, 1)

    wrapped_model = ChiWrapper(model)

    # Create DeepExplainer
    explainer = shap.DeepExplainer(wrapped_model, X_background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # shap_values is a list if multiple outputs, array if single output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Take first output

    base_values = explainer.expected_value
    if isinstance(base_values, (list, np.ndarray)):
        base_values = base_values[0] if len(base_values) > 0 else base_values

    logger.info(f"Computed SHAP values for {X_test.size(0)} samples")
    return shap_values, base_values


def get_feature_names(config) -> List[str]:
    """
    Generate feature names for molecular fingerprints and descriptors.

    Args:
        config: Configuration object

    Returns:
        List of feature names
    """
    feature_names = []

    # Morgan fingerprint bits
    n_bits = config.features.morgan_n_bits
    feature_names.extend([f"Morgan_bit_{i}" for i in range(n_bits)])

    # RDKit descriptors (if used)
    if config.features.use_features:
        descriptor_names = [
            "MolWt",
            "MolLogP",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "NumAromaticRings",
            "NumSaturatedRings",
            "NumAliphaticRings",
            "RingCount",
            "FractionCSP3",
            "NumHeteroatoms",
            "HeavyAtomCount",
        ]
        feature_names.extend(descriptor_names)

    return feature_names


def analyze_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Aggregate SHAP values across samples to identify most important features.

    Args:
        shap_values: SHAP values, shape (n_samples, n_features)
        feature_names: List of feature names
        top_k: Number of top features to return

    Returns:
        Dictionary with feature importance analysis:
        - 'mean_abs_shap': Mean absolute SHAP value per feature
        - 'top_features': Names of top K features
        - 'top_indices': Indices of top K features
        - 'top_values': Mean abs SHAP for top K features
    """
    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top K features
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    logger.info(f"Top {top_k} most important features identified")
    for i, (feat, val) in enumerate(zip(top_features, top_values), 1):
        logger.info(f"  {i}. {feat}: {val:.6f}")

    return {
        "mean_abs_shap": mean_abs_shap,
        "top_features": top_features,
        "top_indices": top_indices,
        "top_values": top_values,
    }


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    save_path: Union[str, Path],
    top_k: int = 20,
    dpi: int = 300,
) -> None:
    """
    Create SHAP summary plot showing top features.

    Args:
        shap_values: SHAP values, shape (n_samples, n_features)
        X_test: Test features, shape (n_samples, n_features)
        feature_names: List of feature names
        save_path: Path to save figure
        top_k: Number of top features to show
        dpi: Figure DPI
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP package required. Install with: pip install shap")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Get top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    # Select top features
    shap_values_top = shap_values[:, top_indices]
    X_test_top = X_test[:, top_indices]
    feature_names_top = [feature_names[i] for i in top_indices]

    # Create summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_top,
        X_test_top,
        feature_names=feature_names_top,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SHAP summary plot to {save_path}")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    save_path: Union[str, Path],
    top_k: int = 20,
    dpi: int = 300,
) -> None:
    """
    Create bar plot of mean absolute SHAP values for top features.

    Args:
        shap_values: SHAP values, shape (n_samples, n_features)
        feature_names: List of feature names
        save_path: Path to save figure
        top_k: Number of top features to show
        dpi: Figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Analyze feature importance
    analysis = analyze_feature_importance(shap_values, feature_names, top_k)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(analysis["top_features"]))
    ax.barh(y_pos, analysis["top_values"], align="center", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(analysis["top_features"])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"Top {top_k} Most Important Features", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SHAP bar plot to {save_path}")


def save_shap_values(
    shap_values: np.ndarray,
    base_values: Union[float, np.ndarray],
    feature_names: List[str],
    save_path: Union[str, Path],
) -> None:
    """
    Save SHAP values to disk for later analysis.

    Args:
        shap_values: SHAP values array
        base_values: Base values
        feature_names: List of feature names
        save_path: Path to save NPZ file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_path.with_suffix(".npz"),
        shap_values=shap_values,
        base_values=base_values,
        feature_names=feature_names,
    )

    logger.info(f"Saved SHAP values to {save_path}")


def load_shap_values(load_path: Union[str, Path]) -> Dict:
    """
    Load previously computed SHAP values.

    Args:
        load_path: Path to NPZ file

    Returns:
        Dictionary with shap_values, base_values, and feature_names
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"SHAP values file not found: {load_path}")

    data = np.load(load_path.with_suffix(".npz"), allow_pickle=True)

    return {
        "shap_values": data["shap_values"],
        "base_values": data["base_values"].item() if data["base_values"].ndim == 0 else data["base_values"],
        "feature_names": data["feature_names"].tolist(),
    }
