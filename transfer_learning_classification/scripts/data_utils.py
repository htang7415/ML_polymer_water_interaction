"""
Data loading utilities for transfer learning project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


def load_dft_data(csv_path: str) -> pd.DataFrame:
    """
    Load DFT chi data from CSV.

    Args:
        csv_path: Path to the DFT CSV file

    Returns:
        DataFrame with SMILES and chi columns
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded DFT data: {len(df)} samples")
    return df


def load_binary_solubility_data(csv_path: str) -> pd.DataFrame:
    """
    Load binary solubility data from CSV.

    Args:
        csv_path: Path to the binary solubility CSV file

    Returns:
        DataFrame with SMILES and water_soluble columns
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded binary solubility data: {len(df)} samples")
    print(f"  Soluble (1): {(df['water_soluble'] == 1).sum()}")
    print(f"  Insoluble (0): {(df['water_soluble'] == 0).sum()}")
    return df


def load_precomputed_features(csv_path: str) -> pd.DataFrame:
    """
    Load precomputed features from CSV.

    Args:
        csv_path: Path to the precomputed features CSV file

    Returns:
        DataFrame with precomputed features
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded precomputed features: {len(df)} samples, {len(df.columns)} columns")
    return df


def split_dft_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DFT data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        test_ratio: Proportion of test data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True
    )

    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True
    )

    print(f"\nDFT data split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def extract_feature_columns(df: pd.DataFrame, feature_mode: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract feature matrix and target from precomputed features DataFrame.

    Args:
        df: DataFrame with precomputed features
        feature_mode: One of "fp", "desc", "fp_desc"

    Returns:
        Tuple of (X, y) where X is feature matrix and y is target (or None for classification)
    """
    # Extract Morgan fingerprint columns
    fp_cols = [col for col in df.columns if col.startswith('mf_')]

    # Extract descriptor columns
    desc_cols = [col for col in df.columns if col.startswith('desc_')]

    # Build feature matrix based on mode
    if feature_mode == "fp":
        X = df[fp_cols].values
    elif feature_mode == "desc":
        X = df[desc_cols].values
    elif feature_mode == "fp_desc":
        X = np.concatenate([df[fp_cols].values, df[desc_cols].values], axis=1)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    # Extract target if available
    y = None
    if 'chi' in df.columns:
        y = df['chi'].values
    elif 'water_soluble' in df.columns:
        y = df['water_soluble'].values

    return X, y


def get_feature_indices(df: pd.DataFrame, feature_mode: str) -> dict:
    """
    Get indices for different feature types in the feature matrix.

    Args:
        df: DataFrame with precomputed features
        feature_mode: One of "fp", "desc", "fp_desc"

    Returns:
        Dictionary with 'fp', 'desc', and 'total' indices
    """
    fp_cols = [col for col in df.columns if col.startswith('mf_')]
    desc_cols = [col for col in df.columns if col.startswith('desc_')]

    n_fp = len(fp_cols)
    n_desc = len(desc_cols)

    indices = {}

    if feature_mode == "fp":
        indices['fp'] = (0, n_fp)
        indices['desc'] = None
        indices['total'] = n_fp
    elif feature_mode == "desc":
        indices['fp'] = None
        indices['desc'] = (0, n_desc)
        indices['total'] = n_desc
    elif feature_mode == "fp_desc":
        indices['fp'] = (0, n_fp)
        indices['desc'] = (n_fp, n_fp + n_desc)
        indices['total'] = n_fp + n_desc
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return indices
