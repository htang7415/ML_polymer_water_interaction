"""
Feature engineering for polymer-water χ prediction.

Handles:
- SMILES canonicalization
- Morgan fingerprint computation
- Molecular descriptor computation and scaling
- Feature mode selection (fp, desc, fp_desc)
"""

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Tuple, List, Dict, Optional
import yaml
import warnings

warnings.filterwarnings('ignore')


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def compute_morgan_fingerprint(smiles: str, radius: int = 3, nBits: int = 1024) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprint for a molecule.

    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius
        nBits: Number of bits in fingerprint

    Returns:
        Binary fingerprint as numpy array or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp)
    except:
        return None


def get_all_rdkit_descriptors() -> List[Tuple[str, callable]]:
    """
    Get all available RDKit molecular descriptors.

    Returns:
        List of (descriptor_name, descriptor_function) tuples
    """
    descriptor_list = []
    for name, func in Descriptors.descList:
        descriptor_list.append((name, func))
    return descriptor_list


def compute_descriptors(smiles: str, descriptor_list: List[Tuple[str, callable]]) -> Optional[Dict[str, float]]:
    """
    Compute molecular descriptors for a molecule.

    Args:
        smiles: SMILES string
        descriptor_list: List of (name, function) tuples

    Returns:
        Dictionary of descriptor values or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        desc_dict = {}
        for name, func in descriptor_list:
            try:
                value = func(mol)
                desc_dict[f'desc_{name}'] = value
            except:
                desc_dict[f'desc_{name}'] = np.nan

        return desc_dict
    except:
        return None


def filter_and_scale_descriptors(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    desc_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], Dict[str, float], Dict[str, float]]:
    """
    Filter out invalid descriptors and compute scaling parameters from training data.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe (optional)
        test_df: Test dataframe (optional)
        desc_columns: List of descriptor column names (if None, auto-detect)

    Returns:
        Tuple of (train_df, val_df, test_df, valid_desc_columns, desc_means, desc_stds)
    """
    # Auto-detect descriptor columns
    if desc_columns is None:
        desc_columns = [col for col in train_df.columns if col.startswith('desc_')]

    # Filter out descriptors with NaN/Inf or constant values in training data
    valid_desc_columns = []
    desc_means = {}
    desc_stds = {}

    for col in desc_columns:
        values = train_df[col].values

        # Check for NaN or Inf
        if np.any(~np.isfinite(values)):
            continue

        # Check for constant values
        if np.std(values) < 1e-10:
            continue

        valid_desc_columns.append(col)
        desc_means[col] = np.mean(values)
        desc_stds[col] = np.std(values)

    print(f"Filtered descriptors: {len(desc_columns)} -> {len(valid_desc_columns)}")

    # Scale descriptors using training statistics
    def scale_descriptors(df, desc_cols, means, stds):
        df_scaled = df.copy()
        for col in desc_cols:
            df_scaled[col] = (df[col] - means[col]) / stds[col]
        return df_scaled

    train_df = scale_descriptors(train_df, valid_desc_columns, desc_means, desc_stds)

    if val_df is not None:
        val_df = scale_descriptors(val_df, valid_desc_columns, desc_means, desc_stds)

    if test_df is not None:
        test_df = scale_descriptors(test_df, valid_desc_columns, desc_means, desc_stds)

    return train_df, val_df, test_df, valid_desc_columns, desc_means, desc_stds


def build_features(
    df: pd.DataFrame,
    feature_mode: str,
    valid_desc_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix based on feature mode.

    Args:
        df: DataFrame with precomputed fingerprints and descriptors
        feature_mode: One of 'fp', 'desc', 'fp_desc'
        valid_desc_columns: List of valid descriptor column names

    Returns:
        Tuple of (X, T, y) where:
            X: feature matrix (N, D)
            T: temperature array (N,)
            y: target chi values (N,)
    """
    # Extract fingerprint columns
    fp_columns = [col for col in df.columns if col.startswith('mf_')]

    if feature_mode == 'fp':
        X = df[fp_columns].values
    elif feature_mode == 'desc':
        X = df[valid_desc_columns].values
    elif feature_mode == 'fp_desc':
        X = np.concatenate([
            df[fp_columns].values,
            df[valid_desc_columns].values
        ], axis=1)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    T = df['temp'].values
    y = df['chi'].values

    return X, T, y


def validate_features(df: pd.DataFrame, desc_columns: List[str]) -> pd.DataFrame:
    """
    Check for NaN/Inf in descriptor columns and fill with 0.

    Args:
        df: DataFrame with scaled descriptors
        desc_columns: List of descriptor column names

    Returns:
        Cleaned DataFrame
    """
    for col in desc_columns:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col]).sum()

        if n_nan > 0 or n_inf > 0:
            print(f"Warning: {col} has {n_nan} NaN and {n_inf} Inf values after scaling - filling with 0")
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    return df


def precompute_features(
    input_csv: str,
    output_csv: str,
    radius: int = 3,
    nBits: int = 1024
) -> pd.DataFrame:
    """
    Precompute features for a dataset.

    Args:
        input_csv: Path to input CSV with SMILES, temp, chi
        output_csv: Path to save output CSV with features
        radius: Morgan fingerprint radius
        nBits: Number of fingerprint bits

    Returns:
        DataFrame with precomputed features
    """
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Processing {len(df)} rows...")

    # Get descriptor list
    descriptor_list = get_all_rdkit_descriptors()
    print(f"Computing {len(descriptor_list)} RDKit descriptors...")

    # Process each row
    results = []
    failed = 0

    for idx, row in df.iterrows():
        smiles = row['SMILES']

        # Canonicalize SMILES
        cano_smiles = canonicalize_smiles(smiles)
        if cano_smiles is None:
            failed += 1
            continue

        # Compute Morgan fingerprint
        fp = compute_morgan_fingerprint(cano_smiles, radius, nBits)
        if fp is None:
            failed += 1
            continue

        # Compute descriptors
        desc_dict = compute_descriptors(cano_smiles, descriptor_list)
        if desc_dict is None:
            failed += 1
            continue

        # Build result row
        result = {
            'SMILES': smiles,
            'cano_smiles': cano_smiles,
            'temp': row['temp'],
            'chi': row['chi']
        }

        # Add fingerprint bits
        for i, bit in enumerate(fp):
            result[f'mf_{i}'] = int(bit)

        # Add descriptors
        result.update(desc_dict)

        results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows ({failed} failed)")

    print(f"Total failed: {failed}/{len(df)}")

    # Create DataFrame
    df_features = pd.DataFrame(results)

    # Save to CSV
    print(f"Saving to {output_csv}...")
    df_features.to_csv(output_csv, index=False)

    print(f"Done! Shape: {df_features.shape}")

    return df_features


def main():
    """Main function for precomputing features."""
    parser = argparse.ArgumentParser(description='Precompute Morgan fingerprints and descriptors')
    parser.add_argument('--precompute', action='store_true', help='Precompute features for all datasets')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')

    args = parser.parse_args()

    if args.precompute:
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Precompute DFT features
        print("\n=== Precomputing DFT features ===")
        precompute_features(
            input_csv=config['data']['dft_raw'],
            output_csv=config['data']['dft_features'],
            radius=config['features']['morgan']['radius'],
            nBits=config['features']['morgan']['nBits']
        )

        # Precompute experimental features
        print("\n=== Precomputing experimental features ===")
        precompute_features(
            input_csv=config['data']['exp_raw'],
            output_csv=config['data']['exp_features'],
            radius=config['features']['morgan']['radius'],
            nBits=config['features']['morgan']['nBits']
        )

        print("\n=== All features precomputed! ===")


if __name__ == '__main__':
    main()
