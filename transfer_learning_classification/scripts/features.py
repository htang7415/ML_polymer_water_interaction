"""
Feature engineering utilities for polymer water interaction prediction.
Includes SMILES canonicalization, Morgan fingerprints, and RDKit descriptors.
"""

import argparse
import warnings
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


warnings.filterwarnings('ignore')


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to canonical form using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES string, or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        warnings.warn(f"Failed to canonicalize SMILES '{smiles}': {e}")
        return None


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 3,
    n_bits: int = 1024
) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprint for a SMILES string.

    Args:
        smiles: Input SMILES string
        radius: Fingerprint radius
        n_bits: Number of bits in fingerprint

    Returns:
        Binary fingerprint array, or None if computation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception as e:
        warnings.warn(f"Failed to compute fingerprint for '{smiles}': {e}")
        return None


def get_all_rdkit_descriptors() -> List[str]:
    """
    Get list of all available RDKit descriptors.

    Returns:
        List of descriptor names
    """
    # Get all descriptor names from RDKit
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    return descriptor_names


def compute_descriptors(smiles: str, descriptor_names: List[str]) -> Optional[np.ndarray]:
    """
    Compute RDKit descriptors for a SMILES string.

    Args:
        smiles: Input SMILES string
        descriptor_names: List of descriptor names to compute

    Returns:
        Array of descriptor values, or None if computation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Create a mapping of descriptor names to functions
        desc_dict = {name: func for name, func in Descriptors.descList}

        # Compute each descriptor by calling its function directly
        desc_values = []
        for desc_name in descriptor_names:
            if desc_name in desc_dict:
                desc_values.append(desc_dict[desc_name](mol))
            else:
                warnings.warn(f"Descriptor '{desc_name}' not found in RDKit descriptors")
                desc_values.append(np.nan)

        return np.array(desc_values, dtype=np.float32)

    except Exception as e:
        warnings.warn(f"Failed to compute descriptors for '{smiles}': {e}")
        return None


def filter_and_scale_descriptors(
    X_train: np.ndarray,
    descriptor_names: List[str],
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, List[int]]:
    """
    Filter out NaN/Inf/constant descriptors and compute scaling parameters.

    Args:
        X_train: Training descriptor matrix (n_samples, n_descriptors)
        descriptor_names: List of descriptor names
        verbose: Whether to print filtering info

    Returns:
        Tuple of (filtered_X_train, valid_descriptor_names, means, stds, valid_indices)
    """
    n_samples, n_descriptors = X_train.shape
    valid_indices = []
    valid_names = []

    if verbose:
        print(f"\nFiltering descriptors on training set ({n_samples} samples, {n_descriptors} descriptors):")

    for i, name in enumerate(descriptor_names):
        values = X_train[:, i]

        # Check for NaN or Inf
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            if verbose:
                print(f"  Removing {name}: contains NaN or Inf")
            continue

        # Check for constant values
        if np.std(values) < 1e-8:
            if verbose:
                print(f"  Removing {name}: constant values")
            continue

        # Check for extreme values (prevent numerical instability)
        max_abs_value = np.abs(values).max()
        if max_abs_value > 1e10:
            if verbose:
                print(f"  Removing {name}: extreme values (max={max_abs_value:.2e})")
            continue

        valid_indices.append(i)
        valid_names.append(name)

    # Filter to valid descriptors only
    X_train_filtered = X_train[:, valid_indices]

    # Compute mean and std for scaling
    means = np.mean(X_train_filtered, axis=0)
    stds = np.std(X_train_filtered, axis=0)

    # Avoid division by zero (should not happen after filtering, but just in case)
    stds = np.where(stds < 1e-8, 1.0, stds)

    if verbose:
        print(f"  Kept {len(valid_indices)} / {n_descriptors} descriptors")

    return X_train_filtered, valid_names, means, stds, valid_indices


def scale_descriptors(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Scale descriptors using provided mean and std.

    Args:
        X: Descriptor matrix
        means: Mean values from training set
        stds: Std values from training set

    Returns:
        Scaled descriptor matrix
    """
    return (X - means) / stds


def precompute_features(
    df: pd.DataFrame,
    radius: int = 3,
    n_bits: int = 1024,
    descriptor_names: Optional[List[str]] = None,
    valid_descriptor_indices: Optional[List[int]] = None,
    descriptor_means: Optional[np.ndarray] = None,
    descriptor_stds: Optional[np.ndarray] = None,
    is_training: bool = False
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Precompute fingerprints and descriptors for all molecules in DataFrame.

    Args:
        df: Input DataFrame with 'SMILES' column
        radius: Morgan fingerprint radius
        n_bits: Number of fingerprint bits
        descriptor_names: List of descriptor names (if None, use all RDKit descriptors)
        valid_descriptor_indices: Indices of valid descriptors (for non-training sets)
        descriptor_means: Means for scaling (for non-training sets)
        descriptor_stds: Stds for scaling (for non-training sets)
        is_training: Whether this is the training set (determines filtering/scaling)

    Returns:
        Tuple of (processed_df, scaling_params_dict or None)
    """
    print(f"\nPrecomputing features for {len(df)} samples...")

    # Get descriptor names if not provided
    if descriptor_names is None:
        descriptor_names = get_all_rdkit_descriptors()
        print(f"Using all {len(descriptor_names)} RDKit descriptors")

    processed_rows = []

    for idx, row in df.iterrows():
        smiles = row['SMILES']

        # Canonicalize SMILES
        cano_smiles = canonicalize_smiles(smiles)
        if cano_smiles is None:
            warnings.warn(f"Skipping row {idx}: failed to canonicalize SMILES '{smiles}'")
            continue

        # Compute Morgan fingerprint
        fp = compute_morgan_fingerprint(cano_smiles, radius, n_bits)
        if fp is None:
            warnings.warn(f"Skipping row {idx}: failed to compute fingerprint for '{smiles}'")
            continue

        # Compute descriptors
        descriptors = compute_descriptors(cano_smiles, descriptor_names)
        if descriptors is None:
            warnings.warn(f"Skipping row {idx}: failed to compute descriptors for '{smiles}'")
            continue

        # Build row data
        row_data = {
            'SMILES': smiles,
            'cano_smiles': cano_smiles,
        }

        # Add target if present
        if 'chi' in row:
            row_data['chi'] = row['chi']
        if 'water_soluble' in row:
            row_data['water_soluble'] = row['water_soluble']

        # Add fingerprint columns
        for i in range(n_bits):
            row_data[f'mf_{i}'] = fp[i]

        # Add descriptor columns (unscaled for now)
        for i, desc_name in enumerate(descriptor_names):
            row_data[f'desc_{desc_name}'] = descriptors[i]

        processed_rows.append(row_data)

    # Create DataFrame
    processed_df = pd.DataFrame(processed_rows)
    print(f"Successfully processed {len(processed_df)} / {len(df)} samples")

    scaling_params = None

    # If this is training set, filter and compute scaling parameters
    if is_training:
        # Extract descriptor columns
        desc_cols = [f'desc_{name}' for name in descriptor_names]
        X_desc = processed_df[desc_cols].values

        # Filter and compute scaling
        X_desc_filtered, valid_names, means, stds, valid_indices = filter_and_scale_descriptors(
            X_desc, descriptor_names, verbose=True
        )

        # Update descriptor columns in DataFrame
        # First, remove all old descriptor columns
        processed_df = processed_df.drop(columns=desc_cols)

        # Add filtered and scaled descriptors
        for i, name in enumerate(valid_names):
            processed_df[f'desc_{name}'] = X_desc_filtered[:, i]

        scaling_params = {
            'descriptor_names': descriptor_names,
            'valid_names': valid_names,
            'valid_indices': valid_indices,
            'means': means,
            'stds': stds
        }

    else:
        # For non-training sets, apply filtering and scaling using provided parameters
        if valid_descriptor_indices is not None and descriptor_means is not None and descriptor_stds is not None:
            # Extract all descriptor columns
            desc_cols = [f'desc_{name}' for name in descriptor_names]
            X_desc = processed_df[desc_cols].values

            # Filter to valid descriptors
            X_desc_filtered = X_desc[:, valid_descriptor_indices]

            # Scale
            X_desc_scaled = scale_descriptors(X_desc_filtered, descriptor_means, descriptor_stds)

            # Update DataFrame
            processed_df = processed_df.drop(columns=desc_cols)

            # Get valid descriptor names
            valid_names = [descriptor_names[i] for i in valid_descriptor_indices]

            for i, name in enumerate(valid_names):
                processed_df[f'desc_{name}'] = X_desc_scaled[:, i]

    return processed_df, scaling_params


def main():
    """
    Main function for precomputing features.
    Run with: python features.py --precompute
    """
    parser = argparse.ArgumentParser(description='Precompute features for transfer learning')
    parser.add_argument('--precompute', action='store_true', help='Precompute features')
    parser.add_argument('--dft_csv', type=str, default='Data/OMG_DFT_COSMOC_chi.csv',
                        help='Path to DFT chi CSV')
    parser.add_argument('--binary_csv', type=str, default='Data/Binary_solubility.csv',
                        help='Path to binary solubility CSV')
    parser.add_argument('--dft_out', type=str, default='Data/DFT_features.csv',
                        help='Output path for DFT features')
    parser.add_argument('--binary_out', type=str, default='Data/binary_features.csv',
                        help='Output path for binary features')
    parser.add_argument('--radius', type=int, default=3, help='Morgan fingerprint radius')
    parser.add_argument('--n_bits', type=int, default=1024, help='Number of fingerprint bits')

    args = parser.parse_args()

    if args.precompute:
        print("=" * 80)
        print("PRECOMPUTING FEATURES")
        print("=" * 80)

        # Load DFT data
        print("\n1. Processing DFT chi data...")
        dft_df = pd.read_csv(args.dft_csv)
        dft_processed, scaling_params = precompute_features(
            dft_df,
            radius=args.radius,
            n_bits=args.n_bits,
            is_training=True
        )
        dft_processed.to_csv(args.dft_out, index=False)
        print(f"Saved DFT features to {args.dft_out}")

        # Load binary solubility data
        print("\n2. Processing binary solubility data...")
        binary_df = pd.read_csv(args.binary_csv)
        binary_processed, _ = precompute_features(
            binary_df,
            radius=args.radius,
            n_bits=args.n_bits,
            descriptor_names=scaling_params['descriptor_names'],
            valid_descriptor_indices=scaling_params['valid_indices'],
            descriptor_means=scaling_params['means'],
            descriptor_stds=scaling_params['stds'],
            is_training=False
        )
        binary_processed.to_csv(args.binary_out, index=False)
        print(f"Saved binary features to {args.binary_out}")

        print("\n" + "=" * 80)
        print("FEATURE PRECOMPUTATION COMPLETE")
        print("=" * 80)


if __name__ == '__main__':
    main()
