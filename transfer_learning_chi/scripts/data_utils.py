"""
Data loading and splitting utilities for transfer learning.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Tuple, List, Dict
from sklearn.model_selection import KFold


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to canonical form using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES string, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def load_and_canonicalize(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and add canonical SMILES column.

    Args:
        csv_path: Path to CSV file with columns: SMILES, temp, chi

    Returns:
        DataFrame with added 'cano_smiles' column
    """
    df = pd.read_csv(csv_path)

    # Canonicalize SMILES
    df['cano_smiles'] = df['SMILES'].apply(canonicalize_smiles)

    # Remove invalid SMILES
    n_invalid = df['cano_smiles'].isna().sum()
    if n_invalid > 0:
        print(f"Warning: Removed {n_invalid} rows with invalid SMILES")
        df = df.dropna(subset=['cano_smiles'])

    return df


def split_dft_data(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DFT dataset into train/val/test based on unique polymers (80/10/10).

    Args:
        df: DataFrame with 'cano_smiles' column
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    # Get unique polymers
    unique_polymers = df['cano_smiles'].unique()
    n_polymers = len(unique_polymers)

    # Shuffle polymers
    rng = np.random.RandomState(seed)
    shuffled_polymers = rng.permutation(unique_polymers)

    # Split polymers 80/10/10
    n_train = int(0.8 * n_polymers)
    n_val = int(0.1 * n_polymers)

    train_polymers = shuffled_polymers[:n_train]
    val_polymers = shuffled_polymers[n_train:n_train + n_val]
    test_polymers = shuffled_polymers[n_train + n_val:]

    # Assign rows to splits
    train_df = df[df['cano_smiles'].isin(train_polymers)].copy()
    val_df = df[df['cano_smiles'].isin(val_polymers)].copy()
    test_df = df[df['cano_smiles'].isin(test_polymers)].copy()

    print(f"DFT split: {len(train_polymers)} train polymers ({len(train_df)} samples), "
          f"{len(val_polymers)} val polymers ({len(val_df)} samples), "
          f"{len(test_polymers)} test polymers ({len(test_df)} samples)")

    return train_df, val_df, test_df


def create_5fold_polymer_split(df: pd.DataFrame, seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create 5-fold cross-validation splits based on unique polymers.

    Args:
        df: DataFrame with 'cano_smiles' column
        seed: Random seed for fold assignment

    Returns:
        List of (train_df, val_df) tuples, one for each fold
    """
    # Get unique polymers
    unique_polymers = df['cano_smiles'].unique()
    n_polymers = len(unique_polymers)

    # Create KFold splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in kf.split(unique_polymers):
        train_polymers = unique_polymers[train_idx]
        val_polymers = unique_polymers[val_idx]

        train_df = df[df['cano_smiles'].isin(train_polymers)].copy()
        val_df = df[df['cano_smiles'].isin(val_polymers)].copy()

        folds.append((train_df, val_df))

    print(f"Experimental 5-fold split created with seed {seed}: "
          f"{n_polymers} total polymers")

    return folds


def get_dft_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split DFT data using configuration.

    Args:
        config: Configuration dictionary

    Returns:
        train_df, val_df, test_df
    """
    dft_path = config['data']['dft_csv']
    seed = config['data']['dft_split_seed']

    print(f"Loading DFT data from {dft_path}")
    df = load_and_canonicalize(dft_path)

    return split_dft_data(df, seed)


def get_exp_data(config: Dict) -> pd.DataFrame:
    """
    Load experimental data using configuration.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with experimental data
    """
    exp_path = config['data']['exp_csv']

    print(f"Loading experimental data from {exp_path}")
    df = load_and_canonicalize(exp_path)

    print(f"Experimental data: {len(df['cano_smiles'].unique())} unique polymers, "
          f"{len(df)} total samples")

    return df
