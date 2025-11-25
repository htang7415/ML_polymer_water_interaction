"""
Data loading and splitting utilities.

Handles:
- Loading precomputed feature CSVs
- Creating fixed DFT train/val/test splits based on polymer identity
- Creating 5-fold CV splits for experimental data based on polymer identity
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def load_features(csv_path: str) -> pd.DataFrame:
    """
    Load precomputed features from CSV.

    Args:
        csv_path: Path to feature CSV

    Returns:
        DataFrame with features
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def get_dft_splits(
    dft_df: pd.DataFrame,
    global_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create fixed train/val/test splits for DFT data based on polymer identity.

    Splits are performed at the polymer level (unique cano_smiles) to prevent
    data leakage between splits.

    Args:
        dft_df: DFT features DataFrame
        global_seed: Random seed for reproducibility
        train_frac: Fraction of polymers for training
        val_frac: Fraction of polymers for validation
        test_frac: Fraction of polymers for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    # Get unique polymers
    unique_polymers = dft_df['cano_smiles'].unique()
    n_polymers = len(unique_polymers)

    print(f"DFT data: {len(dft_df)} samples from {n_polymers} unique polymers")

    # Split polymers into train/temp
    train_polymers, temp_polymers = train_test_split(
        unique_polymers,
        test_size=(1.0 - train_frac),
        random_state=global_seed
    )

    # Split temp into val/test
    val_size = val_frac / (val_frac + test_frac)
    val_polymers, test_polymers = train_test_split(
        temp_polymers,
        test_size=(1.0 - val_size),
        random_state=global_seed
    )

    # Create splits based on polymer membership
    train_df = dft_df[dft_df['cano_smiles'].isin(train_polymers)].copy()
    val_df = dft_df[dft_df['cano_smiles'].isin(val_polymers)].copy()
    test_df = dft_df[dft_df['cano_smiles'].isin(test_polymers)].copy()

    print(f"DFT train: {len(train_df)} samples from {len(train_polymers)} polymers")
    print(f"DFT val:   {len(val_df)} samples from {len(val_polymers)} polymers")
    print(f"DFT test:  {len(test_df)} samples from {len(test_polymers)} polymers")

    return train_df, val_df, test_df


def get_experiment_folds(
    exp_df: pd.DataFrame,
    split_seed: int,
    n_folds: int = 5
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create k-fold CV splits for experimental data based on polymer identity.

    Args:
        exp_df: Experimental features DataFrame
        split_seed: Random seed for this particular split
        n_folds: Number of folds

    Returns:
        List of dictionaries, each containing 'train' and 'val' DataFrames
    """
    # Get unique polymers
    unique_polymers = exp_df['cano_smiles'].unique()
    n_polymers = len(unique_polymers)

    print(f"Experimental data: {len(exp_df)} samples from {n_polymers} unique polymers")

    # Shuffle polymers with the given seed
    rng = np.random.RandomState(split_seed)
    shuffled_polymers = unique_polymers.copy()
    rng.shuffle(shuffled_polymers)

    # Assign polymers to folds
    fold_assignments = np.array_split(shuffled_polymers, n_folds)

    # Create folds
    folds = []
    for fold_idx in range(n_folds):
        val_polymers = fold_assignments[fold_idx]
        train_polymers = np.concatenate([
            fold_assignments[i] for i in range(n_folds) if i != fold_idx
        ])

        train_df = exp_df[exp_df['cano_smiles'].isin(train_polymers)].copy()
        val_df = exp_df[exp_df['cano_smiles'].isin(val_polymers)].copy()

        folds.append({
            'train': train_df,
            'val': val_df,
            'train_polymers': len(train_polymers),
            'val_polymers': len(val_polymers)
        })

        print(f"Fold {fold_idx}: train={len(train_df)} samples ({len(train_polymers)} polymers), "
              f"val={len(val_df)} samples ({len(val_polymers)} polymers)")

    return folds


def print_data_summary(dft_df: pd.DataFrame, exp_df: pd.DataFrame):
    """
    Print summary statistics for DFT and experimental datasets.

    Args:
        dft_df: DFT features DataFrame
        exp_df: Experimental features DataFrame
    """
    print("\n=== Dataset Summary ===")

    print("\nDFT Dataset:")
    print(f"  Total samples: {len(dft_df)}")
    print(f"  Unique polymers: {dft_df['cano_smiles'].nunique()}")
    print(f"  Temperature range: {dft_df['temp'].min():.1f} - {dft_df['temp'].max():.1f} K")
    print(f"  Chi range: {dft_df['chi'].min():.4f} - {dft_df['chi'].max():.4f}")
    print(f"  Chi mean ± std: {dft_df['chi'].mean():.4f} ± {dft_df['chi'].std():.4f}")

    print("\nExperimental Dataset:")
    print(f"  Total samples: {len(exp_df)}")
    print(f"  Unique polymers: {exp_df['cano_smiles'].nunique()}")
    print(f"  Temperature range: {exp_df['temp'].min():.1f} - {exp_df['temp'].max():.1f} K")
    print(f"  Chi range: {exp_df['chi'].min():.4f} - {exp_df['chi'].max():.4f}")
    print(f"  Chi mean ± std: {exp_df['chi'].mean():.4f} ± {exp_df['chi'].std():.4f}")

    print()


if __name__ == '__main__':
    # Test data loading and splitting
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    dft_df = load_features(config['data']['dft_features'])
    exp_df = load_features(config['data']['exp_features'])

    # Print summary
    print_data_summary(dft_df, exp_df)

    # Test DFT splits
    print("\n=== Testing DFT Splits ===")
    train_df, val_df, test_df = get_dft_splits(
        dft_df,
        global_seed=config['global']['dft_split_seed'],
        train_frac=config['global']['dft_train_frac'],
        val_frac=config['global']['dft_val_frac'],
        test_frac=config['global']['dft_test_frac']
    )

    # Test experimental folds
    print("\n=== Testing Experimental Folds ===")
    folds = get_experiment_folds(
        exp_df,
        split_seed=42,
        n_folds=config['global']['n_folds']
    )

    print("\nData utilities test completed successfully!")
