"""
Pre-calculate Morgan fingerprints and RDKit descriptors for faster training.

This script computes features once and saves them to disk, so that hyperparameter
optimization and training can load them directly instead of recomputing every time.
"""

import yaml
import os
import pickle
from typing import Dict
import numpy as np

from data_utils import load_and_canonicalize
from features import compute_morgan_fingerprint, compute_rdkit_descriptors, get_all_descriptor_names


def load_config(config_path: str = '../config.yaml'):
    """Load configuration from YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_and_save_features(config: Dict):
    """
    Calculate Morgan fingerprints and RDKit descriptors for all molecules
    in both DFT and experimental datasets, and save to disk.

    Args:
        config: Configuration dictionary
    """
    print("=" * 60)
    print("Pre-calculating Features")
    print("=" * 60)

    # Get feature parameters
    fp_radius = config['features']['morgan_fp']['radius']
    fp_nBits = config['features']['morgan_fp']['nBits']
    descriptor_names = get_all_descriptor_names()

    # Adjust paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Load datasets
    dft_path = os.path.join(project_dir, config['data']['dft_csv'])
    exp_path = os.path.join(project_dir, config['data']['exp_csv'])

    print(f"\nLoading DFT data from {dft_path}")
    dft_df = load_and_canonicalize(dft_path)

    print(f"Loading experimental data from {exp_path}")
    exp_df = load_and_canonicalize(exp_path)

    # Get all unique SMILES from both datasets
    all_smiles = set(dft_df['cano_smiles'].unique()) | set(exp_df['cano_smiles'].unique())
    print(f"\nTotal unique molecules: {len(all_smiles)}")

    # Calculate Morgan fingerprints
    print("\nCalculating Morgan fingerprints...")
    fp_dict = {}
    for i, smiles in enumerate(all_smiles):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(all_smiles)}")
        fp = compute_morgan_fingerprint(smiles, fp_radius, fp_nBits)
        fp_dict[smiles] = fp

    # Calculate RDKit descriptors
    print("\nCalculating RDKit descriptors...")
    desc_dict = {}
    for i, smiles in enumerate(all_smiles):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(all_smiles)}")
        descriptors = compute_rdkit_descriptors(smiles, descriptor_names)
        desc_dict[smiles] = descriptors

    # Save features
    data_dir = os.path.join(project_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    fp_file = os.path.join(data_dir, 'features_fp.pkl')
    desc_file = os.path.join(data_dir, 'features_desc.pkl')

    print(f"\nSaving Morgan fingerprints to {fp_file}")
    with open(fp_file, 'wb') as f:
        pickle.dump(fp_dict, f)

    print(f"Saving RDKit descriptors to {desc_file}")
    with open(desc_file, 'wb') as f:
        pickle.dump(desc_dict, f)

    # Save metadata
    metadata = {
        'fp_radius': fp_radius,
        'fp_nBits': fp_nBits,
        'descriptor_names': descriptor_names,
        'n_molecules': len(all_smiles)
    }

    metadata_file = os.path.join(data_dir, 'features_metadata.pkl')
    print(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print("\n" + "=" * 60)
    print("Feature calculation complete!")
    print("=" * 60)
    print(f"Fingerprints: {fp_file}")
    print(f"Descriptors: {desc_file}")
    print(f"Metadata: {metadata_file}")
    print("=" * 60)


if __name__ == '__main__':
    config = load_config()
    calculate_and_save_features(config)
