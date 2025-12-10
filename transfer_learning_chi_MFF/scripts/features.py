"""
Feature engineering for polymer-water chi parameter prediction.
"""

import numpy as np
import pandas as pd
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Tuple, Dict, List, Optional


def compute_morgan_fingerprint(smiles: str, radius: int = 2, nBits: int = 1024) -> np.ndarray:
    """
    Compute Morgan fingerprint for a SMILES string.

    Args:
        smiles: Canonical SMILES string
        radius: Fingerprint radius
        nBits: Number of bits in fingerprint

    Returns:
        Binary fingerprint as numpy array
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp)
    except:
        return np.zeros(nBits)


def compute_morgan_fingerprint_frequency(smiles: str, radius: int = 2) -> Dict[int, int]:
    """
    Compute Morgan fingerprint with frequency counts as dictionary.

    Args:
        smiles: Canonical SMILES string
        radius: Fingerprint radius

    Returns:
        Dictionary mapping hash code -> frequency count
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        # Get frequency-based fingerprint
        fp_freq = AllChem.GetMorganFingerprint(mol, radius)
        return fp_freq.GetNonzeroElements()
    except:
        return {}


def get_all_descriptor_names() -> List[str]:
    """
    Get list of all valid RDKit descriptor names.

    Returns:
        List of descriptor names
    """
    # Get all descriptor calculation functions
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    return descriptor_names


def compute_rdkit_descriptors(smiles: str, descriptor_names: List[str]) -> Dict[str, float]:
    """
    Compute RDKit descriptors for a SMILES string.

    Args:
        smiles: Canonical SMILES string
        descriptor_names: List of descriptor names to compute

    Returns:
        Dictionary of descriptor name -> value
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: np.nan for name in descriptor_names}

    desc_dict = {}
    for name in descriptor_names:
        try:
            calc_func = getattr(Descriptors, name)
            value = calc_func(mol)
            # Check for invalid values
            if np.isnan(value) or np.isinf(value):
                desc_dict[name] = np.nan
            else:
                desc_dict[name] = value
        except:
            desc_dict[name] = np.nan

    return desc_dict


class FeatureBuilder:
    """
    Builds features for polymer-water chi prediction with different feature modes.
    """

    def __init__(self, config: Dict):
        """
        Initialize feature builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fp_radius = config['features']['morgan_fp']['radius']
        self.fp_nBits = config['features']['morgan_fp']['nBits']

        # Descriptor settings
        self.descriptor_names = get_all_descriptor_names()

        # Statistics for descriptor scaling (fitted on DFT train)
        self.valid_descriptors: Optional[List[str]] = None
        self.desc_mean: Optional[np.ndarray] = None
        self.desc_std: Optional[np.ndarray] = None

        # Pre-computed features (loaded from disk if available)
        self.precomputed_fp: Optional[Dict] = None
        self.precomputed_desc: Optional[Dict] = None
        self.precomputed_fpf: Optional[Dict] = None  # Pre-computed frequency fingerprints
        self.fpf_n_features: Optional[int] = None    # Number of fpf features (variable)
        self._load_precomputed_features()

    def _load_precomputed_features(self):
        """
        Load pre-computed features from disk if available.
        This significantly speeds up feature building by avoiding redundant calculations.
        """
        # Determine Data directory path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(project_dir, 'Data')

        fp_file = os.path.join(data_dir, 'features_fp.pkl')
        desc_file = os.path.join(data_dir, 'features_desc.pkl')

        # Load fingerprints if available
        if os.path.exists(fp_file):
            try:
                with open(fp_file, 'rb') as f:
                    self.precomputed_fp = pickle.load(f)
                print(f"Loaded pre-computed Morgan fingerprints for {len(self.precomputed_fp)} molecules")
            except Exception as e:
                print(f"Warning: Failed to load pre-computed fingerprints: {e}")
                self.precomputed_fp = None

        # Load descriptors if available
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'rb') as f:
                    self.precomputed_desc = pickle.load(f)
                print(f"Loaded pre-computed RDKit descriptors for {len(self.precomputed_desc)} molecules")
            except Exception as e:
                print(f"Warning: Failed to load pre-computed descriptors: {e}")
                self.precomputed_desc = None

        # Load frequency fingerprints if available
        fpf_file = os.path.join(data_dir, 'features_fpf.pkl')
        fpf_metadata_file = os.path.join(data_dir, 'features_fpf_metadata.pkl')
        if os.path.exists(fpf_file) and os.path.exists(fpf_metadata_file):
            try:
                with open(fpf_file, 'rb') as f:
                    self.precomputed_fpf = pickle.load(f)
                with open(fpf_metadata_file, 'rb') as f:
                    fpf_metadata = pickle.load(f)
                self.fpf_n_features = fpf_metadata['n_features']
                print(f"Loaded pre-computed frequency fingerprints: {len(self.precomputed_fpf)} molecules, {self.fpf_n_features} features")
            except Exception as e:
                print(f"Warning: Failed to load pre-computed frequency fingerprints: {e}")
                self.precomputed_fpf = None
                self.fpf_n_features = None

    def fit_descriptor_scaler(self, train_df: pd.DataFrame):
        """
        Fit descriptor scaler on DFT training data.
        - Remove descriptors that are NaN/Inf or constant
        - Compute mean and std for valid descriptors

        Args:
            train_df: Training DataFrame with 'cano_smiles' column
        """
        print("Fitting descriptor scaler on DFT training data...")

        # Compute descriptors for all training molecules
        unique_smiles = train_df['cano_smiles'].unique()
        desc_list = []

        for smiles in unique_smiles:
            # Use pre-computed descriptors if available
            if self.precomputed_desc is not None and smiles in self.precomputed_desc:
                desc_dict = self.precomputed_desc[smiles]
            else:
                desc_dict = compute_rdkit_descriptors(smiles, self.descriptor_names)
            desc_list.append(desc_dict)

        # Convert to DataFrame
        desc_df = pd.DataFrame(desc_list)

        # Remove columns with any NaN/Inf
        valid_cols = []
        for col in desc_df.columns:
            if not desc_df[col].isna().any() and not np.isinf(desc_df[col]).any():
                valid_cols.append(col)

        desc_df = desc_df[valid_cols]

        # Remove constant columns (std == 0)
        valid_cols = []
        for col in desc_df.columns:
            if desc_df[col].std() > 1e-10:
                valid_cols.append(col)

        desc_df = desc_df[valid_cols]

        # Store valid descriptor names and statistics
        self.valid_descriptors = list(desc_df.columns)
        self.desc_mean = desc_df.mean().values
        self.desc_std = desc_df.std().values

        print(f"Valid descriptors: {len(self.valid_descriptors)} out of {len(self.descriptor_names)}")

    def compute_fingerprints(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute Morgan fingerprints for all molecules in DataFrame.
        Uses pre-computed features if available.

        Args:
            df: DataFrame with 'cano_smiles' column

        Returns:
            Array of shape (n_samples, nBits)
        """
        fps = []
        for smiles in df['cano_smiles']:
            # Use pre-computed fingerprint if available
            if self.precomputed_fp is not None and smiles in self.precomputed_fp:
                fp = self.precomputed_fp[smiles]
            else:
                fp = compute_morgan_fingerprint(smiles, self.fp_radius, self.fp_nBits)
            fps.append(fp)
        return np.array(fps)

    def compute_fingerprints_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute frequency-based Morgan fingerprints (NO scaling/normalization).

        Args:
            df: DataFrame with 'cano_smiles' column

        Returns:
            Array of shape (n_samples, n_fpf_features) with raw frequency counts
        """
        if self.precomputed_fpf is None:
            raise ValueError("Pre-computed fpf not loaded. Run MF_descriptors.sh first.")

        fpfs = []
        for smiles in df['cano_smiles']:
            if smiles in self.precomputed_fpf:
                fpf = self.precomputed_fpf[smiles]
            else:
                # For new molecules not in pre-computed, create zero array
                fpf = np.zeros(self.fpf_n_features, dtype=np.float32)
            fpfs.append(fpf)

        return np.array(fpfs)

    def compute_scaled_descriptors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute and scale RDKit descriptors for all molecules in DataFrame.

        Args:
            df: DataFrame with 'cano_smiles' column

        Returns:
            Array of shape (n_samples, n_valid_descriptors)
        """
        if self.valid_descriptors is None:
            raise ValueError("Descriptor scaler not fitted. Call fit_descriptor_scaler first.")

        # Compute descriptors for unique molecules
        unique_smiles = df['cano_smiles'].unique()
        smiles_to_desc = {}
        n_sanitized = 0

        for smiles in unique_smiles:
            # Use pre-computed descriptors if available
            if self.precomputed_desc is not None and smiles in self.precomputed_desc:
                desc_dict = self.precomputed_desc[smiles]
            else:
                desc_dict = compute_rdkit_descriptors(smiles, self.valid_descriptors)

            desc_vec = np.array([desc_dict[name] for name in self.valid_descriptors])

            # Sanitize NaN/Inf values that might appear for new molecule structures
            if np.any(np.isnan(desc_vec)) or np.any(np.isinf(desc_vec)):
                n_sanitized += 1
                desc_vec = np.nan_to_num(desc_vec, nan=0.0, posinf=0.0, neginf=0.0)

            smiles_to_desc[smiles] = desc_vec

        if n_sanitized > 0:
            print(f"Warning: Sanitized NaN/Inf descriptors for {n_sanitized} molecules (replaced with 0)")

        # Get descriptors for all samples
        desc_matrix = np.array([smiles_to_desc[smiles] for smiles in df['cano_smiles']])

        # Scale using DFT train statistics
        desc_scaled = (desc_matrix - self.desc_mean) / self.desc_std

        # Final sanitization check after scaling
        if np.any(np.isnan(desc_scaled)) or np.any(np.isinf(desc_scaled)):
            print("Warning: NaN/Inf found after scaling, applying final sanitization")
            desc_scaled = np.nan_to_num(desc_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return desc_scaled

    def build_features(self, df: pd.DataFrame, feature_mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix and target vector for a DataFrame.

        Args:
            df: DataFrame with columns: cano_smiles, temp, chi
            feature_mode: One of "fp_T", "desc_T", "fp_desc_T", "fpf_T", "fpf_desc_T"

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        # Temperature feature (not scaled)
        temp = df['temp'].values.reshape(-1, 1)

        # Target
        y = df['chi'].values

        # Build features based on mode
        if feature_mode == "fp_T":
            fps = self.compute_fingerprints(df)
            X = np.concatenate([fps, temp], axis=1)

        elif feature_mode == "desc_T":
            desc = self.compute_scaled_descriptors(df)
            X = np.concatenate([desc, temp], axis=1)

        elif feature_mode == "fp_desc_T":
            fps = self.compute_fingerprints(df)
            desc = self.compute_scaled_descriptors(df)
            X = np.concatenate([fps, desc, temp], axis=1)

        elif feature_mode == "fpf_T":
            fpf = self.compute_fingerprints_frequency(df)
            X = np.concatenate([fpf, temp], axis=1)

        elif feature_mode == "fpf_desc_T":
            fpf = self.compute_fingerprints_frequency(df)
            desc = self.compute_scaled_descriptors(df)
            X = np.concatenate([fpf, desc, temp], axis=1)

        else:
            raise ValueError(f"Invalid feature_mode: {feature_mode}")

        print(f"Built features with mode '{feature_mode}': X shape = {X.shape}")

        return X, y

    def get_input_dim(self, feature_mode: str) -> int:
        """
        Get input dimension for a given feature mode.

        Args:
            feature_mode: One of "fp_T", "desc_T", "fp_desc_T", "fpf_T", "fpf_desc_T"

        Returns:
            Input dimension
        """
        if feature_mode == "fp_T":
            return self.fp_nBits + 1

        elif feature_mode == "desc_T":
            if self.valid_descriptors is None:
                raise ValueError("Descriptor scaler not fitted.")
            return len(self.valid_descriptors) + 1

        elif feature_mode == "fp_desc_T":
            if self.valid_descriptors is None:
                raise ValueError("Descriptor scaler not fitted.")
            return self.fp_nBits + len(self.valid_descriptors) + 1

        elif feature_mode == "fpf_T":
            if self.fpf_n_features is None:
                raise ValueError("FPF features not loaded. Run MF_descriptors.sh first.")
            return self.fpf_n_features + 1

        elif feature_mode == "fpf_desc_T":
            if self.fpf_n_features is None:
                raise ValueError("FPF features not loaded. Run MF_descriptors.sh first.")
            if self.valid_descriptors is None:
                raise ValueError("Descriptor scaler not fitted.")
            return self.fpf_n_features + len(self.valid_descriptors) + 1

        else:
            raise ValueError(f"Invalid feature_mode: {feature_mode}")
