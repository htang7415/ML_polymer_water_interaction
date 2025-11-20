"""
Normalizers for chi values to improve training stability.

This module implements Approach B for normalization:
- A and B parameters remain physical (not normalized)
- Normalization is applied only during loss calculation
- Predictions are denormalized for evaluation and logging
"""

import numpy as np
import torch
from typing import Union, Optional


class ChiNormalizer:
    """Normalize chi values using standardization (z-score)."""

    def __init__(self, method: str = 'standardize'):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('standardize' for z-score)
        """
        self.method = method
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, chi_values: Union[np.ndarray, list, torch.Tensor]):
        """
        Fit normalizer on training chi values.

        Args:
            chi_values: Chi values from training set
        """
        if isinstance(chi_values, torch.Tensor):
            chi_values = chi_values.cpu().numpy()
        elif isinstance(chi_values, list):
            chi_values = np.array(chi_values)

        if self.method == 'standardize':
            self.mean = float(np.mean(chi_values))
            self.std = float(np.std(chi_values))
            if self.std < 1e-8:
                self.std = 1.0  # Avoid division by zero
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self.fitted = True

    def transform(self, chi_values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize chi values.

        Args:
            chi_values: Chi values to normalize

        Returns:
            Normalized chi values (same type as input)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")

        is_tensor = isinstance(chi_values, torch.Tensor)

        if self.method == 'standardize':
            if is_tensor:
                # Keep torch ops to preserve gradients
                mean = torch.tensor(self.mean, device=chi_values.device, dtype=chi_values.dtype)
                std = torch.tensor(self.std, device=chi_values.device, dtype=chi_values.dtype)
                return (chi_values - mean) / std
            else:
                chi_values_np = np.asarray(chi_values)
                return (chi_values_np - self.mean) / self.std
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def inverse_transform(self, chi_normalized: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize chi values back to original scale.

        Args:
            chi_normalized: Normalized chi values

        Returns:
            Original-scale chi values (same type as input)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform. Call fit() first.")

        is_tensor = isinstance(chi_normalized, torch.Tensor)

        if self.method == 'standardize':
            if is_tensor:
                mean = torch.tensor(self.mean, device=chi_normalized.device, dtype=chi_normalized.dtype)
                std = torch.tensor(self.std, device=chi_normalized.device, dtype=chi_normalized.dtype)
                return chi_normalized * std + mean
            else:
                chi_normalized_np = np.asarray(chi_normalized)
                return chi_normalized_np * self.std + self.mean
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def __repr__(self):
        if self.fitted:
            return f"ChiNormalizer(method='{self.method}', mean={self.mean:.4f}, std={self.std:.4f})"
        else:
            return f"ChiNormalizer(method='{self.method}', not fitted)"
