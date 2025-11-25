"""
PyTorch MLP model for polymer-water chi parameter prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ChiMLP(nn.Module):
    """
    Multi-layer perceptron for chi parameter regression with dropout.
    """

    def __init__(self, input_dim: int, n_layers: int, hidden_dim: int, dropout_rate: float):
        """
        Initialize MLP.

        Args:
            input_dim: Input feature dimension
            n_layers: Number of hidden layers
            hidden_dim: Width of hidden layers
            dropout_rate: Dropout probability
        """
        super(ChiMLP, self).__init__()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Additional hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, 1)
        """
        h = self.hidden_layers(x)
        out = self.output_layer(h)
        return out


def evaluate_mc_dropout(
    model: ChiMLP,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 50,
    device: str = 'cpu',
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model with MC Dropout to estimate predictive uncertainty.

    Args:
        model: Trained ChiMLP model
        X: Feature matrix (n_samples, n_features)
        y: True targets (n_samples,)
        n_samples: Number of MC dropout samples
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        mu: Mean predictions (n_samples,)
        sigma: Predictive standard deviations (n_samples,)
        metrics: Dictionary with R², MAE, RMSE
    """
    model.train()  # Keep dropout enabled

    X_tensor = torch.FloatTensor(X).to(device)
    n_data = X.shape[0]

    # Collect predictions from multiple forward passes
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for i in range(0, n_data, batch_size):
                batch_X = X_tensor[i:i + batch_size]
                batch_pred = model(batch_X).cpu().numpy().flatten()
                batch_preds.append(batch_pred)
            predictions.append(np.concatenate(batch_preds))

    predictions = np.array(predictions)  # Shape: (n_samples, n_data)

    # Compute mean and std
    mu = predictions.mean(axis=0)
    sigma = predictions.std(axis=0)

    # Compute metrics using mean predictions
    r2 = r2_score(y, mu)
    mae = mean_absolute_error(y, mu)
    rmse = np.sqrt(mean_squared_error(y, mu))

    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }

    return mu, sigma, metrics


def apply_freeze_strategy(model: ChiMLP, freeze_strategy: str):
    """
    Apply freezing strategy to model parameters.

    Args:
        model: ChiMLP model
        freeze_strategy: "all_trainable" or "freeze_lower"
    """
    if freeze_strategy == "all_trainable":
        # All parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

    elif freeze_strategy == "freeze_lower":
        # Freeze all hidden layers, keep only output layer trainable
        for param in model.hidden_layers.parameters():
            param.requires_grad = False
        for param in model.output_layer.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Invalid freeze_strategy: {freeze_strategy}")

    # Count trainable parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Freeze strategy '{freeze_strategy}': {n_trainable}/{n_total} parameters trainable")
