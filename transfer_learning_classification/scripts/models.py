"""
Neural network models for transfer learning.
Includes encoder MLP, regression head, and classification head.
"""

import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """
    MLP encoder for feature extraction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Encoded features (batch_size, output_dim)
        """
        return self.encoder(x)


class RegressionHead(nn.Module):
    """
    Regression head for chi prediction.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Input dimension from encoder
        """
        super(RegressionHead, self).__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoded features (batch_size, input_dim)

        Returns:
            Predictions (batch_size, 1)
        """
        return self.head(x)


class ClassificationHead(nn.Module):
    """
    Binary classification head for water solubility prediction.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Input dimension from encoder
        """
        super(ClassificationHead, self).__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoded features (batch_size, input_dim)

        Returns:
            Logits (batch_size, 1)
        """
        return self.head(x)


class RegressionModel(nn.Module):
    """
    Full regression model (encoder + regression head).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function
        """
        super(RegressionModel, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dims, dropout_rate, activation)
        self.regression_head = RegressionHead(self.encoder.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Predictions (batch_size, 1)
        """
        features = self.encoder(x)
        return self.regression_head(features)

    def get_encoder(self) -> Encoder:
        """
        Get the encoder module.

        Returns:
            Encoder module
        """
        return self.encoder


class ClassificationModel(nn.Module):
    """
    Full classification model (encoder + classification head).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        pretrained_encoder: Encoder = None,
        n_freeze_layers: int = 0
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function
            pretrained_encoder: Pretrained encoder to use (if None, create new)
            n_freeze_layers: Number of encoder layers to freeze
        """
        super(ClassificationModel, self).__init__()

        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = Encoder(input_dim, hidden_dims, dropout_rate, activation)

        # Freeze encoder layers if specified
        if n_freeze_layers > 0:
            self._freeze_encoder_layers(n_freeze_layers)

        self.classification_head = ClassificationHead(self.encoder.output_dim)

    def _freeze_encoder_layers(self, n_freeze: int):
        """
        Freeze the first n_freeze layers of the encoder.

        Args:
            n_freeze: Number of layers to freeze
        """
        # Get encoder sequential layers
        encoder_layers = list(self.encoder.encoder.children())

        # Each "layer" consists of Linear + Activation + Dropout (3 modules)
        modules_per_layer = 3
        n_freeze_modules = n_freeze * modules_per_layer

        for i, module in enumerate(encoder_layers):
            if i < n_freeze_modules:
                for param in module.parameters():
                    param.requires_grad = False

        print(f"Froze {n_freeze} encoder layer(s) ({n_freeze_modules} modules)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Logits (batch_size, 1)
        """
        features = self.encoder(x)
        return self.classification_head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities using sigmoid.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Probabilities (batch_size, 1)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


def load_encoder_weights(encoder: Encoder, regression_model_path: str):
    """
    Load encoder weights from a saved regression model.

    Args:
        encoder: Encoder module to load weights into
        regression_model_path: Path to saved regression model

    Returns:
        Encoder with loaded weights
    """
    checkpoint = torch.load(regression_model_path, map_location='cpu')

    # Extract encoder state dict
    encoder_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('encoder.'):
            # Remove 'encoder.' prefix
            new_key = key[8:]
            encoder_state_dict[new_key] = value

    encoder.load_state_dict(encoder_state_dict)
    print(f"Loaded encoder weights from {regression_model_path}")

    return encoder


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
