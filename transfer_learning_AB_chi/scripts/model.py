"""
PyTorch model for polymer-water χ prediction.

The model outputs two scalars A and B per sample.
Physical relation: χ(T) = A / T + B
"""

import torch
import torch.nn as nn
from typing import Optional


class ChiModel(nn.Module):
    """
    MLP model that predicts A and B parameters for χ(T) = A/T + B.

    Architecture:
        Input → [Linear → ReLU → Dropout] × n_layers → Linear(2)

    The output has 2 dimensions:
        - output[:, 0] = A
        - output[:, 1] = B
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Width of hidden layers
            n_layers: Number of hidden layers
            dropout_rate: Dropout probability
        """
        super(ChiModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer: 2 outputs (A and B)
        layers.append(nn.Linear(hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Output of shape (batch_size, 2) where:
                - output[:, 0] = A
                - output[:, 1] = B
        """
        return self.network(x)

    def predict_chi(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Predict χ using the physical relation χ(T) = A/T + B.

        Args:
            x: Input features of shape (batch_size, input_dim)
            T: Temperature in Kelvin of shape (batch_size,) or (batch_size, 1)

        Returns:
            Predicted χ of shape (batch_size,)
        """
        output = self.forward(x)  # (batch_size, 2)
        A = output[:, 0]  # (batch_size,)
        B = output[:, 1]  # (batch_size,)

        # Ensure T is 1D
        if T.dim() == 2:
            T = T.squeeze(1)

        chi = A / T + B
        return chi

    def freeze_lower_layers(self):
        """
        Freeze all layers except the last (output) layer.

        Used for transfer learning with freeze_strategy='freeze_lower'.
        """
        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.network[-1].parameters():
            param.requires_grad = True

        print("Froze all layers except the output layer")

    def unfreeze_all(self):
        """
        Unfreeze all layers.

        Used for transfer learning with freeze_strategy='all_trainable'.
        """
        for param in self.parameters():
            param.requires_grad = True

        print("All layers are trainable")

    def get_num_trainable_params(self) -> int:
        """
        Get the number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """
        Get the total number of parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


def create_model(
    input_dim: int,
    hidden_dim: int = 128,
    n_layers: int = 3,
    dropout_rate: float = 0.2,
    device: Optional[torch.device] = None
) -> ChiModel:
    """
    Factory function to create a ChiModel.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Width of hidden layers
        n_layers: Number of hidden layers
        dropout_rate: Dropout probability
        device: Device to place the model on (if None, uses CPU)

    Returns:
        ChiModel instance
    """
    if device is None:
        device = torch.device('cpu')

    model = ChiModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )

    model = model.to(device)

    print(f"Created model with {model.get_num_total_params():,} total parameters")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

    return model


if __name__ == '__main__':
    # Test the model
    print("Testing ChiModel...")

    # Create a small model
    input_dim = 100
    model = create_model(
        input_dim=input_dim,
        hidden_dim=64,
        n_layers=2,
        dropout_rate=0.2
    )

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    T = torch.rand(batch_size) * 100 + 273.15  # Random temperatures

    # Forward pass
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"A shape: {output[:, 0].shape}")
    print(f"B shape: {output[:, 1].shape}")

    # Predict chi
    chi = model.predict_chi(x, T)
    print(f"Chi shape: {chi.shape}")

    # Test freezing
    print("\n=== Testing layer freezing ===")
    print(f"Initial trainable params: {model.get_num_trainable_params():,}")

    model.freeze_lower_layers()
    print(f"After freezing lower layers: {model.get_num_trainable_params():,}")

    model.unfreeze_all()
    print(f"After unfreezing all: {model.get_num_trainable_params():,}")

    print("\nModel test completed successfully!")
