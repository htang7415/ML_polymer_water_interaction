"""
Training utilities for polymer-water χ prediction.

Handles:
- Training loop with physical loss χ(T) = A/T + B
- MC dropout evaluation for uncertainty quantification
- Metrics computation (R², MAE, RMSE)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import copy


class ChiLoss(nn.Module):
    """
    Loss function for χ prediction using physical relation χ(T) = A/T + B.
    """

    def __init__(self):
        super(ChiLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        A_pred: torch.Tensor,
        B_pred: torch.Tensor,
        T: torch.Tensor,
        chi_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss on predicted chi.

        Args:
            A_pred: Predicted A values (batch_size,)
            B_pred: Predicted B values (batch_size,)
            T: Temperature in Kelvin (batch_size,)
            chi_true: True chi values (batch_size,)

        Returns:
            MSE loss
        """
        # Compute predicted chi using physical relation
        chi_pred = A_pred / T + B_pred

        # Compute MSE loss
        loss = self.mse(chi_pred, chi_true)

        return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ChiLoss,
    device: torch.device
) -> float:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_X, batch_T, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_T = batch_T.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        output = model(batch_X)  # (batch_size, 2)
        A_pred = output[:, 0]
        B_pred = output[:, 1]

        # Compute loss
        loss = criterion(A_pred, B_pred, batch_T, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ChiLoss,
    device: torch.device
) -> float:
    """
    Evaluate the model (standard evaluation without MC dropout).

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_X, batch_T, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_T = batch_T.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            output = model(batch_X)
            A_pred = output[:, 0]
            B_pred = output[:, 1]

            # Compute loss
            loss = criterion(A_pred, B_pred, batch_T, batch_y)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_mc_dropout(
    model: nn.Module,
    X: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    n_samples: int = 50,
    batch_size: int = 256,
    device: torch.device = torch.device('cpu')
) -> Dict[str, np.ndarray]:
    """
    Evaluate with MC dropout for uncertainty quantification.

    Keeps dropout ON at evaluation time and runs multiple forward passes.

    Args:
        model: PyTorch model
        X: Input features (N, D)
        T: Temperature (N,)
        y: True chi values (N,)
        n_samples: Number of MC dropout samples
        batch_size: Batch size for evaluation
        device: Device to use

    Returns:
        Dictionary containing:
            - 'chi_mean': Mean predicted chi (N,)
            - 'chi_std': Std of predicted chi (N,)
            - 'A_mean': Mean predicted A (N,)
            - 'A_std': Std of predicted A (N,)
            - 'B_mean': Mean predicted B (N,)
            - 'B_std': Std of predicted B (N,)
            - 'y_true': True chi values (N,)
            - 'r2': R² score
            - 'mae': Mean absolute error
            - 'rmse': Root mean squared error
    """
    model.train()  # Keep dropout ON

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)

    # Storage for predictions
    all_chi_preds = []
    all_A_preds = []
    all_B_preds = []

    # Run multiple forward passes with dropout
    with torch.no_grad():
        for _ in range(n_samples):
            chi_preds_sample = []
            A_preds_sample = []
            B_preds_sample = []

            # Process in batches
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_T = T_tensor[i:i+batch_size]

                # Forward pass
                output = model(batch_X)  # (batch_size, 2)
                A_pred = output[:, 0].cpu().numpy()
                B_pred = output[:, 1].cpu().numpy()

                # Compute chi
                chi_pred = A_pred / batch_T.cpu().numpy() + B_pred

                chi_preds_sample.append(chi_pred)
                A_preds_sample.append(A_pred)
                B_preds_sample.append(B_pred)

            all_chi_preds.append(np.concatenate(chi_preds_sample))
            all_A_preds.append(np.concatenate(A_preds_sample))
            all_B_preds.append(np.concatenate(B_preds_sample))

    # Stack predictions: (n_samples, N)
    all_chi_preds = np.array(all_chi_preds)
    all_A_preds = np.array(all_A_preds)
    all_B_preds = np.array(all_B_preds)

    # Compute mean and std
    chi_mean = np.mean(all_chi_preds, axis=0)
    chi_std = np.std(all_chi_preds, axis=0)
    A_mean = np.mean(all_A_preds, axis=0)
    A_std = np.std(all_A_preds, axis=0)
    B_mean = np.mean(all_B_preds, axis=0)
    B_std = np.std(all_B_preds, axis=0)

    # Compute metrics using mean predictions
    r2 = r2_score(y, chi_mean)
    mae = mean_absolute_error(y, chi_mean)
    rmse = np.sqrt(mean_squared_error(y, chi_mean))

    return {
        'chi_mean': chi_mean,
        'chi_std': chi_std,
        'A_mean': A_mean,
        'A_std': A_std,
        'B_mean': B_mean,
        'B_std': B_std,
        'y_true': y,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ChiLoss,
    device: torch.device,
    n_epochs: int,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the model for a fixed number of epochs.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        n_epochs: Number of epochs to train
        verbose: Whether to print progress

    Returns:
        Dictionary containing training history:
            - 'train_loss': List of training losses
            - 'val_loss': List of validation losses
    """
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return history


def create_dataloader(
    X: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader from numpy arrays.

    Args:
        X: Input features (N, D)
        T: Temperature (N,)
        y: Target chi values (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader
    """
    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)
    y_tensor = torch.FloatTensor(y)

    dataset = TensorDataset(X_tensor, T_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    return device


if __name__ == '__main__':
    # Test training utilities
    from model import create_model

    print("Testing training utilities...")

    # Create synthetic data
    n_samples = 1000
    input_dim = 50

    X_train = np.random.randn(n_samples, input_dim).astype(np.float32)
    T_train = np.random.uniform(273, 373, n_samples).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)

    X_val = np.random.randn(200, input_dim).astype(np.float32)
    T_val = np.random.uniform(273, 373, 200).astype(np.float32)
    y_val = np.random.randn(200).astype(np.float32)

    # Create dataloaders
    train_loader = create_dataloader(X_train, T_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val, T_val, y_val, batch_size=64, shuffle=False)

    # Create model
    device = get_device()
    model = create_model(
        input_dim=input_dim,
        hidden_dim=32,
        n_layers=2,
        dropout_rate=0.1,
        device=device
    )

    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = ChiLoss()

    # Train for a few epochs
    print("\n=== Training ===")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=5,
        verbose=True
    )

    # Test MC dropout evaluation
    print("\n=== Testing MC Dropout ===")
    results = evaluate_mc_dropout(
        model=model,
        X=X_val,
        T=T_val,
        y=y_val,
        n_samples=10,
        device=device
    )

    print(f"R²: {results['r2']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Mean uncertainty (std): {np.mean(results['chi_std']):.4f}")

    print("\nTraining utilities test completed successfully!")
