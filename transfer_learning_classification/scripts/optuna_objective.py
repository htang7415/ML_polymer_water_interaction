"""
Optuna hyperparameter optimization for transfer learning.
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import optuna

from data_utils import (
    load_precomputed_features, split_dft_data,
    extract_feature_columns
)
from models import RegressionModel, ClassificationModel
from utils import (
    set_random_seed, get_device,
    compute_regression_metrics, evaluate_mc_dropout,
    train_epoch, evaluate_epoch, evaluate_classification_epoch,
    compute_classification_metrics
)


def objective(trial: optuna.Trial, config: dict) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        config: Base configuration dictionary

    Returns:
        F1 score (to be maximized)
    """
    # Sample hyperparameters
    search_space = config['optuna']['search_space']

    # Feature mode
    feature_mode = trial.suggest_categorical(
        'feature_mode',
        search_space['feature_mode']['choices']
    )

    # Model architecture
    n_layers = trial.suggest_int(
        'n_layers',
        search_space['n_layers']['low'],
        search_space['n_layers']['high']
    )
    # Sample each layer dimension independently
    hidden_dims = []
    for i in range(n_layers):
        dim = trial.suggest_categorical(f'hidden_dim_layer_{i}', [64, 128, 256, 512])
        hidden_dims.append(dim)

    dropout_rate = trial.suggest_float(
        'dropout_rate',
        search_space['dropout_rate']['low'],
        search_space['dropout_rate']['high']
    )

    # Pretraining hyperparameters
    lr_pre = trial.suggest_float(
        'lr_pre',
        search_space['lr_pre']['low'],
        search_space['lr_pre']['high'],
        log=True
    )
    epochs_pre = trial.suggest_int(
        'epochs_pre',
        search_space['epochs_pre']['low'],
        search_space['epochs_pre']['high']
    )
    batch_pre = trial.suggest_categorical(
        'batch_pre',
        search_space['batch_pre']['choices']
    )
    weight_decay_pre = trial.suggest_float(
        'weight_decay_pre',
        search_space['weight_decay_pre']['low'],
        search_space['weight_decay_pre']['high'],
        log=True
    )

    # Fine-tuning hyperparameters
    lr_ft = trial.suggest_float(
        'lr_ft',
        search_space['lr_ft']['low'],
        search_space['lr_ft']['high'],
        log=True
    )
    epochs_ft = trial.suggest_int(
        'epochs_ft',
        search_space['epochs_ft']['low'],
        search_space['epochs_ft']['high']
    )
    batch_ft = trial.suggest_categorical(
        'batch_ft',
        search_space['batch_ft']['choices']
    )
    # Sample n_freeze_layers dynamically based on actual n_layers
    # This ensures we never try to freeze more layers than exist
    n_freeze_layers = trial.suggest_int('n_freeze_layers', 0, n_layers)
    weight_decay_ft = trial.suggest_float(
        'weight_decay_ft',
        search_space['weight_decay_ft']['low'],
        search_space['weight_decay_ft']['high'],
        log=True
    )

    # Split seed
    split_seed = trial.suggest_int(
        'split_seed',
        search_space['split_seed']['low'],
        search_space['split_seed']['high']
    )

    # Set random seed
    set_random_seed(split_seed)

    # =========================================================================
    # PRETRAINING: DFT Chi Regression
    # =========================================================================
    print(f"\n[Trial {trial.number}] Starting pretraining...")

    # Load DFT data
    dft_df = load_precomputed_features(config['data']['dft_features_csv'])

    # Split data (always use fixed seed for pretraining consistency across trials)
    train_df, val_df, test_df = split_dft_data(
        dft_df,
        train_ratio=config['training']['split']['dft_train_ratio'],
        val_ratio=config['training']['split']['dft_val_ratio'],
        test_ratio=config['training']['split']['dft_test_ratio'],
        seed=config['random_seed']  # Always 42 for pretraining
    )

    # Extract features
    X_train, y_train = extract_feature_columns(train_df, feature_mode)
    X_val, y_val = extract_feature_columns(val_df, feature_mode)
    X_test, y_test = extract_feature_columns(test_df, feature_mode)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_pre, shuffle=True)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_pre, shuffle=False)

    # Create model
    input_dim = X_train.shape[1]

    model = RegressionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation='relu'
    )

    device = get_device()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_pre, weight_decay=weight_decay_pre)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs_pre):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate pretraining
    _, _, _, train_metrics = evaluate_mc_dropout(
        model, X_train_t, y_train_t, n_samples=50, device=device
    )
    _, _, _, val_metrics = evaluate_mc_dropout(
        model, X_val_t, y_val_t, n_samples=50, device=device
    )
    _, _, _, test_metrics = evaluate_mc_dropout(
        model, X_test_t, y_test_t, n_samples=50, device=device
    )

    # Store DFT R² scores
    trial.set_user_attr('r2_dft_train', train_metrics['r2'])
    trial.set_user_attr('r2_dft_val', val_metrics['r2'])
    trial.set_user_attr('r2_dft_test', test_metrics['r2'])

    print(f"[Trial {trial.number}] Pretraining R² - Train: {train_metrics['r2']:.4f}, "
          f"Val: {val_metrics['r2']:.4f}, Test: {test_metrics['r2']:.4f}")

    # =========================================================================
    # FINE-TUNING: Binary Solubility Classification
    # =========================================================================
    print(f"[Trial {trial.number}] Starting fine-tuning...")

    # Load binary data
    binary_df = load_precomputed_features(config['data']['binary_features_csv'])
    X, y = extract_feature_columns(binary_df, feature_mode)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)

    # 5-fold stratified CV
    n_folds = config['training']['split']['cv_folds']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)

    fold_val_f1s = []
    fold_train_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train_fold = X_t[train_idx]
        y_train_fold = y_t[train_idx]
        X_val_fold = X_t[val_idx]
        y_val_fold = y_t[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_ft, shuffle=True)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_ft, shuffle=False)

        # Create classification model with pretrained encoder
        reg_model = RegressionModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation='relu'
        )
        reg_model.load_state_dict(best_model_state)
        pretrained_encoder = reg_model.get_encoder()

        clf_model = ClassificationModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation='relu',
            pretrained_encoder=pretrained_encoder,
            n_freeze_layers=n_freeze_layers
        )
        clf_model = clf_model.to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(clf_model.parameters(), lr=lr_ft, weight_decay=weight_decay_ft)

        # Training loop
        best_val_acc = 0.0

        for epoch in range(epochs_ft):
            train_loss = train_epoch(clf_model, train_loader, criterion, optimizer, device)
            val_loss, val_proba, val_true, val_acc = evaluate_classification_epoch(
                clf_model, val_loader, criterion, device
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Final evaluation
        clf_model.eval()
        with torch.no_grad():
            # Validation
            val_logits = clf_model(X_val_fold.to(device))
            val_proba_final = torch.sigmoid(val_logits).cpu().numpy().squeeze()
            val_pred_final = (val_proba_final >= 0.5).astype(int)
            val_true_final = y_val_fold.cpu().numpy().squeeze()

            # Training
            train_logits = clf_model(X_train_fold.to(device))
            train_proba_final = torch.sigmoid(train_logits).cpu().numpy().squeeze()
            train_pred_final = (train_proba_final >= 0.5).astype(int)
            train_true_final = y_train_fold.cpu().numpy().squeeze()

        # Compute metrics
        val_metrics = compute_classification_metrics(val_true_final, val_pred_final, val_proba_final)
        train_metrics = compute_classification_metrics(train_true_final, train_pred_final, train_proba_final)

        fold_val_f1s.append(val_metrics['f1'])
        fold_train_f1s.append(train_metrics['f1'])

        # Store per-fold F1
        trial.set_user_attr(f'f1_cv_val_fold_{fold}', val_metrics['f1'])

    # Compute mean and std
    f1_cv_val_mean = np.mean(fold_val_f1s)
    f1_cv_val_std = np.std(fold_val_f1s)
    f1_cv_train_mean = np.mean(fold_train_f1s)
    f1_cv_train_std = np.std(fold_train_f1s)

    trial.set_user_attr('f1_cv_val_mean', f1_cv_val_mean)
    trial.set_user_attr('f1_cv_val_std', f1_cv_val_std)
    trial.set_user_attr('f1_cv_train_mean', f1_cv_train_mean)
    trial.set_user_attr('f1_cv_train_std', f1_cv_train_std)

    print(f"[Trial {trial.number}] Fine-tuning F1 - Train: {f1_cv_train_mean:.4f} ± {f1_cv_train_std:.4f}, "
          f"Val: {f1_cv_val_mean:.4f} ± {f1_cv_val_std:.4f}")

    return f1_cv_val_mean


def run_optimization(config_path: str = 'config.yaml'):
    """
    Run Optuna hyperparameter optimization.

    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = config['outputs']['hyperopt_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        study_name=config['optuna']['study_name'],
        direction=config['optuna']['direction'],
        storage=config['optuna']['storage']
    )

    # Create files for logging
    all_trials_path = os.path.join(output_dir, 'hy.txt')
    best_params_path = os.path.join(output_dir, 'best_hyperparameters.txt')

    # Clear existing files
    with open(all_trials_path, 'w') as f:
        f.write("Trial\tF1_Val_Mean\tR2_Train\tR2_Val\tR2_Test\tF1_Train_Mean\tParams\n")

    # Custom callback to log trials
    def log_trial(study, trial):
        # Get trial info
        trial_num = trial.number
        f1_val = trial.user_attrs.get('f1_cv_val_mean', float('nan'))
        f1_train = trial.user_attrs.get('f1_cv_train_mean', float('nan'))
        r2_train = trial.user_attrs.get('r2_dft_train', float('nan'))
        r2_val = trial.user_attrs.get('r2_dft_val', float('nan'))
        r2_test = trial.user_attrs.get('r2_dft_test', float('nan'))
        params = trial.params

        # Append to all trials file
        with open(all_trials_path, 'a') as f:
            f.write(f"{trial_num}\t{f1_val:.6f}\t{r2_train:.6f}\t{r2_val:.6f}\t"
                   f"{r2_test:.6f}\t{f1_train:.6f}\t{params}\n")

        # Update best parameters file
        if study.best_trial.number == trial_num:
            with open(best_params_path, 'w') as f:
                f.write("BEST HYPERPARAMETERS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Trial: {trial_num}\n")
                f.write(f"F1 Score (Val): {f1_val:.6f}\n")
                f.write(f"F1 Score (Train): {f1_train:.6f}\n")
                f.write(f"R² (DFT Train): {r2_train:.6f}\n")
                f.write(f"R² (DFT Val): {r2_val:.6f}\n")
                f.write(f"R² (DFT Test): {r2_test:.6f}\n\n")
                f.write("Parameters:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")

    # Run optimization
    n_trials = config['optuna']['n_trials']
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    print(f"Results will be saved to: {output_dir}")

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=n_trials,
        callbacks=[log_trial]
    )

    # Print best results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to:")
    print(f"  All trials: {all_trials_path}")
    print(f"  Best params: {best_params_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    run_optimization(args.config)


if __name__ == '__main__':
    main()
