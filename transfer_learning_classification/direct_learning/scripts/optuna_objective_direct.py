"""
Optuna hyperparameter optimization for direct learning (no transfer learning).

This script performs hyperparameter search for training a classification model
from scratch on binary solubility data, enabling comparison with transfer learning.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import optuna

# Import from parent scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from data_utils import load_precomputed_features, extract_feature_columns
from models import ClassificationModel
from utils import (
    set_random_seed, get_device,
    train_epoch, evaluate_classification_epoch,
    compute_classification_metrics
)


def objective(trial: optuna.Trial, config: dict) -> float:
    """
    Optuna objective function for direct learning hyperparameter optimization.

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

    # Training hyperparameters
    lr = trial.suggest_float(
        'lr',
        search_space['lr']['low'],
        search_space['lr']['high'],
        log=True
    )
    epochs = trial.suggest_int(
        'epochs',
        search_space['epochs']['low'],
        search_space['epochs']['high']
    )
    batch_size = trial.suggest_categorical(
        'batch_size',
        search_space['batch_size']['choices']
    )
    weight_decay = trial.suggest_float(
        'weight_decay',
        search_space['weight_decay']['low'],
        search_space['weight_decay']['high'],
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

    # Load binary solubility data
    print(f"\n[Trial {trial.number}] Loading data (feature_mode={feature_mode})...")
    binary_df = load_precomputed_features(config['data']['binary_features_csv'])
    X, y = extract_feature_columns(binary_df, feature_mode)

    # Handle NaN/Inf
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)

    # 5-fold stratified CV
    n_folds = config['training']['split']['cv_folds']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)

    fold_val_f1s = []
    fold_train_f1s = []

    device = get_device()
    input_dim = X.shape[1]

    print(f"[Trial {trial.number}] Architecture: {n_layers} layers {hidden_dims}, "
          f"dropout={dropout_rate:.3f}, lr={lr:.6f}, epochs={epochs}, batch={batch_size}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train_fold = X_t[train_idx]
        y_train_fold = y_t[train_idx]
        X_val_fold = X_t[val_idx]
        y_val_fold = y_t[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model FROM SCRATCH (no pretraining)
        model = ClassificationModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation='relu',
            pretrained_encoder=None,  # No transfer learning
            n_freeze_layers=0
        )
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop
        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_proba, val_true, val_acc = evaluate_classification_epoch(
                model, val_loader, criterion, device
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Validation
            val_logits = model(X_val_fold.to(device))
            val_proba_final = torch.sigmoid(val_logits).cpu().numpy().squeeze()
            val_pred_final = (val_proba_final >= 0.5).astype(int)
            val_true_final = y_val_fold.cpu().numpy().squeeze()

            # Training
            train_logits = model(X_train_fold.to(device))
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

    print(f"[Trial {trial.number}] Direct Learning F1 - Train: {f1_cv_train_mean:.4f} ± {f1_cv_train_std:.4f}, "
          f"Val: {f1_cv_val_mean:.4f} ± {f1_cv_val_std:.4f}")

    return f1_cv_val_mean


def run_optimization(config_path: str = 'config.yaml'):
    """
    Run Optuna hyperparameter optimization for direct learning.

    Args:
        config_path: Path to configuration file
    """
    # Load config
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

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
        f.write("Trial\tF1_Val_Mean\tF1_Val_Std\tF1_Train_Mean\tF1_Train_Std\tParams\n")

    # Custom callback to log trials
    def log_trial(study, trial):
        # Get trial info
        trial_num = trial.number
        f1_val = trial.user_attrs.get('f1_cv_val_mean', float('nan'))
        f1_val_std = trial.user_attrs.get('f1_cv_val_std', float('nan'))
        f1_train = trial.user_attrs.get('f1_cv_train_mean', float('nan'))
        f1_train_std = trial.user_attrs.get('f1_cv_train_std', float('nan'))
        params = trial.params

        # Append to all trials file
        with open(all_trials_path, 'a') as f:
            f.write(f"{trial_num}\t{f1_val:.6f}\t{f1_val_std:.6f}\t{f1_train:.6f}\t{f1_train_std:.6f}\t{params}\n")

        # Update best parameters file
        if study.best_trial.number == trial_num:
            with open(best_params_path, 'w') as f:
                f.write("BEST HYPERPARAMETERS (DIRECT LEARNING - NO TRANSFER LEARNING)\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Trial: {trial_num}\n")
                f.write(f"F1 Score (Val): {f1_val:.6f} ± {f1_val_std:.6f}\n")
                f.write(f"F1 Score (Train): {f1_train:.6f} ± {f1_train_std:.6f}\n\n")
                f.write("Parameters:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                f.write("To use these parameters, update config.yaml accordingly.\n")

    # Run optimization
    n_trials = config['optuna']['n_trials']
    print("\n" + "=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION (DIRECT LEARNING)")
    print("=" * 80)
    print(f"\nStarting optimization with {n_trials} trials...")
    print(f"Objective: Maximize validation F1 score")
    print(f"Results will be saved to: {output_dir}")
    print("\n")

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
    print("\nUpdate config.yaml with the best parameters and run direct_learning.sh")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Direct Learning Hyperparameter Optimization')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    run_optimization(args.config)


if __name__ == '__main__':
    main()
