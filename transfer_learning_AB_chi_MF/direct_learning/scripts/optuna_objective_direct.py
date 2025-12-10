"""
Optuna hyperparameter optimization for polymer-water χ direct learning.

Objective: Maximize mean validation R² over 5-fold CV on experimental data.
No transfer learning - train from random initialization.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import optuna
from optuna.trial import FrozenTrial
import torch
import torch.optim as optim
import os
from typing import Dict, Any

from data_utils import load_features, get_experiment_folds
from features import filter_and_scale_descriptors, build_features, validate_features
from model import create_model
from train import ChiLoss, train_model, create_dataloader, evaluate_mc_dropout, get_device


def sample_hyperparameters(trial: optuna.Trial, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sample hyperparameters using Optuna trial.

    Args:
        trial: Optuna trial object
        config: Configuration dictionary

    Returns:
        Dictionary of sampled hyperparameters
    """
    hp_space = config['hyperparameters']
    hp = {}

    # Sample n_layers first (needed for per-layer dimensions)
    n_layers = None
    if 'n_layers' in hp_space:
        param_config = hp_space['n_layers']
        n_layers = trial.suggest_int('n_layers', param_config['low'], param_config['high'])
        hp['n_layers'] = n_layers

    # Sample per-layer dimensions if configured
    if 'hidden_dim_per_layer' in hp_space:
        param_config = hp_space['hidden_dim_per_layer']
        hidden_dims = []
        for layer_idx in range(n_layers):
            dim = trial.suggest_categorical(
                f'hidden_dim_layer_{layer_idx}',
                param_config['choices']
            )
            hidden_dims.append(dim)
        hp['hidden_dims'] = hidden_dims

    # Sample all other parameters
    for param_name, param_config in hp_space.items():
        if param_name in ['n_layers', 'hidden_dim_per_layer']:
            continue  # Already sampled above

        param_type = param_config['type']

        if param_type == 'categorical':
            hp[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        elif param_type == 'int':
            hp[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
        elif param_type == 'float':
            hp[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
        elif param_type == 'loguniform':
            hp[param_name] = trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high'],
                log=True
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return hp


def run_direct_learning_cv(
    hp: Dict[str, Any],
    exp_df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Run 5-fold CV direct learning on experimental data.
    Train from random initialization (no transfer learning).

    Args:
        hp: Hyperparameters
        exp_df: Experimental DataFrame
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Dictionary with CV results
    """
    # Store results for each fold
    fold_metrics = []
    fold_val_results = []
    fold_train_results = []

    # Get folds
    folds = get_experiment_folds(exp_df, split_seed=hp['split_seed'], n_folds=config['global']['n_folds'])

    for fold_idx, fold_data in enumerate(folds):
        train_df = fold_data['train']
        val_df = fold_data['val']

        # Filter and scale descriptors based on this fold's training data
        train_scaled, val_scaled, _, valid_desc_cols, _, _ = \
            filter_and_scale_descriptors(train_df, val_df, None)

        # Build features
        X_train, T_train, y_train = build_features(train_scaled, hp['feature_mode'], valid_desc_cols)
        X_val, T_val, y_val = build_features(val_scaled, hp['feature_mode'], valid_desc_cols)

        # Create dataloaders
        train_loader = create_dataloader(X_train, T_train, y_train, batch_size=hp['batch_size'], shuffle=True)
        val_loader = create_dataloader(X_val, T_val, y_val, batch_size=hp['batch_size'], shuffle=False)

        # Create model from RANDOM initialization (key difference from transfer learning)
        input_dim = X_train.shape[1]

        # Support both new and legacy formats
        if 'hidden_dims' in hp:
            model = create_model(
                input_dim=input_dim,
                hidden_dims=hp['hidden_dims'],
                n_layers=hp['n_layers'],
                dropout_rate=hp['dropout_rate'],
                device=device
            )
        else:
            # Legacy format
            model = create_model(
                input_dim=input_dim,
                hidden_dim=hp['hidden_dim'],
                n_layers=hp['n_layers'],
                dropout_rate=hp['dropout_rate'],
                device=device
            )

        # Create optimizer and criterion
        optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
        criterion = ChiLoss()

        # Train from scratch
        _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_epochs=hp['epochs'],
            verbose=False
        )

        # Evaluate with MC dropout
        n_mc_samples = config['global']['mc_dropout_samples']

        train_results = evaluate_mc_dropout(model, X_train, T_train, y_train, n_samples=n_mc_samples, device=device)
        val_results = evaluate_mc_dropout(model, X_val, T_val, y_val, n_samples=n_mc_samples, device=device)

        fold_metrics.append({
            'r2': val_results['r2'],
            'mae': val_results['mae'],
            'rmse': val_results['rmse']
        })

        fold_val_results.append(val_results)
        fold_train_results.append(train_results)

    # Compute aggregate metrics
    r2_values = [m['r2'] for m in fold_metrics]
    mae_values = [m['mae'] for m in fold_metrics]
    rmse_values = [m['rmse'] for m in fold_metrics]

    train_r2_values = [r['r2'] for r in fold_train_results]

    return {
        'fold_metrics': fold_metrics,
        'fold_val_results': fold_val_results,
        'fold_train_results': fold_train_results,
        'r2_val_mean': np.mean(r2_values),
        'r2_val_std': np.std(r2_values),
        'r2_train_mean': np.mean(train_r2_values),
        'r2_train_std': np.std(train_r2_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values)
    }


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for direct learning.

    Args:
        trial: Optuna trial

    Returns:
        Mean validation R² over 5-fold CV (to be maximized)
    """
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Sample hyperparameters
    hp = sample_hyperparameters(trial, config)

    print(f"\nTrial {trial.number}:")
    print(f"Hyperparameters: {hp}")

    # Get device
    device = get_device()

    # Load experimental data
    exp_df = load_features(config['data']['exp_features'])

    # Run 5-fold CV direct learning
    print("Running 5-fold CV direct learning...")
    cv_results = run_direct_learning_cv(hp, exp_df.copy(), config, device)

    # Store CV results as user attributes
    for fold_idx, metrics in enumerate(cv_results['fold_metrics']):
        trial.set_user_attr(f'r2_cv_val_fold_{fold_idx}', metrics['r2'])

    trial.set_user_attr('r2_cv_val_mean', cv_results['r2_val_mean'])
    trial.set_user_attr('r2_cv_val_std', cv_results['r2_val_std'])
    trial.set_user_attr('r2_cv_train_mean', cv_results['r2_train_mean'])
    trial.set_user_attr('r2_cv_train_std', cv_results['r2_train_std'])

    print(f"CV - Val R²: {cv_results['r2_val_mean']:.4f} ± {cv_results['r2_val_std']:.4f}")
    print(f"CV - Train R²: {cv_results['r2_train_mean']:.4f} ± {cv_results['r2_train_std']:.4f}")

    # Return objective value to maximize
    return cv_results['r2_val_mean']


def save_trial_results(study: optuna.Study, output_dir: str):
    """
    Save all trial results to hy.txt in compact tab-separated format.

    Args:
        study: Optuna study
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'hy.txt')

    with open(output_file, 'w') as f:
        # Header
        f.write("Trial\tObjective_R2\tCV_Train_R2_Mean\tCV_Val_R2_Mean\tHyperparameters\n")

        for trial in study.trials:
            # Format hyperparameters as compact string
            hp_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])

            # Get R² values, handle None/missing values
            obj_val = trial.value if trial.value is not None else float('nan')
            cv_train = trial.user_attrs.get('r2_cv_train_mean', float('nan'))
            cv_val = trial.user_attrs.get('r2_cv_val_mean', float('nan'))

            f.write(f"{trial.number}\t"
                   f"{obj_val:.4f}\t"
                   f"{cv_train:.4f}\t"
                   f"{cv_val:.4f}\t"
                   f"{hp_str}\n")

    print(f"Saved all trial results to {output_file}")


def save_best_params(study: optuna.Study, output_dir: str):
    """
    Save best hyperparameters to a text file.

    Args:
        study: Optuna study
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    best_trial = study.best_trial
    output_file = os.path.join(output_dir, 'best_hyperparameters.txt')

    with open(output_file, 'w') as f:
        f.write("Best Hyperparameters (Direct Learning)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Best Trial: {best_trial.number}\n")
        f.write(f"Best CV Val R²: {best_trial.value:.4f}\n\n")

        f.write("Hyperparameters:\n")

        # Group per-layer dimensions for readability
        layer_dims = []
        other_params = {}

        for key, value in sorted(best_trial.params.items()):
            if key.startswith('hidden_dim_layer_'):
                layer_idx = int(key.split('_')[-1])
                layer_dims.append((layer_idx, value))
            else:
                other_params[key] = value

        # Write architecture summary if per-layer dimensions exist
        if layer_dims:
            layer_dims.sort()
            f.write("# Architecture:\n")
            for idx, dim in layer_dims:
                f.write(f"hidden_dim_layer_{idx}: {dim}\n")
            f.write("\n")

        # Write other params
        for key, value in other_params.items():
            f.write(f"{key}: {value}\n")

        f.write("\nPerformance:\n")
        f.write(f"CV Train R²:  {best_trial.user_attrs.get('r2_cv_train_mean', 'N/A'):.4f}\n")
        f.write(f"CV Val R²:    {best_trial.user_attrs.get('r2_cv_val_mean', 'N/A'):.4f} ± "
                f"{best_trial.user_attrs.get('r2_cv_val_std', 'N/A'):.4f}\n")

    print(f"Saved best hyperparameters to {output_file}")


class SaveResultsCallback:
    """
    Optuna callback to save results after each trial.

    This ensures incremental saving so no data is lost if the optimization
    is interrupted or crashes.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, 'hy.txt')
        os.makedirs(output_dir, exist_ok=True)

        # Initialize file with header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                f.write("Trial\tObjective_R2\tCV_Train_R2_Mean\tCV_Val_R2_Mean\tHyperparameters\n")

    def __call__(self, study: optuna.Study, trial: FrozenTrial):
        """
        Called after each trial completes.

        Args:
            study: Optuna study
            trial: Completed trial
        """
        # Append this trial's results to hy.txt
        with open(self.output_file, 'a') as f:
            # Format hyperparameters as compact string
            hp_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])

            # Get R² values, handle None/missing values
            obj_val = trial.value if trial.value is not None else float('nan')
            cv_train = trial.user_attrs.get('r2_cv_train_mean', float('nan'))
            cv_val = trial.user_attrs.get('r2_cv_val_mean', float('nan'))

            f.write(f"{trial.number}\t"
                   f"{obj_val:.4f}\t"
                   f"{cv_train:.4f}\t"
                   f"{cv_val:.4f}\t"
                   f"{hp_str}\n")

        # Update best hyperparameters if this is the best trial so far
        if study.best_trial.number == trial.number:
            save_best_params(study, self.output_dir)

        print(f"Saved trial {trial.number} to {self.output_file}")


def main():
    """Main function to run Optuna optimization."""
    parser = argparse.ArgumentParser(description='Run Optuna hyperparameter optimization for direct learning')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create study (in-memory only, no database)
    study = optuna.create_study(
        study_name=config['optuna']['study_name'],
        storage=None,
        direction=config['optuna']['direction']
    )

    # Create callback for incremental saving
    output_dir = 'hyperparameter_optimization'
    callback = SaveResultsCallback(output_dir)

    # Run optimization with callback
    print(f"Results will be saved incrementally to {output_dir}/hy.txt after each trial")
    study.optimize(objective, n_trials=config['optuna']['n_trials'], callbacks=[callback])

    # Print best results
    print("\n" + "=" * 80)
    print("Optimization completed!")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV Val R²: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results (backup - already saved incrementally during optimization)
    save_trial_results(study, output_dir)
    save_best_params(study, output_dir)


if __name__ == '__main__':
    main()
