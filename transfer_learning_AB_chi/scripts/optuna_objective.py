"""
Optuna hyperparameter optimization for polymer-water χ transfer learning.

Objective: Maximize mean validation R² over 5-fold CV on experimental data.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import optuna
import torch
import torch.optim as optim
import os
import copy
from typing import Dict, Any

from data_utils import load_features, get_dft_splits, get_experiment_folds
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

    for param_name, param_config in hp_space.items():
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


def run_pretraining(
    hp: Dict[str, Any],
    dft_train_df: pd.DataFrame,
    dft_val_df: pd.DataFrame,
    dft_test_df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Run DFT pretraining with given hyperparameters.

    Args:
        hp: Hyperparameters
        dft_train_df: DFT training DataFrame
        dft_val_df: DFT validation DataFrame
        dft_test_df: DFT test DataFrame
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Dictionary with pretrained model, history, and metrics
    """
    # Filter and scale descriptors based on DFT training data
    dft_train_scaled, dft_val_scaled, dft_test_scaled, valid_desc_cols, desc_means, desc_stds = \
        filter_and_scale_descriptors(dft_train_df, dft_val_df, dft_test_df)

    # Build features
    X_train, T_train, y_train = build_features(dft_train_scaled, hp['feature_mode'], valid_desc_cols)
    X_val, T_val, y_val = build_features(dft_val_scaled, hp['feature_mode'], valid_desc_cols)
    X_test, T_test, y_test = build_features(dft_test_scaled, hp['feature_mode'], valid_desc_cols)

    # Create dataloaders
    train_loader = create_dataloader(X_train, T_train, y_train, batch_size=hp['batch_pre'], shuffle=True)
    val_loader = create_dataloader(X_val, T_val, y_val, batch_size=hp['batch_pre'], shuffle=False)

    # Create model
    input_dim = X_train.shape[1]
    model = create_model(
        input_dim=input_dim,
        hidden_dim=hp['hidden_dim'],
        n_layers=hp['n_layers'],
        dropout_rate=hp['dropout_rate'],
        device=device
    )

    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=hp['lr_pre'], weight_decay=hp['weight_decay'])
    criterion = ChiLoss()

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=hp['epochs_pre'],
        verbose=False
    )

    # Evaluate with MC dropout
    n_mc_samples = config['global']['mc_dropout_samples']

    train_results = evaluate_mc_dropout(model, X_train, T_train, y_train, n_samples=n_mc_samples, device=device)
    val_results = evaluate_mc_dropout(model, X_val, T_val, y_val, n_samples=n_mc_samples, device=device)
    test_results = evaluate_mc_dropout(model, X_test, T_test, y_test, n_samples=n_mc_samples, device=device)

    return {
        'model': model,
        'history': history,
        'train_results': train_results,
        'val_results': val_results,
        'test_results': test_results,
        'valid_desc_cols': valid_desc_cols,
        'desc_means': desc_means,
        'desc_stds': desc_stds,
        'input_dim': input_dim
    }


def run_finetuning_cv(
    hp: Dict[str, Any],
    pretrain_results: Dict[str, Any],
    exp_df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Run 5-fold CV fine-tuning on experimental data.

    Args:
        hp: Hyperparameters
        pretrain_results: Results from pretraining
        exp_df: Experimental DataFrame
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Dictionary with CV results
    """
    # FIRST: Scale experimental descriptors using DFT statistics
    desc_cols = pretrain_results['valid_desc_cols']
    exp_df_scaled = exp_df.copy()
    for col in desc_cols:
        exp_df_scaled[col] = (exp_df[col] - pretrain_results['desc_means'][col]) / pretrain_results['desc_stds'][col]

    # Validate and clean NaN/Inf values
    exp_df_scaled = validate_features(exp_df_scaled, desc_cols)

    # THEN: Get folds from the SCALED dataframe
    folds = get_experiment_folds(exp_df_scaled, split_seed=hp['split_seed'], n_folds=config['global']['n_folds'])

    # Store results for each fold
    fold_metrics = []
    fold_val_results = []
    fold_train_results = []

    for fold_idx, fold_data in enumerate(folds):
        train_df = fold_data['train']
        val_df = fold_data['val']

        # Build features
        X_train, T_train, y_train = build_features(train_df, hp['feature_mode'], desc_cols)
        X_val, T_val, y_val = build_features(val_df, hp['feature_mode'], desc_cols)

        # Create dataloaders
        train_loader = create_dataloader(X_train, T_train, y_train, batch_size=hp['batch_ft'], shuffle=True)
        val_loader = create_dataloader(X_val, T_val, y_val, batch_size=hp['batch_ft'], shuffle=False)

        # Create new model and load pretrained weights
        model = create_model(
            input_dim=pretrain_results['input_dim'],
            hidden_dim=hp['hidden_dim'],
            n_layers=hp['n_layers'],
            dropout_rate=hp['dropout_rate'],
            device=device
        )

        # Load pretrained weights
        model.load_state_dict(pretrain_results['model'].state_dict())

        # Apply freeze strategy
        if hp['freeze_strategy'] == 'freeze_lower':
            model.freeze_lower_layers()
        else:  # 'all_trainable'
            model.unfreeze_all()

        # Create optimizer and criterion
        optimizer = optim.AdamW(model.parameters(), lr=hp['lr_ft'], weight_decay=hp['weight_decay'])
        criterion = ChiLoss()

        # Fine-tune
        _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_epochs=hp['epochs_ft'],
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
    Optuna objective function.

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

    # Load data
    dft_df = load_features(config['data']['dft_features'])
    exp_df = load_features(config['data']['exp_features'])

    # Get fixed DFT splits
    dft_train_df, dft_val_df, dft_test_df = get_dft_splits(
        dft_df,
        global_seed=config['global']['dft_split_seed'],
        train_frac=config['global']['dft_train_frac'],
        val_frac=config['global']['dft_val_frac'],
        test_frac=config['global']['dft_test_frac']
    )

    # Run pretraining
    print("Running DFT pretraining...")
    pretrain_results = run_pretraining(hp, dft_train_df, dft_val_df, dft_test_df, config, device)

    # Store DFT results as user attributes
    trial.set_user_attr('r2_dft_train', pretrain_results['train_results']['r2'])
    trial.set_user_attr('r2_dft_val', pretrain_results['val_results']['r2'])
    trial.set_user_attr('r2_dft_test', pretrain_results['test_results']['r2'])

    print(f"DFT - Train R²: {pretrain_results['train_results']['r2']:.4f}, "
          f"Val R²: {pretrain_results['val_results']['r2']:.4f}, "
          f"Test R²: {pretrain_results['test_results']['r2']:.4f}")

    # Run 5-fold CV fine-tuning
    print("Running 5-fold CV fine-tuning...")
    cv_results = run_finetuning_cv(hp, pretrain_results, exp_df.copy(), config, device)

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
        f.write("Trial\tObjective_R2\tDFT_Train_R2\tDFT_Val_R2\tDFT_Test_R2\tCV_Train_R2_Mean\tCV_Val_R2_Mean\tHyperparameters\n")

        for trial in study.trials:
            # Format hyperparameters as compact string
            hp_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])

            # Get R² values, handle None/missing values
            obj_val = trial.value if trial.value is not None else float('nan')
            dft_train = trial.user_attrs.get('r2_dft_train', float('nan'))
            dft_val = trial.user_attrs.get('r2_dft_val', float('nan'))
            dft_test = trial.user_attrs.get('r2_dft_test', float('nan'))
            cv_train = trial.user_attrs.get('r2_cv_train_mean', float('nan'))
            cv_val = trial.user_attrs.get('r2_cv_val_mean', float('nan'))

            f.write(f"{trial.number}\t"
                   f"{obj_val:.4f}\t"
                   f"{dft_train:.4f}\t"
                   f"{dft_val:.4f}\t"
                   f"{dft_test:.4f}\t"
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
        f.write("Best Hyperparameters\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Best Trial: {best_trial.number}\n")
        f.write(f"Best CV Val R²: {best_trial.value:.4f}\n\n")

        f.write("Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")

        f.write("\nPerformance:\n")
        f.write(f"DFT Train R²: {best_trial.user_attrs.get('r2_dft_train', 'N/A'):.4f}\n")
        f.write(f"DFT Val R²:   {best_trial.user_attrs.get('r2_dft_val', 'N/A'):.4f}\n")
        f.write(f"DFT Test R²:  {best_trial.user_attrs.get('r2_dft_test', 'N/A'):.4f}\n")
        f.write(f"CV Train R²:  {best_trial.user_attrs.get('r2_cv_train_mean', 'N/A'):.4f}\n")
        f.write(f"CV Val R²:    {best_trial.user_attrs.get('r2_cv_val_mean', 'N/A'):.4f} ± "
                f"{best_trial.user_attrs.get('r2_cv_val_std', 'N/A'):.4f}\n")

    print(f"Saved best hyperparameters to {output_file}")


def main():
    """Main function to run Optuna optimization."""
    parser = argparse.ArgumentParser(description='Run Optuna hyperparameter optimization')
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

    # Run optimization
    study.optimize(objective, n_trials=config['optuna']['n_trials'])

    # Print best results
    print("\n" + "=" * 80)
    print("Optimization completed!")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV Val R²: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    output_dir = 'hyperparameter_optimization'
    save_trial_results(study, output_dir)
    save_best_params(study, output_dir)


if __name__ == '__main__':
    main()
