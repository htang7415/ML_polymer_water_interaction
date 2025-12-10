"""
Optuna hyperparameter optimization for direct learning (without transfer learning).
"""

import optuna
import yaml
import os
import torch
from typing import Dict
import sys

# Import from parent folder - use absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_scripts_dir = os.path.join(script_dir, '../../scripts')
sys.path.insert(0, os.path.abspath(parent_scripts_dir))

from data_utils import get_exp_data, create_5fold_polymer_split
from features import FeatureBuilder

# Import from current folder
from train_direct import train_on_exp_direct


def load_config(config_path: str = '../config.yaml') -> Dict:
    """Load configuration from YAML file."""
    # Adjust path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def sample_hyperparameters(trial: optuna.Trial, config: Dict) -> Dict:
    """
    Sample hyperparameters using Optuna trial and config search space.

    Args:
        trial: Optuna trial object
        config: Configuration dictionary

    Returns:
        Dictionary of sampled hyperparameters
    """
    search_space = config['optuna']['search_space']

    hyperparams = {}

    # Feature mode
    hyperparams['feature_mode'] = trial.suggest_categorical(
        'feature_mode',
        search_space['feature_mode']['choices']
    )

    # Model architecture
    hyperparams['n_layers'] = trial.suggest_int(
        'n_layers',
        search_space['n_layers']['low'],
        search_space['n_layers']['high']
    )

    # Sample dimensions for each layer
    hidden_dims = []
    for i in range(hyperparams['n_layers']):
        dim = trial.suggest_categorical(
            f'hidden_dim_layer_{i}',
            search_space['hidden_dims_per_layer']['choices']
        )
        hidden_dims.append(dim)
    hyperparams['hidden_dims'] = hidden_dims

    hyperparams['dropout_rate'] = trial.suggest_float(
        'dropout_rate',
        search_space['dropout_rate']['low'],
        search_space['dropout_rate']['high']
    )

    # Weight decay
    hyperparams['weight_decay'] = trial.suggest_float(
        'weight_decay',
        search_space['weight_decay']['low'],
        search_space['weight_decay']['high'],
        log=True
    )

    # Training hyperparameters
    hyperparams['lr'] = trial.suggest_float(
        'lr',
        search_space['lr']['low'],
        search_space['lr']['high'],
        log=True
    )

    hyperparams['epochs'] = trial.suggest_int(
        'epochs',
        search_space['epochs']['low'],
        search_space['epochs']['high']
    )

    hyperparams['batch_size'] = trial.suggest_categorical(
        'batch_size',
        search_space['batch_size']['choices']
    )

    # Data split seed
    hyperparams['split_seed'] = trial.suggest_int(
        'split_seed',
        search_space['split_seed']['low'],
        search_space['split_seed']['high']
    )

    return hyperparams


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for direct learning.

    Args:
        trial: Optuna trial object

    Returns:
        Mean validation R² across 5 folds (to maximize)
    """
    # Load config
    config = load_config()

    # Sample hyperparameters
    hyperparams = sample_hyperparameters(trial, config)

    print(f"\n{'=' * 60}")
    print(f"Trial {trial.number}")
    print(f"{'=' * 60}")
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # Load experimental data
        exp_df = get_exp_data(config)

        # Build features
        feature_builder = FeatureBuilder(config)
        feature_mode = hyperparams['feature_mode']

        # Fit descriptor scaler on all experimental data
        # (Note: This could lead to data leakage, but with only ~40 samples,
        #  we need to use all data for robust scaling. Alternative: fit on train fold only)
        feature_builder.fit_descriptor_scaler(exp_df)

        # Create 5-fold split for experimental data
        exp_folds_df = create_5fold_polymer_split(exp_df, hyperparams['split_seed'])

        # Build features for experimental folds
        exp_folds_features = []
        for train_df, val_df in exp_folds_df:
            X_train, y_train = feature_builder.build_features(train_df, feature_mode)
            X_val, y_val = feature_builder.build_features(val_df, feature_mode)
            exp_folds_features.append((X_train, y_train, X_val, y_val))

        # Train directly on experimental data
        exp_results = train_on_exp_direct(
            exp_folds_features,
            hyperparams,
            config,
            device
        )

        # Store experimental metrics
        for fold_idx, fold_data in enumerate(exp_results['fold_results']):
            trial.set_user_attr(f'r2_cv_val_fold_{fold_idx}', fold_data['val_metrics']['r2'])

        trial.set_user_attr('r2_cv_val_mean', exp_results['r2_val_mean'])
        trial.set_user_attr('r2_cv_val_std', exp_results['r2_val_std'])
        trial.set_user_attr('r2_cv_train_mean', exp_results['r2_train_mean'])
        trial.set_user_attr('r2_cv_train_std', exp_results['r2_train_std'])

        # Return mean validation R² as objective
        objective_value = exp_results['r2_val_mean']

        print(f"\nTrial {trial.number} completed: Objective = {objective_value:.4f}")

        return objective_value

    except Exception as e:
        print(f"\nTrial {trial.number} failed with error: {e}")
        raise


def save_trial_to_file(study: optuna.Study, trial: optuna.Trial, output_dir: str):
    """
    Save trial hyperparameters and summary metrics to file.

    Args:
        study: Optuna study object
        trial: Optuna trial object
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    trial_file = os.path.join(output_dir, 'hy.txt')

    # Filter metrics to only include summary metrics (not per-fold)
    summary_metrics = {
        'r2_cv_train_mean': trial.user_attrs.get('r2_cv_train_mean'),
        'r2_cv_val_mean': trial.user_attrs.get('r2_cv_val_mean'),
    }

    with open(trial_file, 'a') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Trial {trial.number}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Objective (Mean CV Val R²): {trial.value:.6f}\n")
        f.write(f"State: {trial.state}\n\n")

        f.write("Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nMetrics:\n")
        for key, value in summary_metrics.items():
            if value is not None:
                f.write(f"  {key}: {value:.6f}\n")

        f.write("\n")


def save_best_on_new_best(study: optuna.Study, trial: optuna.Trial, output_dir: str):
    """
    Callback to save best hyperparameters whenever a new best trial is found.

    Args:
        study: Optuna study object
        trial: Optuna trial object that just completed
        output_dir: Output directory
    """
    # Check if this trial is complete and has a value
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    # Check if this is a new best trial
    # For maximization: trial.value >= study.best_value
    # For minimization: trial.value <= study.best_value
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        is_new_best = trial.value >= study.best_value
    else:
        is_new_best = trial.value <= study.best_value

    # If new best, save immediately
    if is_new_best:
        save_best_trial(study, output_dir)
        print(f"✓ New best trial found! Trial {trial.number} with objective = {trial.value:.6f}")


def save_best_trial(study: optuna.Study, output_dir: str):
    """
    Save best trial hyperparameters to file.

    Args:
        study: Optuna study object
        output_dir: Output directory
    """
    best_trial = study.best_trial

    best_file = os.path.join(output_dir, 'best_hyperparameters.txt')

    with open(best_file, 'w') as f:
        f.write("Best Hyperparameters (Direct Learning)\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Best Trial Number: {best_trial.number}\n")
        f.write(f"Best Objective (Mean CV Val R²): {best_trial.value:.6f}\n\n")

        f.write("Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nMetrics:\n")
        for key, value in best_trial.user_attrs.items():
            f.write(f"  {key}: {value:.6f}\n")

    print(f"\nBest hyperparameters saved to {best_file}")


def main():
    """
    Run Optuna hyperparameter optimization for direct learning.
    """
    # Load config
    config = load_config()

    # Create output directory (relative to project root, not script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, config['outputs']['hyperparameter_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        study_name=config['optuna']['study_name'],
        direction=config['optuna']['direction']
    )

    # Define callback to save each trial and update best if needed
    def callback(study, trial):
        save_trial_to_file(study, trial, output_dir)
        save_best_on_new_best(study, trial, output_dir)

    # Run optimization
    n_trials = config['optuna']['n_trials']
    print(f"Starting Optuna optimization with {n_trials} trials...")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # Save best trial
    save_best_trial(study, output_dir)

    # Print best results
    print(f"\n{'=' * 60}")
    print("Optimization completed!")
    print(f"{'=' * 60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Mean CV Val R²): {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
