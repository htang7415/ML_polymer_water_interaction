"""
Run final direct learning training and evaluation with best hyperparameters.
"""

import yaml
import os
import torch
import json
import sys

# Import from parent folder - use absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_scripts_dir = os.path.join(script_dir, '../../scripts')
sys.path.insert(0, os.path.abspath(parent_scripts_dir))

from data_utils import get_exp_data, create_5fold_polymer_split
from features import FeatureBuilder
from plotting import (
    setup_plot_style,
    plot_parity_folds,
    plot_calibration_folds
)

# Import from current folder
from train_direct import train_on_exp_direct


def load_config(config_path: str = '../config.yaml'):
    """Load configuration from YAML file."""
    # Adjust path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_best_hyperparameters(best_params_file: str):
    """
    Load best hyperparameters from file.

    Args:
        best_params_file: Path to best_hyperparameters.txt

    Returns:
        Dictionary of hyperparameters
    """
    hyperparams = {}

    with open(best_params_file, 'r') as f:
        lines = f.readlines()

        # Find the start of hyperparameters section
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if line.strip() == "Hyperparameters:":
                start_idx = i + 1
            elif start_idx is not None and line.strip() == "Metrics:":
                end_idx = i
                break

        if start_idx is None or end_idx is None:
            raise ValueError("Could not parse hyperparameters from file")

        # Parse hyperparameters
        for i in range(start_idx, end_idx):
            line = lines[i].strip()
            if line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Convert to appropriate type
                try:
                    if '.' in value:
                        hyperparams[key] = float(value)
                    else:
                        hyperparams[key] = int(value)
                except:
                    hyperparams[key] = value

    return hyperparams


def get_hyperparameters(config):
    """
    Get hyperparameters: either from best file or from config defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of hyperparameters
    """
    # Adjust path relative to script location (go up one directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    best_params_file = os.path.join(
        project_dir,
        config['outputs']['hyperparameter_dir'],
        'best_hyperparameters.txt'
    )

    if os.path.exists(best_params_file):
        print(f"Loading best hyperparameters from {best_params_file}")
        hyperparams = load_best_hyperparameters(best_params_file)

        # Convert hidden_dim_layer_i format to hidden_dims list
        n_layers = hyperparams['n_layers']
        hidden_dims = []
        for i in range(n_layers):
            layer_key = f'hidden_dim_layer_{i}'
            if layer_key in hyperparams:
                hidden_dims.append(hyperparams[layer_key])
                del hyperparams[layer_key]
        hyperparams['hidden_dims'] = hidden_dims
    else:
        print("Best hyperparameters file not found, using defaults from config.yaml")
        hyperparams = {
            'feature_mode': config['training']['feature_mode'],
            'n_layers': config['model']['n_layers'],
            'hidden_dims': config['model']['hidden_dims'],
            'dropout_rate': config['model']['dropout_rate'],
            'weight_decay': config['training']['weight_decay'],
            'lr': config['training']['lr'],
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'split_seed': config['training']['split_seed'],
        }

    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    return hyperparams


def save_metrics(exp_results, output_dir):
    """
    Save all metrics to JSON file.

    Args:
        exp_results: Results from direct learning
        output_dir: Output directory
    """
    metrics = {
        'experimental': {
            'r2_val_mean': exp_results['r2_val_mean'],
            'r2_val_std': exp_results['r2_val_std'],
            'r2_train_mean': exp_results['r2_train_mean'],
            'r2_train_std': exp_results['r2_train_std'],
            'folds': []
        }
    }

    for fold_data in exp_results['fold_results']:
        metrics['experimental']['folds'].append({
            'fold': fold_data['fold'],
            'train': fold_data['train_metrics'],
            'val': fold_data['val_metrics'],
        })

    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")


def generate_all_plots(exp_results, config, output_dir):
    """
    Generate all plots.

    Args:
        exp_results: Results from direct learning
        config: Configuration dictionary
        output_dir: Output directory
    """
    setup_plot_style(config)

    plots_dir = os.path.join(output_dir, config['outputs']['figures_dir'])
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating plots...")

    # Experimental fold parity plots
    plot_parity_folds(
        exp_results['fold_results'],
        'train',
        os.path.join(plots_dir, 'direct_train_parity.png'),
        config
    )

    plot_parity_folds(
        exp_results['fold_results'],
        'val',
        os.path.join(plots_dir, 'direct_val_parity.png'),
        config
    )

    # Experimental calibration plots
    plot_calibration_folds(
        exp_results['fold_results'],
        'train',
        os.path.join(plots_dir, 'direct_train_calibration.png'),
        config
    )

    plot_calibration_folds(
        exp_results['fold_results'],
        'val',
        os.path.join(plots_dir, 'direct_val_calibration.png'),
        config
    )

    print(f"All plots saved to {plots_dir}")


def main():
    """
    Run final direct learning training and evaluation.
    """
    print("=" * 60)
    print("Direct Learning: Final Training and Evaluation")
    print("=" * 60)

    # Load config
    config = load_config()

    # Get hyperparameters
    hyperparams = get_hyperparameters(config)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create output directory (relative to project root, not script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, config['outputs']['results_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Load experimental data
    print("\n" + "=" * 60)
    exp_df = get_exp_data(config)

    # Build features
    print("\n" + "=" * 60)
    print("Building features...")
    feature_builder = FeatureBuilder(config)
    feature_mode = hyperparams['feature_mode']

    # Fit descriptor scaler on all experimental data
    feature_builder.fit_descriptor_scaler(exp_df)

    # Create 5-fold split for experimental data
    print("\n" + "=" * 60)
    exp_folds_df = create_5fold_polymer_split(exp_df, hyperparams['split_seed'])

    # Build features for experimental folds
    exp_folds_features = []
    for train_df, val_df in exp_folds_df:
        X_train, y_train = feature_builder.build_features(train_df, feature_mode)
        X_val, y_val = feature_builder.build_features(val_df, feature_mode)
        exp_folds_features.append((X_train, y_train, X_val, y_val))

    # Train directly on experimental data
    print("\n" + "=" * 60)
    exp_results = train_on_exp_direct(
        exp_folds_features,
        hyperparams,
        config,
        device
    )

    # Save metrics
    print("\n" + "=" * 60)
    save_metrics(exp_results, output_dir)

    # Generate plots
    print("\n" + "=" * 60)
    generate_all_plots(
        exp_results,
        config,
        output_dir
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)

    print("\nDirect Learning (5-fold CV):")
    print(f"  Train R²: {exp_results['r2_train_mean']:.4f} ± {exp_results['r2_train_std']:.4f}")
    print(f"  Val R²:   {exp_results['r2_val_mean']:.4f} ± {exp_results['r2_val_std']:.4f}")

    print("\n" + "=" * 60)
    print("All results saved to:", output_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
