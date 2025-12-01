"""
Run direct learning with best hyperparameters.

This script:
1. Loads best hyperparameters (or uses defaults)
2. Trains directly on experimental data with 5-fold CV (no transfer learning)
3. Generates all plots
4. Saves all metrics

Key difference from transfer learning: trains from random initialization,
no pretraining on DFT data, no layer freezing.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
import json
from typing import Dict, Any

from data_utils import load_features, get_experiment_folds
from features import filter_and_scale_descriptors, build_features, validate_features
from model import create_model
from train import ChiLoss, train_model, create_dataloader, evaluate_mc_dropout, get_device
from plotting import plot_cv_results, plot_parity_multicolor


def load_best_hyperparameters(best_params_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load best hyperparameters from file, or use defaults if file doesn't exist.

    Args:
        best_params_file: Path to best hyperparameters file
        config: Configuration dictionary

    Returns:
        Dictionary of hyperparameters
    """
    if os.path.exists(best_params_file):
        print(f"Loading best hyperparameters from {best_params_file}")

        hp = {}
        with open(best_params_file, 'r') as f:
            lines = f.readlines()
            in_params = False

            for line in lines:
                line = line.strip()

                if line == "Hyperparameters:":
                    in_params = True
                    continue

                if in_params:
                    if line.startswith("Performance:") or line == "":
                        break

                    # Skip comment lines
                    if line.startswith('#'):
                        continue

                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # Special handling for per-layer dimensions
                        if key.startswith('hidden_dim_layer_'):
                            if 'hidden_dims' not in hp:
                                hp['hidden_dims'] = []
                            hp['hidden_dims'].append(int(value))
                            continue

                        # Parse value
                        try:
                            # Try int
                            hp[key] = int(value)
                        except ValueError:
                            try:
                                # Try float
                                hp[key] = float(value)
                            except ValueError:
                                # String
                                hp[key] = value

        # Backward compatibility: convert single hidden_dim to list
        if 'hidden_dim' in hp and 'hidden_dims' not in hp:
            hp['hidden_dims'] = [hp['hidden_dim']] * hp['n_layers']
            del hp['hidden_dim']
            print(f"Converted hidden_dim to hidden_dims: {hp['hidden_dims']}")

        print(f"Loaded hyperparameters: {hp}")
        return hp

    else:
        print(f"Best hyperparameters file not found: {best_params_file}")
        print("Using default hyperparameters from config")
        return config['defaults']


def save_metrics(metrics: Dict[str, Any], output_file: str):
    """
    Save metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics
        output_file: Output file path
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_type(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_python_type(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_type(item) for item in obj]
        else:
            return obj

    metrics_converted = convert_to_python_type(metrics)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(metrics_converted, f, indent=2)

    print(f"Saved metrics to {output_file}")


def save_metrics_text(metrics: Dict[str, Any], output_file: str):
    """
    Save metrics to a human-readable text file.

    Args:
        metrics: Dictionary of metrics
        output_file: Output file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Direct Learning Results (No Transfer Learning)\n")
        f.write("=" * 80 + "\n\n")

        # Experimental results
        f.write("Experimental Training (5-fold CV, Direct Learning):\n")
        f.write("-" * 80 + "\n")

        f.write("Validation (Out-of-Fold):\n")
        f.write(f"R²:   {metrics['experimental']['val']['r2_mean']:.4f} ± {metrics['experimental']['val']['r2_std']:.4f}\n")
        f.write(f"MAE:  {metrics['experimental']['val']['mae_mean']:.4f} ± {metrics['experimental']['val']['mae_std']:.4f}\n")
        f.write(f"RMSE: {metrics['experimental']['val']['rmse_mean']:.4f} ± {metrics['experimental']['val']['rmse_std']:.4f}\n\n")

        f.write("Training:\n")
        f.write(f"R²:   {metrics['experimental']['train']['r2_mean']:.4f} ± {metrics['experimental']['train']['r2_std']:.4f}\n")
        f.write(f"MAE:  {metrics['experimental']['train']['mae_mean']:.4f} ± {metrics['experimental']['train']['mae_std']:.4f}\n")
        f.write(f"RMSE: {metrics['experimental']['train']['rmse_mean']:.4f} ± {metrics['experimental']['train']['rmse_std']:.4f}\n\n")

        # Per-fold results
        f.write("Per-Fold Results:\n")
        for fold_idx in range(len(metrics['experimental']['folds'])):
            fold_metrics = metrics['experimental']['folds'][fold_idx]
            f.write(f"\nFold {fold_idx}:\n")
            f.write(f"  Val R²:   {fold_metrics['val']['r2']:.4f}\n")
            f.write(f"  Val MAE:  {fold_metrics['val']['mae']:.4f}\n")
            f.write(f"  Val RMSE: {fold_metrics['val']['rmse']:.4f}\n")
            f.write(f"  Train R²: {fold_metrics['train']['r2']:.4f}\n")

    print(f"Saved metrics to {output_file}")


def main():
    """Main function for direct learning."""
    parser = argparse.ArgumentParser(description='Run direct learning with best hyperparameters')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--best-params', type=str, default='hyperparameter_optimization/best_hyperparameters.txt',
                        help='Path to best hyperparameters file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load best hyperparameters
    hp = load_best_hyperparameters(args.best_params, config)

    print("\n" + "=" * 80)
    print("Starting Direct Learning (No Transfer Learning)")
    print("=" * 80)
    print(f"Hyperparameters: {hp}\n")

    # Get device
    device = get_device()

    # Load experimental data
    print("\nLoading experimental data...")
    exp_df = load_features(config['data']['exp_features'])

    # Get 5-fold CV splits
    print("\nCreating 5-fold CV splits...")
    folds = get_experiment_folds(exp_df, split_seed=hp['split_seed'], n_folds=config['global']['n_folds'])

    # =====================
    # Direct Learning (5-fold CV)
    # =====================
    print("\n" + "=" * 80)
    print("Direct Learning (5-fold CV)")
    print("=" * 80)

    # Storage for CV results
    fold_val_results = []
    fold_train_results = []
    fold_histories = []

    all_val_y_true = []
    all_val_y_pred = []
    all_val_y_std = []
    all_val_fold_labels = []

    all_train_y_true = []
    all_train_y_pred = []
    all_train_y_std = []
    all_train_fold_labels = []

    for fold_idx, fold_data in enumerate(folds):
        print(f"\n--- Fold {fold_idx} ---")

        train_df = fold_data['train']
        val_df = fold_data['val']

        # Filter and scale descriptors based on THIS FOLD'S training data
        # (key difference from transfer learning: no DFT-based scaling)
        train_scaled, val_scaled, _, valid_desc_cols, _, _ = \
            filter_and_scale_descriptors(train_df, val_df, None)

        # Build features
        X_train_fold, T_train_fold, y_train_fold = build_features(train_scaled, hp['feature_mode'], valid_desc_cols)
        X_val_fold, T_val_fold, y_val_fold = build_features(val_scaled, hp['feature_mode'], valid_desc_cols)

        print(f"Feature dimension: {X_train_fold.shape[1]}")

        # Create dataloaders
        train_loader_fold = create_dataloader(X_train_fold, T_train_fold, y_train_fold, batch_size=hp['batch_size'], shuffle=True)
        val_loader_fold = create_dataloader(X_val_fold, T_val_fold, y_val_fold, batch_size=hp['batch_size'], shuffle=False)

        # Create model from RANDOM initialization (key difference from transfer learning)
        input_dim = X_train_fold.shape[1]

        # Support both new and legacy formats
        if 'hidden_dims' in hp:
            model_fold = create_model(
                input_dim=input_dim,
                hidden_dims=hp['hidden_dims'],
                n_layers=hp['n_layers'],
                dropout_rate=hp['dropout_rate'],
                device=device
            )
        else:
            # Legacy format
            model_fold = create_model(
                input_dim=input_dim,
                hidden_dim=hp['hidden_dim'],
                n_layers=hp['n_layers'],
                dropout_rate=hp['dropout_rate'],
                device=device
            )

        # Create optimizer and criterion
        optimizer_fold = optim.AdamW(model_fold.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
        criterion_fold = ChiLoss()

        # Train from scratch
        print("Training from scratch...")
        history_fold = train_model(
            model=model_fold,
            train_loader=train_loader_fold,
            val_loader=val_loader_fold,
            optimizer=optimizer_fold,
            criterion=criterion_fold,
            device=device,
            n_epochs=hp['epochs'],
            verbose=False
        )

        fold_histories.append(history_fold)

        # Evaluate with MC dropout
        n_mc_samples = config['global']['mc_dropout_samples']

        train_results_fold = evaluate_mc_dropout(
            model_fold, X_train_fold, T_train_fold, y_train_fold,
            n_samples=n_mc_samples, device=device
        )
        val_results_fold = evaluate_mc_dropout(
            model_fold, X_val_fold, T_val_fold, y_val_fold,
            n_samples=n_mc_samples, device=device
        )

        fold_train_results.append(train_results_fold)
        fold_val_results.append(val_results_fold)

        print(f"Train R²: {train_results_fold['r2']:.4f}, Val R²: {val_results_fold['r2']:.4f}")

        # Collect for combined plots
        all_val_y_true.append(val_results_fold['y_true'])
        all_val_y_pred.append(val_results_fold['chi_mean'])
        all_val_y_std.append(val_results_fold['chi_std'])
        all_val_fold_labels.append(np.full(len(val_results_fold['y_true']), fold_idx))

        all_train_y_true.append(train_results_fold['y_true'])
        all_train_y_pred.append(train_results_fold['chi_mean'])
        all_train_y_std.append(train_results_fold['chi_std'])
        all_train_fold_labels.append(np.full(len(train_results_fold['y_true']), fold_idx))

    # Combine all folds
    all_val_y_true = np.concatenate(all_val_y_true)
    all_val_y_pred = np.concatenate(all_val_y_pred)
    all_val_y_std = np.concatenate(all_val_y_std)
    all_val_fold_labels = np.concatenate(all_val_fold_labels)

    all_train_y_true = np.concatenate(all_train_y_true)
    all_train_y_pred = np.concatenate(all_train_y_pred)
    all_train_y_std = np.concatenate(all_train_y_std)
    all_train_fold_labels = np.concatenate(all_train_fold_labels)

    # Compute aggregate metrics
    val_r2_values = [r['r2'] for r in fold_val_results]
    val_mae_values = [r['mae'] for r in fold_val_results]
    val_rmse_values = [r['rmse'] for r in fold_val_results]

    train_r2_values = [r['r2'] for r in fold_train_results]
    train_mae_values = [r['mae'] for r in fold_train_results]
    train_rmse_values = [r['rmse'] for r in fold_train_results]

    print(f"\nExperimental CV Results:")
    print(f"Val   - R²: {np.mean(val_r2_values):.4f} ± {np.std(val_r2_values):.4f}")
    print(f"Val   - MAE: {np.mean(val_mae_values):.4f} ± {np.std(val_mae_values):.4f}")
    print(f"Val   - RMSE: {np.mean(val_rmse_values):.4f} ± {np.std(val_rmse_values):.4f}")
    print(f"Train - R²: {np.mean(train_r2_values):.4f} ± {np.std(train_r2_values):.4f}")

    # =====================
    # Save Results
    # =====================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # Prepare metrics dictionary
    metrics = {
        'hyperparameters': hp,
        'experimental': {
            'val': {
                'r2_mean': np.mean(val_r2_values),
                'r2_std': np.std(val_r2_values),
                'mae_mean': np.mean(val_mae_values),
                'mae_std': np.std(val_mae_values),
                'rmse_mean': np.mean(val_rmse_values),
                'rmse_std': np.std(val_rmse_values)
            },
            'train': {
                'r2_mean': np.mean(train_r2_values),
                'r2_std': np.std(train_r2_values),
                'mae_mean': np.mean(train_mae_values),
                'mae_std': np.std(train_mae_values),
                'rmse_mean': np.mean(train_rmse_values),
                'rmse_std': np.std(train_rmse_values)
            },
            'folds': [
                {
                    'train': {'r2': fold_train_results[i]['r2'], 'mae': fold_train_results[i]['mae'],
                              'rmse': fold_train_results[i]['rmse']},
                    'val': {'r2': fold_val_results[i]['r2'], 'mae': fold_val_results[i]['mae'],
                            'rmse': fold_val_results[i]['rmse']}
                }
                for i in range(len(folds))
            ]
        }
    }

    # Save metrics
    save_metrics(metrics, os.path.join(config['outputs']['metrics_dir'], 'metrics.json'))
    save_metrics_text(metrics, os.path.join(config['outputs']['metrics_dir'], 'metrics.txt'))

    # Generate plots
    print("\nGenerating plots...")

    # Experimental CV plots
    fold_metrics = [
        {'r2': fold_val_results[i]['r2'], 'mae': fold_val_results[i]['mae'], 'rmse': fold_val_results[i]['rmse']}
        for i in range(len(folds))
    ]

    plot_cv_results(
        all_y_true=all_val_y_true,
        all_y_pred=all_val_y_pred,
        all_y_std=all_val_y_std,
        all_fold_labels=all_val_fold_labels,
        fold_metrics=fold_metrics,
        output_dir=config['outputs']['plots_dir'],
        prefix='exp'
    )

    # Plot experimental CV results (training)
    train_fold_metrics = [
        {'r2': fold_train_results[i]['r2'], 'mae': fold_train_results[i]['mae'], 'rmse': fold_train_results[i]['rmse']}
        for i in range(len(folds))
    ]

    train_agg_metrics = {
        'r2_mean': np.mean(train_r2_values),
        'r2_std': np.std(train_r2_values),
        'mae_mean': np.mean(train_mae_values),
        'mae_std': np.std(train_mae_values),
        'rmse_mean': np.mean(train_rmse_values),
        'rmse_std': np.std(train_rmse_values)
    }

    plot_parity_multicolor(
        y_true=all_train_y_true,
        y_pred=all_train_y_pred,
        fold_labels=all_train_fold_labels,
        metrics=train_agg_metrics,
        save_path=os.path.join(config['outputs']['plots_dir'], 'exp_train_parity.png'),
        title='Out-of-Fold Training'
    )

    print("\n" + "=" * 80)
    print("Direct Learning Completed!")
    print("=" * 80)
    print(f"\nResults saved in {config['outputs']['base_dir']}/")
    print(f"  - Metrics: {config['outputs']['metrics_dir']}/")
    print(f"  - Plots: {config['outputs']['plots_dir']}/")

    print("\nComparison with Transfer Learning:")
    print("  - Transfer learning: Pretrain on DFT → Fine-tune on exp chi")
    print("  - Direct learning: Train from scratch on exp chi only")


if __name__ == '__main__':
    main()
