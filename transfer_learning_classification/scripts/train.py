"""
Training script for transfer learning: chi regression -> binary solubility classification.
Includes both pretraining (regression) and fine-tuning (classification) logic.
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from data_utils import (
    load_precomputed_features, split_dft_data,
    extract_feature_columns, get_feature_indices
)
from models import RegressionModel, ClassificationModel, count_parameters
from utils import (
    set_random_seed, get_device,
    compute_regression_metrics, compute_classification_metrics,
    evaluate_mc_dropout, train_epoch, evaluate_epoch,
    evaluate_classification_epoch, compute_calibration_bins
)
from plotting import (
    plot_training_curves, plot_parity, plot_calibration,
    plot_roc_curve, plot_confusion_matrix,
    plot_probability_histogram, plot_multiple_roc_curves,
    plot_aggregate_classification_curves
)


def pretrain_regression(config: dict, output_dir: str = "outputs/regression") -> dict:
    """
    Pretrain regression model on DFT chi data.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for saving results

    Returns:
        Dictionary with results and paths
    """
    print("\n" + "=" * 80)
    print("PRETRAINING: CHI REGRESSION")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    seed = config['training']['split']['seed']
    set_random_seed(seed)

    # Load precomputed features
    print("\nLoading precomputed DFT features...")
    dft_df = load_precomputed_features(config['data']['dft_features_csv'])

    # Split data
    train_df, val_df, test_df = split_dft_data(
        dft_df,
        train_ratio=config['training']['split']['dft_train_ratio'],
        val_ratio=config['training']['split']['dft_val_ratio'],
        test_ratio=config['training']['split']['dft_test_ratio'],
        seed=seed
    )

    # Extract features
    feature_mode = config['features']['feature_mode']
    print(f"\nUsing feature mode: {feature_mode}")

    X_train, y_train = extract_feature_columns(train_df, feature_mode)
    X_val, y_val = extract_feature_columns(val_df, feature_mode)
    X_test, y_test = extract_feature_columns(test_df, feature_mode)

    # Validate features and targets
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"\nData validation:")
    print(f"  Feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Target (chi) range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  Target (chi) mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

    # Check for NaN/Inf in PRETRAINING - raise error if found
    if not np.isfinite(X_train).all():
        raise ValueError("PRETRAINING: Features contain NaN or Inf. Data must be cleaned before pretraining.")
    if not np.isfinite(y_train).all():
        raise ValueError("PRETRAINING: Target values contain NaN or Inf. Data must be cleaned.")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # Create data loaders
    batch_size = config['training']['pretrain']['batch_size']
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = X_train.shape[1]
    hidden_dims = config['model']['hidden_dims']
    dropout_rate = config['model']['dropout_rate']

    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Dropout: {dropout_rate}")

    model = RegressionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation=config['model']['activation']
    )

    device = get_device()
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Trainable parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['pretrain']['learning_rate'],
        weight_decay=config['training']['pretrain']['weight_decay']
    )

    # Training loop
    n_epochs = config['training']['pretrain']['epochs']
    print(f"\nTraining for {n_epochs} epochs...")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_regression_model.pt')

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(output_dir, 'loss_curves.png'),
        title='Regression Training Curves'
    )

    # Evaluate with MC Dropout
    print("\nEvaluating with MC Dropout...")
    mc_samples = config['mc_dropout']['n_samples']

    # Train set
    train_pred_mean, train_pred_std, train_true, train_metrics = evaluate_mc_dropout(
        model, X_train_t, y_train_t, n_samples=mc_samples, device=device
    )
    print(f"Train - MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")

    # Val set
    val_pred_mean, val_pred_std, val_true, val_metrics = evaluate_mc_dropout(
        model, X_val_t, y_val_t, n_samples=mc_samples, device=device
    )
    print(f"Val   - MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")

    # Test set
    test_pred_mean, test_pred_std, test_true, test_metrics = evaluate_mc_dropout(
        model, X_test_t, y_test_t, n_samples=mc_samples, device=device
    )
    print(f"Test  - MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")

    # Plot parity plots
    parity_size = config['plotting']['parity_size']
    plot_parity(train_true, train_pred_mean, train_pred_std, train_metrics,
                save_path=os.path.join(output_dir, 'parity_train.png'),
                title='Train Parity Plot', figsize=parity_size)

    plot_parity(val_true, val_pred_mean, val_pred_std, val_metrics,
                save_path=os.path.join(output_dir, 'parity_val.png'),
                title='Validation Parity Plot', figsize=parity_size)

    plot_parity(test_true, test_pred_mean, test_pred_std, test_metrics,
                save_path=os.path.join(output_dir, 'parity_test.png'),
                title='Test Parity Plot', figsize=parity_size)

    # Plot calibration
    n_bins_dft = config['plotting']['calibration_bins_dft']

    train_bin_centers, train_bin_errors, _ = compute_calibration_bins(
        train_true, train_pred_mean, train_pred_std, n_bins=n_bins_dft
    )
    plot_calibration(train_bin_centers, train_bin_errors,
                     save_path=os.path.join(output_dir, 'calibration_train.png'),
                     title='Train Uncertainty Calibration')

    val_bin_centers, val_bin_errors, _ = compute_calibration_bins(
        val_true, val_pred_mean, val_pred_std, n_bins=n_bins_dft
    )
    plot_calibration(val_bin_centers, val_bin_errors,
                     save_path=os.path.join(output_dir, 'calibration_val.png'),
                     title='Validation Uncertainty Calibration')

    test_bin_centers, test_bin_errors, _ = compute_calibration_bins(
        test_true, test_pred_mean, test_pred_std, n_bins=n_bins_dft
    )
    plot_calibration(test_bin_centers, test_bin_errors,
                     save_path=os.path.join(output_dir, 'calibration_test.png'),
                     title='Test Uncertainty Calibration')

    # Save metrics
    metrics_dict = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    print("\n" + "=" * 80)
    print("PRETRAINING COMPLETE")
    print("=" * 80)

    return {
        'model_path': best_model_path,
        'metrics': metrics_dict
    }


def finetune_classification(
    config: dict,
    pretrained_model_path: str,
    output_dir: str = "outputs/classification"
) -> dict:
    """
    Fine-tune classification model on binary solubility data using 5-fold CV.

    Args:
        config: Configuration dictionary
        pretrained_model_path: Path to pretrained regression model
        output_dir: Output directory for saving results

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 80)
    print("FINE-TUNING: BINARY SOLUBILITY CLASSIFICATION")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    seed = config['training']['split']['seed']
    set_random_seed(seed)

    # Load precomputed features
    print("\nLoading precomputed binary solubility features...")
    binary_df = load_precomputed_features(config['data']['binary_features_csv'])

    # Extract features
    feature_mode = config['features']['feature_mode']
    print(f"Using feature mode: {feature_mode}")

    X, y = extract_feature_columns(binary_df, feature_mode)
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Samples: {len(X)}")

    # For FINE-TUNING: Replace NaN/Inf with 0
    if not np.isfinite(X).all():
        num_nan_inf = np.sum(~np.isfinite(X))
        print(f"WARNING: Found {num_nan_inf} NaN/Inf values in fine-tuning features. Setting to 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to PyTorch tensors
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)

    # Stratified K-Fold cross-validation
    n_folds = config['training']['split']['cv_folds']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    print(f"\nPerforming {n_folds}-fold stratified cross-validation...")

    # Storage for results
    fold_results = []
    all_train_losses = []
    all_val_losses = []
    all_val_proba = []
    all_val_true = []
    all_train_proba = []
    all_train_true = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*80}")

        # Split data
        X_train_fold = X_t[train_idx]
        y_train_fold = y_t[train_idx]
        X_val_fold = X_t[val_idx]
        y_val_fold = y_t[val_idx]

        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Create data loaders
        batch_size = config['training']['finetune']['batch_size']
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        input_dim = X.shape[1]
        hidden_dims = config['model']['hidden_dims']
        dropout_rate = config['model']['dropout_rate']
        n_freeze_layers = config['training']['finetune']['n_freeze_layers']

        # Create regression model to load pretrained weights
        from models import RegressionModel, Encoder
        reg_model = RegressionModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=config['model']['activation']
        )
        reg_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))

        # Get pretrained encoder
        pretrained_encoder = reg_model.get_encoder()

        # Create classification model with pretrained encoder
        model = ClassificationModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=config['model']['activation'],
            pretrained_encoder=pretrained_encoder,
            n_freeze_layers=n_freeze_layers
        )

        device = get_device()
        model = model.to(device)
        print(f"Device: {device}")
        print(f"Trainable parameters: {count_parameters(model):,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['finetune']['learning_rate'],
            weight_decay=config['training']['finetune']['weight_decay']
        )

        # Training loop
        n_epochs = config['training']['finetune']['epochs']
        print(f"Training for {n_epochs} epochs...")

        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0

        for epoch in range(n_epochs):
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)

            # Validate
            val_loss, val_proba, val_true, val_acc = evaluate_classification_epoch(
                model, val_loader, criterion, device
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")

        print(f"Best validation accuracy: {best_val_acc:.4f}")

        # Store losses for aggregate plotting
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Validation set
            val_logits = model(X_val_fold.to(device))
            val_proba_final = torch.sigmoid(val_logits).cpu().numpy().squeeze()
            val_pred_final = (val_proba_final >= 0.5).astype(int)
            val_true_final = y_val_fold.cpu().numpy().squeeze()

            # Training set
            train_logits = model(X_train_fold.to(device))
            train_proba_final = torch.sigmoid(train_logits).cpu().numpy().squeeze()
            train_pred_final = (train_proba_final >= 0.5).astype(int)
            train_true_final = y_train_fold.cpu().numpy().squeeze()

        # Compute metrics
        train_metrics = compute_classification_metrics(
            train_true_final, train_pred_final, train_proba_final
        )
        val_metrics = compute_classification_metrics(
            val_true_final, val_pred_final, val_proba_final
        )

        print(f"\nFold {fold + 1} Train Metrics:")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"  F1: {train_metrics['f1']:.4f}")

        print(f"\nFold {fold + 1} Validation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")

        # Store results
        fold_results.append({
            'fold': fold + 1,
            'train_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                             for k, v in train_metrics.items()},
            'val_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                           for k, v in val_metrics.items()}
        })

        # Store out-of-fold predictions
        all_val_proba.extend(val_proba_final)
        all_val_true.extend(val_true_final)
        all_train_proba.extend(train_proba_final)
        all_train_true.extend(train_true_final)

        # Plot ROC for this fold
        plot_roc_curve(
            val_true_final, val_proba_final,
            save_path=os.path.join(output_dir, f'roc_fold{fold+1}.png'),
            label=f'Fold {fold+1}',
            title=f'ROC Curve - Fold {fold+1}'
        )

        # Plot confusion matrix for this fold
        plot_confusion_matrix(
            val_metrics['confusion_matrix'],
            save_path=os.path.join(output_dir, f'confusion_fold{fold+1}.png'),
            title=f'Confusion Matrix - Fold {fold+1}'
        )

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS FOLDS")
    print("=" * 80)

    # Compute mean and std of metrics
    train_accs = [r['train_metrics']['accuracy'] for r in fold_results]
    val_accs = [r['val_metrics']['accuracy'] for r in fold_results]
    val_f1s = [r['val_metrics']['f1'] for r in fold_results]
    val_aucs = [r['val_metrics']['roc_auc'] for r in fold_results]

    print(f"\nTrain Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Validation F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Validation ROC-AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")

    # Convert to arrays
    all_val_proba = np.array(all_val_proba)
    all_val_true = np.array(all_val_true)
    all_train_proba = np.array(all_train_proba)
    all_train_true = np.array(all_train_true)

    # Plot aggregate ROC curve
    plot_roc_curve(
        all_val_true, all_val_proba,
        save_path=os.path.join(output_dir, 'roc_aggregate_val.png'),
        label='Aggregate (Val)',
        title='Aggregate ROC Curve (All Validation Folds)'
    )

    plot_roc_curve(
        all_train_true, all_train_proba,
        save_path=os.path.join(output_dir, 'roc_aggregate_train.png'),
        label='Aggregate (Train)',
        title='Aggregate ROC Curve (All Training Folds)'
    )

    # Plot aggregate confusion matrix
    all_val_pred = (all_val_proba >= 0.5).astype(int)
    cm_aggregate = confusion_matrix(all_val_true, all_val_pred)
    plot_confusion_matrix(
        cm_aggregate,
        save_path=os.path.join(output_dir, 'confusion_aggregate.png'),
        title='Aggregate Confusion Matrix'
    )

    # Plot probability histogram
    plot_probability_histogram(
        all_val_true, all_val_proba,
        save_path=os.path.join(output_dir, 'probability_histogram.png'),
        title='Predicted Probability Distribution'
    )

    # Plot aggregate training curves
    plot_aggregate_classification_curves(
        all_train_losses, all_val_losses,
        save_path=os.path.join(output_dir, 'loss_curves_aggregate.png'),
        title='Aggregate Classification Training Curves'
    )

    # Save results
    results_dict = {
        'fold_results': fold_results,
        'aggregate': {
            'train_accuracy_mean': float(np.mean(train_accs)),
            'train_accuracy_std': float(np.std(train_accs)),
            'val_accuracy_mean': float(np.mean(val_accs)),
            'val_accuracy_std': float(np.std(val_accs)),
            'val_f1_mean': float(np.mean(val_f1s)),
            'val_f1_std': float(np.std(val_f1s)),
            'val_roc_auc_mean': float(np.mean(val_aucs)),
            'val_roc_auc_std': float(np.std(val_aucs))
        }
    }

    results_path = os.path.join(output_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE")
    print("=" * 80)

    return results_dict


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Transfer Learning Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--pretrain_only', action='store_true',
                        help='Only run pretraining')
    parser.add_argument('--finetune_only', action='store_true',
                        help='Only run fine-tuning')
    parser.add_argument('--pretrained_model', type=str,
                        default='outputs/regression/best_regression_model.pt',
                        help='Path to pretrained model (for fine-tuning only)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.finetune_only:
        # Only fine-tune
        finetune_classification(
            config,
            pretrained_model_path=args.pretrained_model,
            output_dir=config['outputs']['classification_dir']
        )
    elif args.pretrain_only:
        # Only pretrain
        pretrain_regression(
            config,
            output_dir=config['outputs']['regression_dir']
        )
    else:
        # Run both
        pretrain_results = pretrain_regression(
            config,
            output_dir=config['outputs']['regression_dir']
        )

        finetune_classification(
            config,
            pretrained_model_path=pretrain_results['model_path'],
            output_dir=config['outputs']['classification_dir']
        )


if __name__ == '__main__':
    main()
