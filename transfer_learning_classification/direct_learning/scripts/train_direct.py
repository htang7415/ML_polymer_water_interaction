"""
Direct learning training script for binary solubility classification.
Train from scratch without transfer learning (no pretraining stage).

This script trains a classification model directly on binary solubility data
using random weight initialization, enabling comparison with transfer learning.
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# Import from parent scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from data_utils import load_precomputed_features, extract_feature_columns
from models import ClassificationModel, count_parameters
from utils import (
    set_random_seed, get_device,
    compute_classification_metrics,
    train_epoch, evaluate_classification_epoch
)
from plotting import (
    plot_training_curves, plot_roc_curve, plot_confusion_matrix,
    plot_probability_histogram, plot_aggregate_classification_curves
)


def train_classification_direct(
    config: dict,
    output_dir: str = "outputs"
) -> dict:
    """
    Train classification model directly on binary solubility data (no pretraining).
    Uses 5-fold stratified cross-validation.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for saving results

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 80)
    print("DIRECT LEARNING: BINARY SOLUBILITY CLASSIFICATION (NO TRANSFER LEARNING)")
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
    print(f"  Soluble (1): {(y == 1).sum()}")
    print(f"  Insoluble (0): {(y == 0).sum()}")

    # Handle NaN/Inf in features
    if not np.isfinite(X).all():
        num_nan_inf = np.sum(~np.isfinite(X))
        print(f"WARNING: Found {num_nan_inf} NaN/Inf values in features. Setting to 0.")
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
        batch_size = config['training']['direct']['batch_size']
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model FROM SCRATCH (no pretrained encoder)
        input_dim = X.shape[1]
        hidden_dims = config['model']['hidden_dims']
        dropout_rate = config['model']['dropout_rate']

        model = ClassificationModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=config['model']['activation'],
            pretrained_encoder=None,  # KEY DIFFERENCE: Random initialization
            n_freeze_layers=0
        )

        device = get_device()
        model = model.to(device)
        print(f"Device: {device}")
        print(f"Model initialized from scratch (no pretraining)")
        print(f"Trainable parameters: {count_parameters(model):,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['direct']['learning_rate'],
            weight_decay=config['training']['direct']['weight_decay']
        )

        # Training loop
        n_epochs = config['training']['direct']['epochs']
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
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")

        print(f"\nFold {fold + 1} Validation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")

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
    train_f1s = [r['train_metrics']['f1'] for r in fold_results]
    train_aucs = [r['train_metrics']['roc_auc'] for r in fold_results]
    val_accs = [r['val_metrics']['accuracy'] for r in fold_results]
    val_f1s = [r['val_metrics']['f1'] for r in fold_results]
    val_aucs = [r['val_metrics']['roc_auc'] for r in fold_results]
    val_precisions = [r['val_metrics']['precision'] for r in fold_results]
    val_recalls = [r['val_metrics']['recall'] for r in fold_results]

    print(f"\nTrain Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Train F1: {np.mean(train_f1s):.4f} ± {np.std(train_f1s):.4f}")
    print(f"Train ROC-AUC: {np.mean(train_aucs):.4f} ± {np.std(train_aucs):.4f}")
    print(f"\nValidation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Validation F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Validation ROC-AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"Validation Precision: {np.mean(val_precisions):.4f} ± {np.std(val_precisions):.4f}")
    print(f"Validation Recall: {np.mean(val_recalls):.4f} ± {np.std(val_recalls):.4f}")

    # Convert to arrays
    all_val_proba = np.array(all_val_proba)
    all_val_true = np.array(all_val_true)
    all_train_proba = np.array(all_train_proba)
    all_train_true = np.array(all_train_true)

    # Compute aggregate metrics
    all_val_pred = (all_val_proba >= 0.5).astype(int)
    aggregate_val_metrics = compute_classification_metrics(
        all_val_true, all_val_pred, all_val_proba
    )

    all_train_pred = (all_train_proba >= 0.5).astype(int)
    aggregate_train_metrics = compute_classification_metrics(
        all_train_true, all_train_pred, all_train_proba
    )

    print(f"\nAggregate Train Metrics (all folds combined):")
    print(f"  Accuracy: {aggregate_train_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {aggregate_train_metrics['roc_auc']:.4f}")
    print(f"  F1: {aggregate_train_metrics['f1']:.4f}")

    print(f"\nAggregate Validation Metrics (all folds combined):")
    print(f"  Accuracy: {aggregate_val_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {aggregate_val_metrics['roc_auc']:.4f}")
    print(f"  F1: {aggregate_val_metrics['f1']:.4f}")

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
        title='Aggregate Direct Learning Training Curves'
    )

    # Save results
    results_dict = {
        'approach': 'direct_learning',
        'description': 'Classification model trained from scratch (no pretraining)',
        'fold_results': fold_results,
        'aggregate': {
            'train_accuracy_mean': float(np.mean(train_accs)),
            'train_accuracy_std': float(np.std(train_accs)),
            'train_f1_mean': float(np.mean(train_f1s)),
            'train_f1_std': float(np.std(train_f1s)),
            'train_roc_auc_mean': float(np.mean(train_aucs)),
            'train_roc_auc_std': float(np.std(train_aucs)),
            'val_accuracy_mean': float(np.mean(val_accs)),
            'val_accuracy_std': float(np.std(val_accs)),
            'val_f1_mean': float(np.mean(val_f1s)),
            'val_f1_std': float(np.std(val_f1s)),
            'val_roc_auc_mean': float(np.mean(val_aucs)),
            'val_roc_auc_std': float(np.std(val_aucs)),
            'val_precision_mean': float(np.mean(val_precisions)),
            'val_precision_std': float(np.std(val_precisions)),
            'val_recall_mean': float(np.mean(val_recalls)),
            'val_recall_std': float(np.std(val_recalls)),
            'aggregate_train_accuracy': float(aggregate_train_metrics['accuracy']),
            'aggregate_train_f1': float(aggregate_train_metrics['f1']),
            'aggregate_train_roc_auc': float(aggregate_train_metrics['roc_auc']),
            'aggregate_val_accuracy': float(aggregate_val_metrics['accuracy']),
            'aggregate_val_f1': float(aggregate_val_metrics['f1']),
            'aggregate_val_roc_auc': float(aggregate_val_metrics['roc_auc'])
        }
    }

    results_path = os.path.join(output_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 80)
    print("DIRECT LEARNING COMPLETE")
    print("=" * 80)
    print("\nTo compare with transfer learning, see:")
    print("  Transfer learning: ../outputs/classification/metrics.json")
    print(f"  Direct learning: {results_path}")

    return results_dict


def main():
    """Main entry point for direct learning training."""
    parser = argparse.ArgumentParser(description='Direct Learning Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run direct learning
    train_classification_direct(
        config,
        output_dir=config['outputs']['base_dir']
    )


if __name__ == '__main__':
    main()
