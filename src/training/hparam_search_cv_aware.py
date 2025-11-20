"""
CV-Aware Hyperparameter Search for Strategy 2 (CV Single-Task).

Optimizes Stage 1 hyperparameters based on their transferability to CV exp chi fine-tuning.
Uses Optuna with TPE sampler for efficient search.

Usage:
    python -m src.training.hparam_search_cv_aware \
        --config configs/config_hparam_search.yaml \
        --n_trials 50 \
        --timeout_hours 48
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.datasets import ExpChiDataset, collate_exp_chi
from src.data.featurization import PolymerFeaturizer
from src.data.splits import create_exp_chi_cv_splits
from src.models.multitask_model import MultiTaskChiSolubilityModel
from src.training.losses import chi_exp_loss, compute_metrics_regression
from src.training.train_dft_pretrain import train_dft
from src.training.optuna_utils import (
    create_or_load_study,
    save_complete_results,
    print_study_summary,
)
from src.utils.config import load_config, save_config
from src.utils.logging_utils import get_logger
from src.utils.seed_utils import set_seed, worker_init_fn


class CVAwareObjective:
    """
    Objective function for CV-aware hyperparameter optimization (Strategy 2).

    Samples Stage 1 hyperparameters, trains Stage 1 model, runs K-fold CV
    on exp chi ONLY, and returns weighted objective based on CV metrics.
    """

    def __init__(
        self,
        base_config_path: Path,
        results_dir: Path,
        n_cv_folds: int = 5,
        cv_epochs: int = 20,
        verbose: bool = True,
    ):
        """
        Initialize objective.

        Args:
            base_config_path: Path to base config file
            results_dir: Directory to save results
            n_cv_folds: Number of CV folds
            cv_epochs: Number of epochs for CV training
            verbose: If True, print detailed logs
        """
        self.base_config = load_config(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_cv_folds = n_cv_folds
        self.cv_epochs = cv_epochs
        self.verbose = verbose

        # Setup logging
        self.logger = get_logger("CVAwareSearch")

        # Set device
        self.device = torch.device(
            self.base_config.training.device if torch.cuda.is_available() else "cpu"
        )

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (lower is better)
        """
        trial_number = trial.number
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRIAL {trial_number} STARTED")
        self.logger.info(f"{'='*80}\n")

        # ====================================================================
        # Sample Stage 1 Hyperparameters
        # ====================================================================
        stage1_params = self._sample_stage1_params(trial)

        if self.verbose:
            self.logger.info("Stage 1 Hyperparameters:")
            for key, value in stage1_params.items():
                self.logger.info(f"  {key}: {value}")

        # ====================================================================
        # Train Stage 1
        # ====================================================================
        self.logger.info("\n[Step 1/3] Training Stage 1 model...")

        stage1_config = self._create_stage1_config(stage1_params, trial_number)
        stage1_results = self._train_stage1(stage1_config)

        if stage1_results is None:
            self.logger.error("Stage 1 training failed")
            return 1000.0  # Penalize failed trials

        stage1_val_mae = stage1_results['val_mae']
        stage1_checkpoint = stage1_results['checkpoint_path']

        self.logger.info(f"Stage 1 Val MAE: {stage1_val_mae:.4f}")

        # Early rejection if Stage 1 is terrible
        if stage1_val_mae > 0.25:
            self.logger.warning(f"Stage 1 validation MAE too high: {stage1_val_mae:.4f}")
            return 1000.0

        # ====================================================================
        # Run K-Fold CV on Exp Chi Only
        # ====================================================================
        self.logger.info(f"\n[Step 2/3] Running {self.n_cv_folds}-fold CV on exp chi...")

        cv_metrics = self._run_cv_exp_chi(stage1_checkpoint, trial_number)

        if cv_metrics is None:
            self.logger.error("CV fine-tuning failed")
            return 1000.0

        # ====================================================================
        # Compute CV-Aware Objective
        # ====================================================================
        self.logger.info("\n[Step 3/3] Computing CV-aware objective...")

        objective = self._compute_objective(stage1_val_mae, cv_metrics)

        # Log results
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRIAL {trial_number} RESULTS")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Stage 1 Val MAE: {stage1_val_mae:.4f}")
        self.logger.info(f"Exp Chi MAE: {cv_metrics['mean_mae']:.4f} ± {cv_metrics['std_mae']:.4f}")
        self.logger.info(f"Exp Chi RMSE: {cv_metrics['mean_rmse']:.4f} ± {cv_metrics['std_rmse']:.4f}")
        self.logger.info(f"Exp Chi R²: {cv_metrics['mean_r2']:.4f} ± {cv_metrics['std_r2']:.4f}")
        self.logger.info(f"OBJECTIVE: {objective:.4f}")
        self.logger.info(f"{'='*80}\n")

        return objective

    def _sample_stage1_params(self, trial: optuna.Trial) -> Dict:
        """Sample Stage 1 hyperparameters (same as Strategy 1)."""
        return {
            'encoder_latent_dim': trial.suggest_categorical('latent_dim', [64, 128, 256]),
            'encoder_dropout': trial.suggest_float('encoder_dropout', 0.15, 0.35),
            'chi_head_dropout': trial.suggest_float('chi_head_dropout', 0.05, 0.25),
            'encoder_hidden_dims': trial.suggest_categorical(
                'encoder_hidden',
                ['512_256', '1024_512_256', '256_128']
            ),
            'lr_pretrain': trial.suggest_float('lr_pretrain', 5e-4, 2e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        }

    def _create_stage1_config(self, params: Dict, trial_number: int):
        """Create Stage 1 config with sampled hyperparameters."""
        config = self.base_config.copy()

        # Update model hyperparameters
        config.model.encoder_latent_dim = params['encoder_latent_dim']
        config.model.encoder_dropout = params['encoder_dropout']
        config.model.chi_head_dropout = params['chi_head_dropout']

        # Parse hidden dims
        hidden_str = params['encoder_hidden_dims']
        config.model.encoder_hidden_dims = [int(x) for x in hidden_str.split('_')]

        # Update training hyperparameters
        config.training.lr_pretrain = params['lr_pretrain']
        config.training.weight_decay = params['weight_decay']

        # Set output directory for this trial
        config.paths.results_dir = str(self.results_dir / f"trial_{trial_number}" / "stage1")

        return config

    def _train_stage1(self, config):
        """Train Stage 1 model."""
        try:
            results = train_dft(config)
            return results

        except Exception as e:
            self.logger.error(f"Stage 1 training error: {e}")
            return None

    def _run_cv_exp_chi(self, stage1_checkpoint: Path, trial_number: int) -> Dict:
        """
        Run K-fold CV on exp chi data only.

        Args:
            stage1_checkpoint: Path to Stage 1 pretrained checkpoint
            trial_number: Trial number for directory naming

        Returns:
            Dictionary with mean and std of CV metrics
        """
        try:
            # Load experimental chi data
            exp_csv_path = Path(self.base_config.paths.exp_chi_csv)
            df_exp = pd.read_csv(exp_csv_path)

            # Featurize
            unique_smiles = df_exp["SMILES"].unique().tolist()
            featurizer = PolymerFeaturizer(self.base_config)
            features, smiles_to_idx = featurizer.featurize(unique_smiles)
            feature_dim = featurizer.get_feature_dim()

            # Create full dataset
            full_dataset = ExpChiDataset(df_exp, features, smiles_to_idx)

            # Create CV splits
            cv_splits = create_exp_chi_cv_splits(df_exp, self.base_config)

            # Run CV
            fold_results = []

            for fold_idx, (train_indices, val_indices) in enumerate(cv_splits[:self.n_cv_folds]):
                if self.verbose:
                    self.logger.info(f"  Fold {fold_idx+1}/{self.n_cv_folds}...")

                # Create fold datasets
                train_subset = Subset(full_dataset, train_indices)
                val_subset = Subset(full_dataset, val_indices)

                # Create dataloaders
                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.base_config.training.batch_size_exp,
                    shuffle=True,
                    num_workers=0,  # Avoid multiprocessing issues in Optuna
                    collate_fn=collate_exp_chi,
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=self.base_config.training.batch_size_exp,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_exp_chi,
                )

                # Train fold
                metrics = self._train_single_fold(
                    feature_dim=feature_dim,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    pretrained_path=stage1_checkpoint,
                    fold=fold_idx,
                )

                fold_results.append(metrics)

            # Compute statistics
            return {
                'mean_mae': np.mean([f['mae'] for f in fold_results]),
                'std_mae': np.std([f['mae'] for f in fold_results]),
                'mean_rmse': np.mean([f['rmse'] for f in fold_results]),
                'std_rmse': np.std([f['rmse'] for f in fold_results]),
                'mean_r2': np.mean([f['r2'] for f in fold_results]),
                'std_r2': np.std([f['r2'] for f in fold_results]),
            }

        except Exception as e:
            self.logger.error(f"CV exp chi error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _train_single_fold(
        self,
        feature_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pretrained_path: Path,
        fold: int,
    ) -> Dict:
        """
        Train a single CV fold.

        Args:
            feature_dim: Feature dimensionality
            train_loader: Training dataloader
            val_loader: Validation dataloader
            pretrained_path: Path to Stage 1 checkpoint
            fold: Fold number

        Returns:
            Dictionary of validation metrics
        """
        # Build model
        model = MultiTaskChiSolubilityModel(feature_dim, self.base_config)

        # Load pretrained weights
        model.load_encoder_and_chi_head(pretrained_path)

        model = model.to(self.device)

        # Setup optimizer
        lr = self.base_config.training.lr_finetune
        weight_decay = self.base_config.training.weight_decay

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop
        best_val_loss = float('inf')
        best_val_metrics = None
        patience_counter = 0
        early_stopping_patience = 10

        for epoch in range(1, self.cv_epochs + 1):
            # Train
            model.train()
            for batch in train_loader:
                x = batch["x"].to(self.device)
                chi_true = batch["chi_exp"].to(self.device)
                temp = batch["temperature"].to(self.device)

                # Forward pass
                outputs = model(x, temperature=temp)
                A, B = outputs["A"], outputs["B"]

                # Compute loss
                loss = chi_exp_loss(A, B, chi_true, temp)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.base_config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.base_config.training.grad_clip_norm
                    )

                optimizer.step()

            # Validate
            model.eval()
            val_preds, val_targets = [], []
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(self.device)
                    chi_true = batch["chi_exp"].to(self.device)
                    temp = batch["temperature"].to(self.device)

                    outputs = model(x, temperature=temp)
                    A, B = outputs["A"], outputs["B"]
                    chi_pred = outputs["chi"]

                    loss = chi_exp_loss(A, B, chi_true, temp)
                    val_loss += loss.item()

                    val_preds.append(chi_pred.cpu())
                    val_targets.append(chi_true.cpu())

            avg_val_loss = val_loss / len(val_loader)
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)

            val_metrics = compute_metrics_regression(val_preds, val_targets)

            # Check for best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                break

        return best_val_metrics

    def _compute_objective(self, stage1_val_mae: float, cv_metrics: Dict) -> float:
        """
        Compute CV-aware objective for Strategy 2.

        Objective = mean_mae + β × std_mae + δ × stage1_mae

        Where:
        - mean_mae: Main metric (minimize)
        - β = 0.2 (penalize high variance across folds)
        - δ = 0.05 (small penalty for bad Stage 1)
        """
        mean_mae = cv_metrics['mean_mae']
        std_mae = cv_metrics['std_mae']

        objective = (
            mean_mae +                    # Minimize mean MAE
            0.2 * std_mae +              # Penalize variance (20%)
            0.05 * stage1_val_mae        # Small Stage 1 penalty
        )

        return objective


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CV-aware hyperparameter search (Strategy 2)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_hparam_search.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--timeout_hours",
        type=float,
        default=48.0,
        help="Timeout in hours",
    )
    parser.add_argument(
        "--n_cv_folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--cv_epochs",
        type=int,
        default=20,
        help="Number of epochs for CV training",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Study name (default: cv_aware_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to study database to resume",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"cv_aware_{timestamp}"
    results_dir = Path("results") / "hparam_search" / study_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create study
    storage_path = results_dir / "study.db" if args.resume is None else Path(args.resume)
    study = create_or_load_study(
        study_name=study_name,
        storage_path=storage_path,
        direction="minimize",
    )

    # Create objective
    objective = CVAwareObjective(
        base_config_path=Path(args.config),
        results_dir=results_dir,
        n_cv_folds=args.n_cv_folds,
        cv_epochs=args.cv_epochs,
        verbose=True,
    )

    # Run optimization
    print(f"\n{'='*80}")
    print(f"STARTING CV-AWARE HYPERPARAMETER SEARCH (STRATEGY 2)")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Results directory: {results_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Timeout: {args.timeout_hours} hours")
    print(f"CV folds: {args.n_cv_folds}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"{'='*80}\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_hours * 3600,  # Convert to seconds
        n_jobs=1,  # Sequential execution (GPU constraint)
    )

    # Save results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")

    print_study_summary(study)
    save_complete_results(study, results_dir, prefix="cv_")

    print(f"\n✅ All results saved to {results_dir}")
    print(f"\nBest Stage 1 hyperparameters for CV exp chi transfer:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
