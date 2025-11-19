"""
Transfer-Aware Hyperparameter Search for Stage 1.

Optimizes Stage 1 hyperparameters based on their transferability to Stage 2.
Uses Optuna with TPE sampler for efficient search.

Usage:
    python -m src.training.hparam_search_transfer_aware \
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
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_dft_pretrain import main as train_dft_main
from src.training.train_multitask_quick import train_multitask_quick
from src.training.optuna_utils import (
    create_or_load_study,
    save_complete_results,
    print_study_summary,
)
from src.utils.config import load_config, save_config
from src.utils.logging_utils import get_logger


class TransferAwareObjective:
    """
    Objective function for transfer-aware hyperparameter optimization.

    Samples Stage 1 hyperparameters, trains Stage 1 model, quickly fine-tunes
    on Stage 2 with CV, and returns weighted objective based on Stage 2 metrics.
    """

    def __init__(
        self,
        base_config_path: Path,
        results_dir: Path,
        n_cv_folds: int = 5,
        stage2_epochs: int = 20,
        verbose: bool = True,
    ):
        """
        Initialize objective.

        Args:
            base_config_path: Path to base config file
            results_dir: Directory to save results
            n_cv_folds: Number of CV folds for Stage 2
            stage2_epochs: Number of epochs for quick Stage 2 training
            verbose: If True, print detailed logs
        """
        self.base_config = load_config(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_cv_folds = n_cv_folds
        self.stage2_epochs = stage2_epochs
        self.verbose = verbose

        # Setup logging
        self.logger = get_logger("TransferAwareSearch")

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
        # Quick Fine-tune Stage 2 with Cross-Validation
        # ====================================================================
        self.logger.info(f"\n[Step 2/3] Quick fine-tuning Stage 2 ({self.n_cv_folds}-fold CV)...")

        stage2_config = self._create_stage2_config(trial_number)
        stage2_cv_metrics = self._finetune_stage2_cv(stage2_config, stage1_checkpoint)

        if stage2_cv_metrics is None:
            self.logger.error("Stage 2 fine-tuning failed")
            return 1000.0

        # ====================================================================
        # Compute Transfer Objective
        # ====================================================================
        self.logger.info("\n[Step 3/3] Computing transfer objective...")

        objective = self._compute_objective(stage1_val_mae, stage2_cv_metrics)

        # Log results
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRIAL {trial_number} RESULTS")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Stage 1 Val MAE: {stage1_val_mae:.4f}")
        self.logger.info(f"Stage 2 Exp MAE: {stage2_cv_metrics['exp_mae']:.4f} ± {stage2_cv_metrics['exp_mae_std']:.4f}")
        self.logger.info(f"Stage 2 Sol F1: {stage2_cv_metrics['sol_f1']:.4f} ± {stage2_cv_metrics['sol_f1_std']:.4f}")
        self.logger.info(f"Stage 2 Sol Recall: {stage2_cv_metrics['sol_recall']:.4f} ± {stage2_cv_metrics['sol_recall_std']:.4f}")
        self.logger.info(f"OBJECTIVE: {objective:.4f}")
        self.logger.info(f"{'='*80}\n")

        return objective

    def _sample_stage1_params(self, trial: optuna.Trial) -> Dict:
        """Sample Stage 1 hyperparameters."""
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

    def _create_stage2_config(self, trial_number: int):
        """Create Stage 2 config (fixed hyperparameters for fair comparison)."""
        config = self.base_config.copy()

        # Fixed Stage 2 hyperparameters
        config.training.freeze_encoder_stage2 = True
        config.training.use_discriminative_lr = True
        config.training.lr_encoder = 1e-5
        config.training.lr_finetune = 3e-4
        config.training.num_epochs_quick = self.stage2_epochs
        config.training.quick_patience = 10

        config.loss_weights.lambda_dft = 0.0
        config.loss_weights.lambda_exp = 3.0
        config.loss_weights.lambda_sol = 1.0

        config.solubility.class_weight_pos = 5.0
        config.solubility.optimize_threshold = True

        # Set output directory
        config.paths.results_dir = str(self.results_dir / f"trial_{trial_number}" / "stage2")

        return config

    def _train_stage1(self, config):
        """Train Stage 1 model."""
        try:
            # Import and run Stage 1 training
            # Note: This assumes train_dft_pretrain.py has a main() function that returns results
            # You may need to adapt this based on your actual Stage 1 training script

            from src.training.train_dft_pretrain import train_dft

            results = train_dft(config)
            return results

        except Exception as e:
            self.logger.error(f"Stage 1 training error: {e}")
            return None

    def _finetune_stage2_cv(self, config, stage1_checkpoint: Path) -> Dict:
        """Fine-tune Stage 2 with cross-validation."""
        try:
            cv_results = {
                'exp_mae': [],
                'exp_r2': [],
                'sol_f1': [],
                'sol_recall': [],
                'sol_precision': [],
            }

            for fold in range(self.n_cv_folds):
                if self.verbose:
                    self.logger.info(f"  Fold {fold+1}/{self.n_cv_folds}...")

                metrics = train_multitask_quick(
                    config=config,
                    pretrained_path=stage1_checkpoint,
                    cv_fold=fold,
                    verbose=False,
                )

                cv_results['exp_mae'].append(metrics['exp_mae'])
                cv_results['exp_r2'].append(metrics['exp_r2'])
                cv_results['sol_f1'].append(metrics['sol_f1'])
                cv_results['sol_recall'].append(metrics['sol_recall'])
                cv_results['sol_precision'].append(metrics['sol_precision'])

            # Compute means and stds
            return {
                'exp_mae': np.mean(cv_results['exp_mae']),
                'exp_mae_std': np.std(cv_results['exp_mae']),
                'exp_r2': np.mean(cv_results['exp_r2']),
                'sol_f1': np.mean(cv_results['sol_f1']),
                'sol_f1_std': np.std(cv_results['sol_f1']),
                'sol_recall': np.mean(cv_results['sol_recall']),
                'sol_recall_std': np.std(cv_results['sol_recall']),
                'sol_precision': np.mean(cv_results['sol_precision']),
            }

        except Exception as e:
            self.logger.error(f"Stage 2 fine-tuning error: {e}")
            return None

    def _compute_objective(self, stage1_val_mae: float, stage2_metrics: Dict) -> float:
        """
        Compute weighted objective focusing on Stage 2 performance.

        Objective = α × exp_mae + β × (1 - sol_f1) + γ × (1 - sol_recall) + δ × stage1_mae

        Where:
        - α = 1.5 (exp chi is critical)
        - β = 2.0 (solubility F1 is important)
        - γ = 1.5 (solubility recall is important)
        - δ = 0.1 (small penalty for bad Stage 1)
        """
        exp_mae = stage2_metrics['exp_mae']
        sol_f1 = stage2_metrics['sol_f1']
        sol_recall = stage2_metrics['sol_recall']

        objective = (
            1.5 * exp_mae +                    # Exp chi (primary task)
            2.0 * (1.0 - sol_f1) +            # Solubility F1
            1.5 * (1.0 - sol_recall) +        # Solubility recall (important!)
            0.1 * stage1_val_mae              # Small penalty for bad Stage 1
        )

        return objective


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transfer-aware hyperparameter search")
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
        help="Number of CV folds for Stage 2",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=20,
        help="Number of epochs for quick Stage 2 training",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Study name (default: transfer_aware_YYYYMMDD_HHMMSS)",
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
    study_name = args.study_name or f"transfer_aware_{timestamp}"
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
    objective = TransferAwareObjective(
        base_config_path=Path(args.config),
        results_dir=results_dir,
        n_cv_folds=args.n_cv_folds,
        stage2_epochs=args.stage2_epochs,
        verbose=True,
    )

    # Run optimization
    print(f"\n{'='*80}")
    print(f"STARTING TRANSFER-AWARE HYPERPARAMETER SEARCH")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Results directory: {results_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Timeout: {args.timeout_hours} hours")
    print(f"CV folds: {args.n_cv_folds}")
    print(f"Stage 2 epochs: {args.stage2_epochs}")
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
    save_complete_results(study, results_dir, prefix="")

    print(f"\n✅ All results saved to {results_dir}")
    print(f"\nBest Stage 1 hyperparameters for Stage 2 transfer:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
