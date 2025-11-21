"""
Optuna utilities for hyperparameter optimization.

Provides helper functions for:
- Study creation and loading
- Best trial visualization
- Hyperparameter importance analysis
- Results export
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_or_load_study(
    study_name: str,
    storage_path: Optional[Path] = None,
    direction: str = "minimize",
) -> optuna.Study:
    """
    Create a new study or load existing one.

    Args:
        study_name: Name of the study
        storage_path: Path to SQLite database (if None, uses in-memory)
        direction: "minimize" or "maximize"

    Returns:
        Optuna study object
    """
    if storage_path is not None:
        storage = f"sqlite:///{storage_path}"
    else:
        storage = None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    return study


def save_best_params(
    study: optuna.Study,
    save_path: Path,
):
    """
    Save best hyperparameters to JSON.

    Args:
        study: Optuna study
        save_path: Path to save JSON file
    """
    best_params = study.best_params
    best_value = study.best_value

    results = {
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best hyperparameters saved to {save_path}")


def export_study_to_csv(
    study: optuna.Study,
    save_path: Path,
):
    """
    Export all trials to CSV.

    Args:
        study: Optuna study
        save_path: Path to save CSV file
    """
    df = study.trials_dataframe()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Study trials exported to {save_path}")


def plot_optimization_history(
    study: optuna.Study,
    save_path: Path,
    title: str = "Optimization History",
):
    """
    Plot optimization history.

    Args:
        study: Optuna study
        save_path: Path to save figure
        title: Plot title
    """
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.set_title(title, fontsize=14)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization history saved to {save_path}")


def plot_param_importances(
    study: optuna.Study,
    save_path: Path,
    title: str = "Hyperparameter Importances",
):
    """
    Plot hyperparameter importances.

    Args:
        study: Optuna study
        save_path: Path to save figure
        title: Plot title
    """
    try:
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.set_title(title, fontsize=14)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter importances saved to {save_path}")
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")


def plot_parallel_coordinate(
    study: optuna.Study,
    save_path: Path,
    params: Optional[List[str]] = None,
):
    """
    Plot parallel coordinate plot.

    Args:
        study: Optuna study
        save_path: Path to save figure
        params: List of parameter names to include (None = all)
    """
    try:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=params)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parallel coordinate plot saved to {save_path}")
    except Exception as e:
        print(f"Could not plot parallel coordinate: {e}")


def plot_slice(
    study: optuna.Study,
    save_path: Path,
    params: Optional[List[str]] = None,
):
    """
    Plot slice plot showing parameter effects.

    Args:
        study: Optuna study
        save_path: Path to save figure
        params: List of parameter names to include (None = all)
    """
    try:
        fig = optuna.visualization.matplotlib.plot_slice(study, params=params)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Slice plot saved to {save_path}")
    except Exception as e:
        print(f"Could not plot slice: {e}")


def print_study_summary(study: optuna.Study):
    """
    Print summary of study results.

    Args:
        study: Optuna study
    """
    print("\n" + "="*80)
    print("OPTUNA STUDY SUMMARY")
    print("="*80)

    print(f"\nStudy name: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if len(study.trials) > 0:
        print(f"\nBest trial:")
        print(f"  Value: {study.best_value:.6f}")
        print(f"  Trial number: {study.best_trial.number}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

    print("="*80 + "\n")


def create_all_plots(
    study: optuna.Study,
    save_dir: Path,
    prefix: str = "",
):
    """
    Create all standard Optuna plots.

    Args:
        study: Optuna study
        save_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optimization history
    plot_optimization_history(
        study,
        save_dir / f"{prefix}optimization_history.png",
        title=f"{prefix}Optimization History"
    )

    # Parameter importances
    plot_param_importances(
        study,
        save_dir / f"{prefix}param_importances.png",
        title=f"{prefix}Hyperparameter Importances"
    )

    # Parallel coordinate
    plot_parallel_coordinate(
        study,
        save_dir / f"{prefix}parallel_coordinate.png"
    )

    # Slice plot
    plot_slice(
        study,
        save_dir / f"{prefix}slice_plot.png"
    )

    print(f"\nAll plots saved to {save_dir}")


def get_top_trials(
    study: optuna.Study,
    n: int = 10,
) -> pd.DataFrame:
    """
    Get top N trials.

    Args:
        study: Optuna study
        n: Number of top trials to return

    Returns:
        DataFrame of top trials
    """
    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=(study.direction == optuna.study.StudyDirection.MINIMIZE))
    return df.head(n)


def save_complete_results(
    study: optuna.Study,
    save_dir: Path,
    prefix: str = "",
):
    """
    Save all study results (params, plots, CSV).

    Args:
        study: Optuna study
        save_dir: Directory to save results
        prefix: Prefix for filenames
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print_study_summary(study)

    # Save best params
    save_best_params(study, save_dir / f"{prefix}best_params.json")

    # Export all trials
    export_study_to_csv(study, save_dir / f"{prefix}all_trials.csv")

    # Create all plots
    create_all_plots(study, save_dir / "figures", prefix=prefix)

    # Save top trials
    top_trials = get_top_trials(study, n=20)
    top_trials.to_csv(save_dir / f"{prefix}top_20_trials.csv", index=False)

    print(f"\n✅ All results saved to {save_dir}")


def save_trial_metrics(
    trial: optuna.Trial,
    hyperparams: Dict,
    metrics: Dict,
    output_path: Path,
):
    """
    Save detailed trial metrics to CSV (append mode).

    Args:
        trial: Optuna trial object
        hyperparams: Dictionary of all hyperparameters (Stage 1 + Stage 2)
        metrics: Dictionary of all metrics (train/val/test)
        output_path: Path to CSV file (will be created/appended)
    """
    # Combine trial number, objective, hyperparams, and metrics
    row_data = {
        'trial_number': trial.number,
        'objective': trial.value if trial.value is not None else float('nan'),
    }

    # Add all hyperparameters
    row_data.update(hyperparams)

    # Add all metrics
    row_data.update(metrics)

    # Convert to DataFrame
    df = pd.DataFrame([row_data])

    # Append to CSV (create if doesn't exist)
    if output_path.exists():
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, mode='w', header=True, index=False)

    # Also store in trial user_attrs for Optuna
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
