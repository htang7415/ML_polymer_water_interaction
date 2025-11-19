"""
Hyperparameter tracking utilities.

Provides functions for logging and analyzing hyperparameters across training runs,
useful for hyperparameter optimization and experiment tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("polymer_chi_ml.hparam_tracker")


def save_hyperparameters(config, save_path: Path) -> None:
    """
    Save all hyperparameters from config to a JSON file.

    Args:
        config: Configuration object
        save_path: Path to save hyperparameters JSON
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract hyperparameters from config
    hparams = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "seed": config.seed,
        },

        "model": {
            "encoder_latent_dim": config.model.encoder_latent_dim,
            "encoder_hidden_dims": config.model.encoder_hidden_dims,
            "encoder_dropout": config.model.encoder_dropout,
            "encoder_batch_norm": config.model.encoder_batch_norm,
            "chi_head_hidden_dim": config.model.chi_head_hidden_dim,
            "chi_head_dropout": config.model.chi_head_dropout,
            "solubility_head_hidden_dim": config.model.solubility_head_hidden_dim,
            "solubility_head_dropout": config.model.solubility_head_dropout,
        },

        "training": {
            "optimizer": config.training.optimizer,
            "lr_pretrain": config.training.lr_pretrain,
            "lr_finetune": config.training.lr_finetune,
            "weight_decay": config.training.weight_decay,
            "batch_size_dft": config.training.batch_size_dft,
            "batch_size_exp": config.training.batch_size_exp,
            "batch_size_sol": config.training.batch_size_sol,
            "num_epochs_pretrain": config.training.num_epochs_pretrain,
            "num_epochs_finetune": config.training.num_epochs_finetune,
            "grad_clip_norm": config.training.grad_clip_norm,
            "early_stopping": config.training.early_stopping,
            "early_stopping_patience": config.training.early_stopping_patience,
            "use_scheduler": config.training.use_scheduler,
            "scheduler_type": config.training.scheduler_type if config.training.use_scheduler else None,
            "scheduler_patience": config.training.scheduler_patience if config.training.use_scheduler else None,
            "scheduler_factor": config.training.scheduler_factor if config.training.use_scheduler else None,
        },

        "loss_weights": {
            "w_dft": config.loss_weights.w_dft,
            "w_exp": config.loss_weights.w_exp,
            "w_sol": config.loss_weights.w_sol,
        },

        "data": {
            "test_fraction": config.data.test_fraction,
            "val_fraction": config.data.val_fraction,
            "stratify_solubility": config.data.stratify_solubility,
            "group_by_smiles_exp": config.data.group_by_smiles_exp,
        },

        "features": {
            "morgan_radius": config.features.morgan_radius,
            "morgan_n_bits": config.features.morgan_n_bits,
            "use_features": config.features.use_features,
            "use_chirality": config.features.use_chirality,
        },

        "solubility": {
            "T_ref": config.solubility.T_ref,
            "decision_threshold": config.solubility.decision_threshold,
            "class_weight_strategy": config.solubility.class_weight_strategy,
            "pos_class_weight": config.solubility.pos_class_weight if config.solubility.class_weight_strategy == "manual" else None,
        },
    }

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(hparams, f, indent=2)

    logger.info(f"Saved hyperparameters to {save_path}")


def load_hyperparameters(hparam_path: Path) -> Dict[str, Any]:
    """
    Load hyperparameters from JSON file.

    Args:
        hparam_path: Path to hyperparameters JSON

    Returns:
        Dictionary of hyperparameters
    """
    hparam_path = Path(hparam_path)

    if not hparam_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hparam_path}")

    with open(hparam_path, "r") as f:
        hparams = json.load(f)

    logger.info(f"Loaded hyperparameters from {hparam_path}")
    return hparams


def compare_hyperparameters(hparams1: Dict, hparams2: Dict) -> Dict[str, Any]:
    """
    Compare two hyperparameter configurations and identify differences.

    Args:
        hparams1: First hyperparameter dictionary
        hparams2: Second hyperparameter dictionary

    Returns:
        Dictionary of differences
    """
    differences = {}

    def compare_nested(d1, d2, prefix=""):
        diffs = {}
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in d1:
                diffs[full_key] = {"in_config1": None, "in_config2": d2[key]}
            elif key not in d2:
                diffs[full_key] = {"in_config1": d1[key], "in_config2": None}
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested_diffs = compare_nested(d1[key], d2[key], full_key)
                diffs.update(nested_diffs)
            elif d1[key] != d2[key]:
                diffs[full_key] = {"in_config1": d1[key], "in_config2": d2[key]}

        return diffs

    differences = compare_nested(hparams1, hparams2)
    return differences


def create_hparam_summary_table(run_dirs: list) -> str:
    """
    Create a markdown table summarizing hyperparameters across multiple runs.

    Args:
        run_dirs: List of paths to run directories containing hyperparameters.json

    Returns:
        Markdown formatted table string
    """
    # Load hyperparameters from all runs
    all_hparams = []
    for run_dir in run_dirs:
        hparam_path = Path(run_dir) / "hyperparameters.json"
        if hparam_path.exists():
            hparams = load_hyperparameters(hparam_path)
            hparams["run_dir"] = str(run_dir)
            all_hparams.append(hparams)

    if not all_hparams:
        return "No hyperparameter files found."

    # Extract key hyperparameters for comparison
    table_rows = []
    header = "| Run | LR | Batch Size | Dropout | Latent Dim | Loss Weights (DFT/Exp/Sol) |"
    separator = "|-----|----|-----------|---------|-----------|-----------------------------|"
    table_rows.append(header)
    table_rows.append(separator)

    for idx, hparams in enumerate(all_hparams, 1):
        run_name = Path(hparams["run_dir"]).name
        lr = hparams.get("training", {}).get("lr_pretrain", "N/A")
        batch_size = hparams.get("training", {}).get("batch_size_dft", "N/A")
        dropout = hparams.get("model", {}).get("encoder_dropout", "N/A")
        latent_dim = hparams.get("model", {}).get("encoder_latent_dim", "N/A")

        loss_weights = hparams.get("loss_weights", {})
        w_dft = loss_weights.get("w_dft", "N/A")
        w_exp = loss_weights.get("w_exp", "N/A")
        w_sol = loss_weights.get("w_sol", "N/A")
        weights_str = f"{w_dft}/{w_exp}/{w_sol}"

        row = f"| {run_name} | {lr} | {batch_size} | {dropout} | {latent_dim} | {weights_str} |"
        table_rows.append(row)

    return "\n".join(table_rows)
