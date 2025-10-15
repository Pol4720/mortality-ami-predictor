"""Global configuration for the ML project.

Only one external input is required: DATASET_PATH. This can be set via
an environment variable, CLI argument, or directly in this variable.

Never hardcode other data paths in the codebase.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


RANDOM_SEED: int = 42


@dataclass
class ProjectConfig:
    """Project-wide configuration parameters.

    Attributes:
        dataset_path: Path to the dataset file (.csv, .parquet, etc.).
        experiment_tracker: Name of experiment tracker ("mlflow" or "wandb").
        tracking_uri: Tracking URI for MLflow or W&B entity/project string.
        target_column: Binary target for in-hospital mortality (0/1).
        arrhythmia_column: Optional binary target for ventricular arrhythmia (0/1).
        regression_target: Optional continuous target (e.g., length_of_stay).
    """

    dataset_path: str = os.environ.get("DATASET_PATH", "")
    experiment_tracker: str = os.environ.get("EXPERIMENT_TRACKER", "mlflow")
    tracking_uri: Optional[str] = os.environ.get("TRACKING_URI")
    target_column: str = os.environ.get("TARGET_COLUMN", "mortality_inhospital")
    arrhythmia_column: Optional[str] = os.environ.get("ARRHYTHMIA_COLUMN", "ventricular_arrhythmia")
    regression_target: Optional[str] = os.environ.get("REGRESSION_TARGET", None)


CONFIG = ProjectConfig()


def validate_config(cfg: ProjectConfig) -> None:
    """Validate minimal config is provided and raise helpful errors.

    Args:
        cfg: ProjectConfig
    """
    if not cfg.dataset_path:
        raise ValueError(
            "DATASET_PATH is not set. Set env var DATASET_PATH or pass --data to scripts."
        )
