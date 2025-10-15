"""Data utilities: loading, EDA summaries, and plotting.

All data must be loaded from a single path provided via DATASET_PATH.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from .config import CONFIG, ProjectConfig, validate_config, RANDOM_SEED


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class DatasetInfo:
    """Container with dataset metadata and splits."""

    columns: List[str]
    n_rows: int
    target_col: Optional[str]
    arrhythmia_col: Optional[str]
    regression_target: Optional[str]


SUPPORTED_EXT = {".csv", ".parquet", ".feather"}


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from a file path. Supports CSV, Parquet, Feather.

    Args:
        path: Path to file. If None, uses CONFIG.dataset_path.

    Returns:
        pandas DataFrame
    """
    if path is None:
        validate_config(CONFIG)
        path = CONFIG.dataset_path

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported dataset extension '{ext}'. Supported: {SUPPORTED_EXT}")

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_feather(path)

    return df


def get_dataset_info(df: pd.DataFrame, cfg: ProjectConfig = CONFIG) -> DatasetInfo:
    """Infer dataset info given dataframe and config."""
    return DatasetInfo(
        columns=list(df.columns),
        n_rows=len(df),
        target_col=cfg.target_column if cfg.target_column in df.columns else None,
        arrhythmia_col=cfg.arrhythmia_column if cfg.arrhythmia_column in df.columns else None,
        regression_target=cfg.regression_target if cfg.regression_target and cfg.regression_target in df.columns else None,
    )


def summarize_dataframe(df: pd.DataFrame, cfg: ProjectConfig = CONFIG) -> Dict[str, pd.DataFrame]:
    """Compute summary tables for EDA."""
    summary = {
        "describe": df.describe(include="all", datetime_is_numeric=True).T,
        "missing": df.isna().mean().sort_values(ascending=False).to_frame("missing_rate"),
        "dtypes": df.dtypes.astype(str).to_frame("dtype"),
    }

    # Class balance for targets if present
    for col in [cfg.target_column, cfg.arrhythmia_column]:
        if col and col in df.columns:
            vc = df[col].value_counts(dropna=False)
            frac = vc / vc.sum()
            summary[f"class_balance_{col}"] = pd.DataFrame({"count": vc, "fraction": frac})

    return summary


def save_eda_plots(df: pd.DataFrame, cfg: ProjectConfig = CONFIG, prefix: str = "eda") -> None:
    """Create and save EDA plots: missingness heatmap, correlation matrix, target distributions."""
    # Missingness heatmap (limit to first 50 cols for readability)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.iloc[:, : min(50, df.shape[1])].isna(), cbar=False)
    plt.title("Missingness heatmap (first 50 columns)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{prefix}_missingness_heatmap.png"))
    plt.close()

    # Correlation matrix for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation matrix (numeric)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{prefix}_correlation_matrix.png"))
        plt.close()

    # Target distributions
    for col in [cfg.target_column, cfg.arrhythmia_column]:
        if col and col in df.columns:
            plt.figure(figsize=(6, 4))
            ax = sns.countplot(x=df[col].astype(str))
            ax.bar_label(ax.containers[0])
            plt.title(f"Target distribution: {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"{prefix}_target_{col}_dist.png"))
            plt.close()


def train_test_split(df: pd.DataFrame, cfg: ProjectConfig = CONFIG, test_size: float = 0.2,
                     time_column: Optional[str] = None, stratify_target: Optional[str] = None,
                     random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a robust train/test split with optional temporal split or stratification."""
    from sklearn.model_selection import train_test_split as sk_split

    if time_column and time_column in df.columns:
        # Temporal split: sort by time and split last fraction as test
        df_sorted = df.sort_values(time_column)
        n_test = int(np.floor(len(df_sorted) * test_size))
        test_df = df_sorted.iloc[-n_test:]
        train_df = df_sorted.iloc[:-n_test]
        return train_df, test_df

    stratify = None
    if stratify_target and stratify_target in df.columns:
        stratify = df[stratify_target]

    train_df, test_df = sk_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    return train_df, test_df


def select_feature_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, dropping target from features."""
    if target_col not in df.columns:
        raise KeyError(f"Required target column '{target_col}' not found in dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
