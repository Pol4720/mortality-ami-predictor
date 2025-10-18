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


def _detect_excel_kind(path: str) -> str:
    """Detect Excel kind by magic bytes: returns 'xlsx' (zip), 'xls' (cfb) or 'unknown'."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if head.startswith(b"PK\x03\x04"):
            return "xlsx"
        if head.startswith(b"\xD0\xCF\x11\xE0"):
            return "xls"
    except Exception:
        pass
    return "unknown"


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from a file path. Supports CSV, Parquet, Feather, Excel (.xlsx, .xls).

    Args:
        path: Path to file. If None, uses CONFIG.dataset_path.

    Returns:
        pandas DataFrame
    """
    if path is None:
        validate_config(CONFIG)
        path = CONFIG.dataset_path

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    accepted_ext = set(SUPPORTED_EXT) | {".xlsx", ".xls"}

    if ext not in accepted_ext:
        raise ValueError(f"Unsupported dataset extension '{ext}'. Supported: {sorted(accepted_ext)}")

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    elif ext in {".xlsx", ".xls"}:
        # Robust handling with format detection and helpful fallbacks
        kind = _detect_excel_kind(path)
        try:
            if ext == ".xlsx" or kind == "xlsx":
                # Require openpyxl for .xlsx
                df = pd.read_excel(path, engine="openpyxl")
            elif ext == ".xls" or kind == "xls":
                # Require xlrd<2.0 for legacy .xls
                # Prefer explicit engine to avoid incompatible xlrd versions
                df = pd.read_excel(path, engine="xlrd")
            else:
                # Unknown signature: try openpyxl first, then xlrd
                try:
                    df = pd.read_excel(path, engine="openpyxl")
                except Exception:
                    df = pd.read_excel(path, engine="xlrd")
        except ImportError as e:
            if ext == ".xlsx" or kind == "xlsx":
                raise ImportError(
                    "Para leer .xlsx necesitas 'openpyxl'. Instala con: pip install openpyxl"
                ) from e
            else:
                raise ImportError(
                    "Para leer .xls necesitas 'xlrd<2.0'. Instala con: pip install xlrd==1.2.0"
                ) from e
        except Exception as e:
            # Provide guidance on common corruption/extension mismatches
            msg = (
                "No se pudo leer el Excel. Posibles causas: archivo corrupto o extensión incorrecta. "
                "Si el archivo es moderno, regrábalo como .xlsx y vuelve a intentar; también puedes exportarlo a CSV. "
                f"Detalle: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e
    else:
        raise ValueError(f"Unhandled extension: {ext}")

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
    """Compute summary tables for EDA with compatibility across pandas versions."""
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).T  # pandas >= 1.1
    except TypeError:
        # Fallback for older pandas without datetime_is_numeric
        desc = df.describe(include="all").T
    summary = {
        "describe": desc,
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
        try:
            corr = df[num_cols].corr(numeric_only=True)
        except TypeError:
            corr = df[num_cols].corr()
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
            try:
                ax.bar_label(ax.containers[0])
            except Exception:
                pass
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


def data_audit(df: pd.DataFrame, feature_cols: Optional[list] = None, top_n: int = 30) -> Dict[str, object]:
    """Early audit to inspect NaNs and feature quality.

    Returns a dict with:
    - head: first rows of features
    - nan_summary: DataFrame with count and fraction of NaNs per column (sorted desc)
    - full_missing: list of columns fully NaN
    - mostly_missing: list of columns with >80% NaNs
    - constant: list of columns with <=1 unique non-NaN value
    """
    if feature_cols is None:
        feature_cols = list(df.columns)
    X = df[feature_cols]
    nan_count = X.isna().sum()
    nan_frac = (nan_count / len(X)).fillna(0.0)
    nan_summary = (
        pd.DataFrame({"nan_count": nan_count, "nan_fraction": nan_frac})
        .sort_values("nan_fraction", ascending=False)
        .head(top_n)
    )
    full_missing = [c for c in feature_cols if X[c].isna().all()]
    mostly_missing = [c for c in feature_cols if X[c].isna().mean() > 0.8]
    constant = [c for c in feature_cols if X[c].nunique(dropna=True) <= 1]
    return {
        "head": X.head(10),
        "nan_summary": nan_summary,
        "full_missing": full_missing,
        "mostly_missing": mostly_missing,
        "constant": constant,
    }
