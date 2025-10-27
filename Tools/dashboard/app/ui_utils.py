"""UI utilities and helper functions for dashboard pages."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.data_load import data_audit, load_dataset, summarize_dataframe, train_test_split
from src.features import safe_feature_columns
from src.preprocessing import PreprocessingConfig
from src.training import fit_and_save_best_classifier, fit_and_save_selected_classifiers


def display_dataframe_info(df: pd.DataFrame, title: str = "Dataset Info"):
    """Display comprehensive dataframe information.
    
    Args:
        df: DataFrame to display
        title: Title for the section
    """
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.2f}%")


def display_dataset_preview(df: pd.DataFrame, n_rows: int = 10):
    """Display dataset preview with expandable sections.
    
    Args:
        df: DataFrame to preview
        n_rows: Number of rows to show
    """
    st.subheader("Data Preview")
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    summaries = summarize_dataframe(df)
    
    with st.expander("📊 Missing Values Summary"):
        st.dataframe(summaries["missing"].head(50), use_container_width=True)
    
    with st.expander("📈 Statistical Description"):
        st.dataframe(summaries["describe"].head(50), use_container_width=True)
    
    with st.expander("🔤 Data Types"):
        st.dataframe(summaries["dtypes"], use_container_width=True)


def display_data_audit(df: pd.DataFrame, feature_cols: list[str]):
    """Display data quality audit results.
    
    Args:
        df: DataFrame to audit
        feature_cols: List of feature column names
    """
    audit = data_audit(df, feature_cols)
    
    st.subheader("🔍 Data Quality Audit")
    
    with st.expander("Preview"):
        st.dataframe(audit["head"], use_container_width=True)
    
    with st.expander("Missing Values by Column"):
        st.dataframe(audit["nan_summary"], use_container_width=True)
    
    if audit["full_missing"]:
        st.warning(f"⚠️ Completely empty columns: {', '.join(audit['full_missing'][:20])}")
    
    if audit["mostly_missing"]:
        st.info(f"ℹ️ Columns with >80% missing: {', '.join(audit['mostly_missing'][:20])}")
    
    if audit["constant"]:
        st.info(f"ℹ️ Constant/near-constant columns: {', '.join(audit['constant'][:20])}")


def get_default_data_path() -> str:
    """Get default dataset path.
    
    Returns:
        Default dataset path as string
    """
    try:
        default_abs = r"C:\Users\HP\Desktop\ML\Proyecto\mortality-ami-predictor\DATA\recuima-020425.xlsx"
        default_rel = os.path.relpath(default_abs, start=os.getcwd()) if os.path.isabs(default_abs) else default_abs
    except Exception:
        default_rel = "DATA\\recuima-020425.xlsx"
    
    return os.environ.get("DATASET_PATH", default_rel)


def sidebar_data_controls():
    """Render sidebar controls for data loading and task selection.
    
    Returns:
        Tuple of (data_path, task)
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Display logo if available
    try:
        assets_dir = Path(__file__).parent / "assets"
        for pat in ("logo.png", "logo.jpg", "logo.jpeg", "logo.ico"):
            cand = assets_dir / pat
            if cand.exists():
                st.sidebar.image(str(cand), use_container_width=True)
                break
    except Exception:
        pass
    
    data_path = st.sidebar.text_input(
        "Dataset Path",
        value=get_default_data_path(),
        help="Path to the dataset file (CSV or Excel)"
    )
    
    task = st.sidebar.selectbox(
        "Prediction Task",
        ["mortality", "arrhythmia"],
        index=0,
        help="Select the prediction target"
    )
    
    return data_path, task


def sidebar_training_controls():
    """Render sidebar controls for model training.
    
    Returns:
        Tuple of (quick_mode, imputer_mode, selected_models)
    """
    st.sidebar.header("🎯 Training Settings")
    
    quick = st.sidebar.checkbox(
        "Quick Mode (Debug)",
        value=True,
        help="Use faster settings for debugging"
    )
    
    imputer_mode = st.sidebar.selectbox(
        "Imputation Strategy",
        ["iterative", "knn", "simple"],
        index=0,
        help="Method for handling missing values"
    )
    
    # Model selection
    from src.models import make_classifiers
    model_keys = list(make_classifiers().keys())
    
    selected_models = st.sidebar.multiselect(
        "Models to Train",
        model_keys,
        default=model_keys,
        help="Select which models to train and evaluate"
    )
    
    return quick, imputer_mode, selected_models


def train_models_with_progress(
    data_path: str,
    task: str,
    quick: bool,
    imputer_mode: str,
    selected_models: list[str],
) -> dict[str, str]:
    """Train selected models with progress feedback.
    
    Args:
        data_path: Path to dataset
        task: Task name (mortality/arrhythmia)
        quick: Whether to use quick mode
        imputer_mode: Imputation strategy
        selected_models: List of model names to train
        
    Returns:
        Dictionary mapping model names to saved file paths
    """
    # Load data
    df = load_dataset(data_path)
    
    # Determine target
    if task == "mortality":
        target = CONFIG.target_column
    else:
        target = CONFIG.arrhythmia_column
    
    # Split data
    train_df, test_df = train_test_split(df, stratify_column=target)
    
    # Save test set immediately after split
    models_dir = Path(__file__).parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    test_path = models_dir / f"testset_{task}.parquet"
    try:
        test_df.to_parquet(test_path)
        st.success(f"✅ Test set guardado: {len(test_df)} muestras en {test_path.name}")
    except Exception as e:
        st.error(f"❌ Error guardando test set: {e}")
        raise
    
    # Prepare features
    X = train_df[safe_feature_columns(train_df, [target])]
    y = train_df[target]
    
    # Progress tracking
    progress = st.progress(0.0)
    status = st.empty()
    
    def progress_callback(msg: str, frac: float):
        status.info(msg)
        progress.progress(min(max(frac, 0.0), 1.0))
    
    # Create preprocessing config from imputer_mode
    preprocessing_config = PreprocessingConfig()
    preprocessing_config.imputer_mode = imputer_mode
    
    # Train models
    if selected_models:
        save_paths = fit_and_save_selected_classifiers(
            X, y,
            selected_models=selected_models,
            quick=quick,
            task_name=task,
            preprocessing_config=preprocessing_config,
            progress_callback=progress_callback,
        )
        status.success(f"✅ Models saved: {', '.join(save_paths.keys())}")
    else:
        path, _ = fit_and_save_best_classifier(
            X, y,
            quick=quick,
            task_name=task,
            preprocessing_config=preprocessing_config,
            progress_callback=progress_callback
        )
        save_paths = {"best": path}
        status.success(f"✅ Model saved at {path}")
    
    progress.progress(1.0)
    
    return save_paths


def list_saved_models(task: str) -> dict[str, str]:
    """List all saved models for a given task.
    
    Args:
        task: Task name (mortality/arrhythmia)
        
    Returns:
        Dictionary mapping model names to file paths
    """
    # Path from dashboard/app/ui_utils.py -> Tools/models
    models_dir = Path(__file__).parents[2] / "models"
    mapping = {}
    
    if models_dir.exists():
        # Individual models
        for p in models_dir.glob(f"model_{task}_*.joblib"):
            name = p.stem.replace(f"model_{task}_", "")
            mapping[name] = str(p)
        
        # Best classifier
        best = models_dir / f"best_classifier_{task}.joblib"
        if best.exists():
            mapping["best"] = str(best)
    
    return mapping


def display_model_list(task: str):
    """Display available saved models.
    
    Args:
        task: Task name (mortality/arrhythmia)
    """
    models = list_saved_models(task)
    
    if not models:
        st.warning(f"⚠️ No saved models found for task '{task}'")
        return
    
    st.success(f"✅ Found {len(models)} saved model(s)")
    
    for name, path in models.items():
        with st.expander(f"📦 {name}"):
            st.code(path, language="text")


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric value for display.
    
    Args:
        value: Metric value
        metric_name: Name of metric
        
    Returns:
        Formatted string
    """
    if metric_name.lower() in ("accuracy", "precision", "recall", "f1", "auc", "roc_auc"):
        return f"{value:.4f}"
    return f"{value:.2f}"


def display_metrics_table(metrics: dict[str, float], title: str = "Metrics"):
    """Display metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the metrics section
    """
    st.subheader(title)
    
    # Create DataFrame
    df = pd.DataFrame([
        {"Metric": k, "Value": format_metric_value(v, k)}
        for k, v in metrics.items()
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
