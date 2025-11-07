"""UI utilities and helper functions for dashboard pages."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.data_load import (
    data_audit,
    load_dataset,
    summarize_dataframe,
    train_test_split,
    save_dataset_with_timestamp,
    cleanup_old_testsets,
    get_latest_model,
    get_latest_testset,
)
from src.features import safe_feature_columns
from src.preprocessing import PreprocessingConfig
from src.training import fit_and_save_best_classifier, fit_and_save_selected_classifiers
from .config import MODELS_DIR, TESTSETS_DIR, CLEANED_DATASETS_DIR


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
    st.dataframe(df.head(n_rows), width='stretch')
    
    summaries = summarize_dataframe(df)
    
    with st.expander("📊 Missing Values Summary"):
        st.dataframe(summaries["missing"].head(50), width='stretch')
    
    with st.expander("📈 Statistical Description"):
        st.dataframe(summaries["describe"].head(50), width='stretch')
    
    with st.expander("🔤 Data Types"):
        st.dataframe(summaries["dtypes"], width='stretch')


def display_data_audit(df: pd.DataFrame, feature_cols: list[str]):
    """Display data quality audit results.
    
    Args:
        df: DataFrame to audit
        feature_cols: List of feature column names
    """
    audit = data_audit(df, feature_cols)
    
    st.subheader("🔍 Data Quality Audit")
    
    with st.expander("Preview"):
        st.dataframe(audit["head"], width='stretch')
    
    with st.expander("Missing Values by Column"):
        st.dataframe(audit["nan_summary"], width='stretch')
    
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
                st.sidebar.image(str(cand), width='stretch')
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
    custom_model_classes: dict = None,
) -> dict[str, str]:
    """Train selected models using the rigorous experiment pipeline.
    
    This function ALWAYS uses the rigorous academic pipeline with:
    - 30+ repeated CV runs for model selection (FASE 1)
    - Learning curves for diagnostics
    - Statistical comparison (Shapiro-Wilk, t-test/Mann-Whitney) (FASE 3)
    
    NOTE: Bootstrap and Jackknife evaluation (FASE 2) is done in the 
    EVALUATION module, not here.
    
    Args:
        data_path: Path to dataset
        task: Task name (mortality/arrhythmia)
        quick: Whether to use quick mode (fewer CV splits)
        imputer_mode: Imputation strategy
        selected_models: List of model names to train (includes custom models)
        custom_model_classes: Dictionary mapping custom model names to their classes
        
    Returns:
        Dictionary mapping model names to saved file paths
    """
    if custom_model_classes is None:
        custom_model_classes = {}
    
    # Load data
    df = load_dataset(data_path)
    
    # Determine target
    if task == "mortality":
        target = CONFIG.target_column
    else:
        target = CONFIG.arrhythmia_column
    
    # Split data (80% train, 20% test) with STRATIFICATION
    train_df, test_df = train_test_split(df, stratify_column=target, test_size=0.2, random_state=42)
    
    # Display split information with class distribution
    st.info(f"""
    📊 **División de datos (estratificada por {target})**
    - Train: {len(train_df)} muestras ({len(train_df)/len(df)*100:.1f}%)
    - Test: {len(test_df)} muestras ({len(test_df)/len(df)*100:.1f}%)
    """)
    
    # Show class distribution to verify stratification
    col1, col2 = st.columns(2)
    with col1:
        train_dist = train_df[target].value_counts(normalize=True).sort_index()
        st.write("**Distribución Train:**")
        for label, prop in train_dist.items():
            st.write(f"  Clase {label}: {prop*100:.2f}%")
    
    with col2:
        test_dist = test_df[target].value_counts(normalize=True).sort_index()
        st.write("**Distribución Test:**")
        for label, prop in test_dist.items():
            st.write(f"  Clase {label}: {prop*100:.2f}%")
    
    # Verify stratification
    max_diff = abs(train_dist - test_dist).max()
    if max_diff < 0.05:  # Less than 5% difference
        st.success(f"✅ Estratificación correcta (diferencia máxima: {max_diff*100:.2f}%)")
    else:
        st.warning(f"⚠️ Posible desbalance en estratificación (diferencia: {max_diff*100:.2f}%)")
    
    # Save test set and train set immediately after split to new location
    # Model type will be determined later, for now save with task name
    # This will be updated when individual models are trained
    try:
        # Save both train and test sets with timestamp
        test_path = save_dataset_with_timestamp(
            test_df, 
            TESTSETS_DIR, 
            prefix=f"testset_{task}",
            format="parquet"
        )
        train_path = save_dataset_with_timestamp(
            train_df,
            TESTSETS_DIR,
            prefix=f"trainset_{task}",
            format="parquet"
        )
        st.success(f"✅ Test set guardado: {len(test_df)} muestras en {test_path.name}")
        st.info(f"ℹ️ Train set guardado: {len(train_df)} muestras en {train_path.name}")
    except Exception as e:
        st.error(f"❌ Error guardando train/test sets: {e}")
        raise
    
    # Prepare features
    X_train = train_df[safe_feature_columns(train_df, [target])]
    y_train = train_df[target]
    X_test = test_df[safe_feature_columns(test_df, [target])]
    y_test = test_df[target]
    
    # Progress tracking with real-time updates
    progress_bar = st.progress(0.0)
    status_text = st.empty()  # For main status message
    details_text = st.empty()  # For detailed progress info
    metrics_container = st.empty()  # For metrics display
    
    def progress_callback(msg: str, frac: float):
        """Callback with real-time progress updates."""
        # Update progress bar
        progress_bar.progress(min(max(frac, 0.0), 1.0))
        
        # Update status text
        status_text.info(f"🔄 {msg}")
        
        # Parse message to extract metrics if available
        if "μ=" in msg and "σ=" in msg:
            try:
                # Extract mean and std
                mu_part = msg.split("μ=")[1].split(",")[0].strip()
                sigma_part = msg.split("σ=")[1].split()[0].strip()
                
                # Display metrics only if valid numbers
                try:
                    mu_val = float(mu_part)
                    sigma_val = float(sigma_part)
                    
                    with metrics_container.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Progreso", f"{frac*100:.1f}%")
                        with col2:
                            st.metric("AUROC Promedio (μ)", f"{mu_val:.4f}")
                        with col3:
                            st.metric("Desviación (σ)", f"{sigma_val:.4f}")
                except ValueError:
                    pass
            except:
                pass
        else:
            # If no metrics, just show progress
            with metrics_container.container():
                st.metric("Progreso", f"{frac*100:.1f}%")
    
    # Create preprocessing config
    preprocessing_config = PreprocessingConfig()
    preprocessing_config.imputer_mode = imputer_mode
    
    # Get selected models
    from src.models import make_classifiers
    all_models = make_classifiers()
    
    # Add custom models if provided
    if custom_model_classes:
        for model_name, model_class in custom_model_classes.items():
            # Instantiate custom model with default parameters
            all_models[model_name] = model_class()
    
    models_to_train = {k: v for k, v in all_models.items() if k in selected_models}
    
    if not models_to_train:
        raise ValueError("No models selected for training")
    
    # ALWAYS use rigorous pipeline
    status_text.info("🚀 Ejecutando pipeline de experimentación riguroso...")
    
    # Import rigorous pipeline
    from src.training import run_rigorous_experiment_pipeline
    
    # Set CV parameters based on quick mode
    n_splits = 3 if quick else 10
    n_repeats = 3 if quick else 10  # 9 or 100 runs total
    
    # Run pipeline (FASE 1 + FASE 3 - train/validation + statistical comparison)
    experiment_results = run_rigorous_experiment_pipeline(
        X=X_train,
        y=y_train,
        models=models_to_train,
        preprocessing_config=preprocessing_config,
        n_splits=n_splits,
        n_repeats=n_repeats,
        scoring="roc_auc",
        test_set=(X_test, y_test),  # Saved for later evaluation
        output_dir=str(MODELS_DIR),
        progress_callback=progress_callback,
    )
    
    # Extract save paths
    best_model_name = experiment_results.get('best_model')
    final_model_path = experiment_results.get('final_model_path')
    learning_curves = experiment_results.get('learning_curves', {})
    
    save_paths = {}
    if final_model_path:
        save_paths[best_model_name] = final_model_path
    
    # Store learning curve results and figure paths in session state
    if learning_curves:
        lc_figure_paths = {}
        for model_name in learning_curves.keys():
            lc_path = MODELS_DIR / f"learning_curve_{model_name}.png"
            if lc_path.exists():
                lc_figure_paths[model_name] = str(lc_path)
        
        st.session_state.learning_curve_paths = lc_figure_paths
        st.session_state.learning_curve_results = learning_curves
    
    # Store experiment results for statistical comparison display
    st.session_state.experiment_results = experiment_results
    
    status_text.success(
        f"✅ Pipeline de entrenamiento completado!\n"
        f"Mejor modelo: {best_model_name}\n"
        f"Modelo guardado en: {final_model_path}\n"
        f"Test set guardado en: {test_path}\n"
        f"⚠️ La evaluación final (Bootstrap/Jackknife) se hará en el módulo de EVALUACIÓN"
    )
    
    progress_bar.progress(1.0)
    
    return save_paths, experiment_results


def list_saved_models(task: str) -> dict[str, str]:
    """List all saved models for a given task.
    
    Args:
        task: Task name (mortality/arrhythmia)
        
    Returns:
        Dictionary mapping model names to file paths
    """
    mapping = {}
    
    if MODELS_DIR.exists():
        # Look in model type subdirectories for latest models
        from src.data_load import get_all_model_types, get_latest_model
        
        for model_type in get_all_model_types(MODELS_DIR):
            latest_model = get_latest_model(model_type, MODELS_DIR)
            if latest_model:
                mapping[model_type] = str(latest_model)
        
        # Also check for best classifier (legacy format)
        best = MODELS_DIR / f"best_classifier_{task}.joblib"
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
    
    st.dataframe(df, width='stretch', hide_index=True)

