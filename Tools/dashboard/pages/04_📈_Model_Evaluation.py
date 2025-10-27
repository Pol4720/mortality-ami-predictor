"""Model Evaluation and Metrics page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st

from app import (
    get_state,
    initialize_state,
    list_saved_models,
)
from src.evaluation import evaluate_main

# Initialize
initialize_state()

# Page config
st.title("📈 Model Evaluation")
st.markdown("---")

# Check if data has been loaded
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    data_path = st.session_state.get('data_path')
    st.success("✅ Usando datos limpios")
elif raw_data is not None:
    df = raw_data
    data_path = st.session_state.get('data_path')
    st.warning("⚠️ Usando datos crudos")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Si no hay data_path o el path no existe, crear un archivo temporal
import tempfile
if not data_path or not Path(data_path).exists():
    st.info("ℹ️ Guardando datos en archivo temporal para la evaluación...")
    temp_dir = Path(tempfile.gettempdir())
    data_path = temp_dir / "streamlit_evaluation_dataset.csv"
    df.to_csv(data_path, index=False)
    st.session_state.data_path = str(data_path)

# Get task from session state
task = st.session_state.get('target_column', 'mortality')
if task == 'exitus':
    task = 'mortality'

# Model selection for evaluation
st.sidebar.markdown("---")
st.sidebar.header("📊 Evaluation Settings")

saved_models = list_saved_models(task)

if not saved_models:
    st.error(f"❌ No trained models found for task '{task}'. Please train models first.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Model to Evaluate",
    list(saved_models.keys()),
    help="Select a model to evaluate on the test set"
)

# Evaluation button
if st.button("🚀 Run Evaluation", type="primary", width='stretch'):
    try:
        # Get the selected model path
        selected_model_path = Path(saved_models[selected_model])
        
        # Copy the selected model to best_classifier_{task}.joblib for evaluation
        # This is needed because evaluate_main expects this specific filename
        models_dir = selected_model_path.parent
        best_classifier_path = models_dir / f"best_classifier_{task}.joblib"
        testset_path = models_dir / f"testset_{task}.parquet"
        
        # Verify testset exists
        if not testset_path.exists():
            st.error(f"❌ Test set no encontrado: {testset_path}")
            st.warning("⚠️ Por favor, re-entrena los modelos en la página **Model Training** para generar el test set.")
            st.stop()
        
        import shutil
        shutil.copy2(selected_model_path, best_classifier_path)
        st.info(f"📋 Evaluando modelo: {selected_model}")
        
        with st.spinner(f"Evaluating {selected_model} on test hold-out set..."):
            evaluate_main(["--data", str(data_path), "--task", task])
        
        st.success(f"✅ Evaluation completed for {selected_model}")
        st.session_state.is_evaluated = True
        
    except Exception as e:
        st.error(f"❌ Evaluation error: {e}")
        st.exception(e)
        st.exception(e)

st.markdown("---")

# Display evaluation results
st.subheader("Evaluation Results")

reports_dir = Path(root_dir) / "reports"
figures_dir = reports_dir / "figures"

# Metrics table
st.subheader("📊 Performance Metrics")

try:
    # Buscar archivo de métricas (nuevo formato)
    metrics_file = reports_dir / f"evaluation_metrics_{task}.csv"
    
    # Si no existe, buscar formato antiguo con timestamp
    if not metrics_file.exists():
        metrics_files = sorted(
            reports_dir.glob(f"final_metrics_{task}_*.csv"),
            key=lambda p: p.stat().st_mtime
        )
        if metrics_files:
            metrics_file = metrics_files[-1]
        else:
            metrics_file = None
    
except Exception:
    metrics_file = None

if metrics_file and metrics_file.exists():
    try:
        # Try multiple encodings for CSV files
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        metrics_df = None
        
        for encoding in encodings:
            try:
                metrics_df = pd.read_csv(metrics_file, encoding=encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if metrics_df is None:
            raise RuntimeError("No se pudo leer el archivo de métricas")
        
        # Display as styled dataframe
        st.dataframe(
            metrics_df.style.format(precision=4),
            width='stretch',
            hide_index=True
        )
        
        # Key metrics highlight
        if not metrics_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            # Try to extract key metrics
            try:
                if "accuracy" in metrics_df.columns:
                    with col1:
                        st.metric("Accuracy", f"{metrics_df['accuracy'].iloc[0]:.4f}")
                if "roc_auc" in metrics_df.columns or "auc" in metrics_df.columns:
                    auc_col = "roc_auc" if "roc_auc" in metrics_df.columns else "auc"
                    with col2:
                        st.metric("ROC AUC", f"{metrics_df[auc_col].iloc[0]:.4f}")
                if "f1" in metrics_df.columns:
                    with col3:
                        st.metric("F1 Score", f"{metrics_df['f1'].iloc[0]:.4f}")
                if "precision" in metrics_df.columns:
                    with col4:
                        st.metric("Precision", f"{metrics_df['precision'].iloc[0]:.4f}")
            except Exception:
                pass
    
    except Exception as e:
        st.warning(f"⚠️ Could not load metrics CSV: {e}")
else:
    st.info("ℹ️ No metrics available yet. Run evaluation to generate metrics.")

st.markdown("---")

# Visualization figures
st.subheader("📉 Evaluation Plots")

def get_latest_figure(pattern: str) -> Path | None:
    """Get the latest figure matching the pattern."""
    try:
        files = sorted(
            figures_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )
        return files[-1] if files else None
    except Exception:
        return None

# Display figures in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Calibration Curve")
    # Try with and without timestamp pattern
    calib_fig = get_latest_figure(f"calibration_{task}_*.png") or get_latest_figure(f"calibration_{task}.png")
    if calib_fig and calib_fig.exists():
        st.image(str(calib_fig), width='stretch')
    else:
        st.info("No calibration plot available")

with col2:
    st.markdown("#### Decision Curve")
    decision_fig = get_latest_figure(f"decision_curve_{task}_*.png") or get_latest_figure(f"decision_curve_{task}.png")
    if decision_fig and decision_fig.exists():
        st.image(str(decision_fig), width='stretch')
    else:
        st.info("No decision curve available")

with col3:
    st.markdown("#### Confusion Matrix")
    confusion_fig = get_latest_figure(f"confusion_{task}_*.png") or get_latest_figure(f"confusion_{task}.png")
    if confusion_fig and confusion_fig.exists():
        st.image(str(confusion_fig), width='stretch')
    else:
        st.info("No confusion matrix available")

st.markdown("---")

# ROC Curve (full width)
st.markdown("#### ROC Curve")
roc_fig = get_latest_figure(f"roc_{task}_*.png") or get_latest_figure(f"roc_{task}.png")
if roc_fig and roc_fig.exists():
    st.image(str(roc_fig), width='stretch')
else:
    st.info("No ROC curve available")

# Additional figures
with st.expander("🔍 Additional Plots"):
    # Precision-Recall curve
    pr_fig = get_latest_figure(f"pr_{task}_*.png") or get_latest_figure(f"pr_{task}.png")
    if pr_fig and pr_fig.exists():
        st.markdown("##### Precision-Recall Curve")
        st.image(str(pr_fig), width='stretch')
    
    # Learning curve
    learning_fig = get_latest_figure(f"learning_curve_{task}_*.png") or get_latest_figure(f"learning_curve_{task}.png")
    if learning_fig and learning_fig.exists():
        st.markdown("##### Learning Curve")
        st.image(str(learning_fig), width='stretch')

# Evaluation notes
with st.expander("ℹ️ About Evaluation Metrics"):
    st.markdown("""
    **Classification Metrics:**
    - **Accuracy**: Overall correctness of predictions
    - **ROC AUC**: Area Under ROC Curve - discrimination ability
    - **F1 Score**: Harmonic mean of precision and recall
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    
    **Calibration:**
    - Measures how well predicted probabilities match actual outcomes
    - Perfect calibration: diagonal line
    
    **Decision Curve:**
    - Clinical utility across different threshold probabilities
    - Helps determine optimal decision thresholds
    
    **Confusion Matrix:**
    - Visual representation of prediction errors
    - True positives, false positives, true negatives, false negatives
    """)
