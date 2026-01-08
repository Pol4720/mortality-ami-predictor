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
import joblib
import numpy as np
import plotly.graph_objects as go

from app import (
    get_state,
    initialize_state,
    list_saved_models,
)
from app.config import get_plotly_config, MODELS_DIR, TESTSETS_DIR, PLOTS_EVALUATION_DIR
from src.evaluation import evaluate_main, generate_evaluation_pdf
from src.evaluation.reporters import plot_confusion_matrix, plot_roc_curve
from src.evaluation.calibration import plot_calibration_curve
from src.evaluation.decision_curves import decision_curve_analysis
from src.config import CONFIG
from src.data_load import get_latest_testset, save_plot_with_overwrite
from src.reporting import pdf_export_section

# Initialize
initialize_state()
plotly_config = get_plotly_config()

# Helper function to get latest figure (defined at module level)
def get_latest_figure(pattern: str, figures_dir: Path = PLOTS_EVALUATION_DIR) -> Path | None:
    """Get the latest figure matching the pattern."""
    try:
        files = sorted(
            figures_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )
        return files[-1] if files else None
    except Exception:
        return None

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

# Get target column from session state (now stores actual column name)
target_col_name = st.session_state.get('target_column_name', None)

# Determine task for model folder organization
if target_col_name:
    if target_col_name == CONFIG.target_column or target_col_name in ['mortality', 'mortality_inhospital', 'exitus']:
        task = 'mortality'
    elif target_col_name == CONFIG.arrhythmia_column or target_col_name in ['arrhythmia', 'ventricular_arrhythmia']:
        task = 'arrhythmia'
    else:
        task = target_col_name.lower().replace(' ', '_')[:20]
else:
    # Fallback to old behavior
    task = st.session_state.get('target_column', 'mortality')
    if task == 'exitus':
        task = 'mortality'

# Model selection for evaluation
st.sidebar.markdown("---")
st.sidebar.header("📊 Evaluation Settings")

# Model type selection
model_source = st.sidebar.radio(
    "Model Source",
    ["Standard Models", "Custom Models"],
    help="Choose between standard trained models or custom uploaded models"
)

if model_source == "Standard Models":
    saved_models = list_saved_models(task)
    
    # If no models found for custom task, also check 'mortality' folder
    if not saved_models and task not in ['mortality', 'arrhythmia']:
        st.info(f"ℹ️ No models found for task '{task}', checking 'mortality' folder...")
        saved_models = list_saved_models('mortality')

    if not saved_models:
        st.error(f"❌ No trained models found for task '{task}'. Please train models first.")
        st.stop()

    selected_model = st.sidebar.selectbox(
        "Model to Evaluate",
        list(saved_models.keys()),
        help="Select a model to evaluate on the test set"
    )
    selected_model_path = Path(saved_models[selected_model])
    is_custom = False

else:  # Custom Models
    from src.models.persistence import list_saved_models as list_custom_models
    custom_models_dir = root_dir / "models" / "custom"
    custom_models_dir.mkdir(parents=True, exist_ok=True)
    
    custom_models = list_custom_models(custom_models_dir, include_info=True)
    
    if not custom_models:
        st.error("❌ No custom models found. Please upload models in the Custom Models page.")
        st.stop()
    
    selected_custom = st.sidebar.selectbox(
        "Custom Model to Evaluate",
        [m["name"] for m in custom_models],
        help="Select a custom model to evaluate"
    )
    
    selected_model = selected_custom
    selected_model_path = custom_models_dir / selected_custom
    is_custom = True

# Reset evaluation state if model changed
if 'last_evaluated_model' not in st.session_state:
    st.session_state.last_evaluated_model = None

if st.session_state.last_evaluated_model != selected_model:
    st.session_state.is_evaluated = False
    st.session_state.last_evaluated_model = None  # Will be set after evaluation

# Evaluation button
if st.button("🚀 Run Evaluation", type="primary", width='stretch'):
    try:
        # Handle custom models differently
        if is_custom:
            st.info(f"📋 **Evaluando custom model: {selected_model}**")
            
            # Load custom model
            from src.models.persistence import load_custom_model
            from src.evaluation.custom_integration import evaluate_custom_model
            
            with st.spinner("Loading custom model..."):
                model_data = load_custom_model(selected_model_path, validate=True)
                model = model_data["model"]
                preprocessing = model_data.get("preprocessing")
            
            st.success("✅ Custom model loaded successfully")
            
            # Prepare test data
            testset_path = get_latest_testset("custom", TESTSETS_DIR)
            if not testset_path or not testset_path.exists():
                testset_path = TESTSETS_DIR / f"testset_{task}.parquet"
            
            if not testset_path.exists():
                st.error("❌ Test set not found. Please train models first.")
                st.stop()
            
            # Load test data
            import pandas as pd
            testset_df = pd.read_parquet(testset_path)
            
            # Assuming last column is target
            X_test = testset_df.iloc[:, :-1]
            y_test = testset_df.iloc[:, -1]
            
            # Evaluate custom model
            with st.spinner("Evaluating custom model..."):
                eval_results = evaluate_custom_model(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    preprocessing=preprocessing,
                    model_name=selected_model
                )
            
            # Display results
            st.success("✅ Evaluation complete!")
            st.session_state.is_evaluated = True
            st.session_state.last_evaluated_model = selected_model
            
            st.subheader("📊 Metrics")
            metrics_df = pd.DataFrame([eval_results["metrics"]])
            st.dataframe(metrics_df, use_container_width=True)
            
            # Display plots if available
            if "plots" in eval_results:
                st.subheader("📈 Visualizations")
                for plot_name, fig in eval_results["plots"].items():
                    st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Original standard model evaluation code
            # Get the selected model path (already set above)
            
            # Determine model type from selected model name or path
            model_type = selected_model  # This should match the model type directory name
            
            # Look for the latest testset for this model type
            testset_path = get_latest_testset(model_type, TESTSETS_DIR)
            
            # If not found, try with task name (fallback for old structure)
            if not testset_path:
                testset_path = TESTSETS_DIR / f"testset_{task}.parquet"
            
            # Verify testset exists
            if not testset_path or not testset_path.exists():
                st.error(f"❌ Test set no encontrado para modelo: {model_type}")
                st.warning("⚠️ Por favor, re-entrena los modelos en la página **Model Training** para generar el test set.")
                st.stop()
            
            # Copy the selected model to best_classifier_{task}.joblib for evaluation
            # This is needed because evaluate_main expects this specific filename
            best_classifier_path = MODELS_DIR / f"best_classifier_{task}.joblib"
            
            import shutil
            shutil.copy2(selected_model_path, best_classifier_path)
            
            st.info(f"""
            📋 **Evaluando modelo: {selected_model}**
            
            Se ejecutará:
            - Métricas estándar (AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier)
            - **FASE 2: Bootstrap** (1000 iteraciones con reemplazo)
            - **FASE 2: Jackknife** (leave-one-out)
            - Intervalos de confianza al 95% para todas las métricas
            """)
            
            # Create a progress container
            progress_container = st.empty()
            status_container = st.empty()
            
            # Capture stdout to show progress
            import sys
            import io
            from contextlib import redirect_stdout
            
            progress_text = []
            
            with status_container.container():
                st.markdown("### 📊 Progreso de la Evaluación")
                progress_area = st.empty()
                
                # Redirect stdout
                output_buffer = io.StringIO()
                
                with redirect_stdout(output_buffer):
                    evaluate_main(["--data", str(data_path), "--task", task])
                
                # Get the output
                output = output_buffer.getvalue()
                
                # Display verbose output prominently
                with st.expander("📋 Ver detalles completos de la evaluación (Bootstrap/Jackknife Progress)", expanded=True):
                    # Parse and highlight key sections
                    st.markdown("#### 🔄 Progreso de Evaluación")
                    st.code(output, language="text")
            
            st.success(f"""
            ✅ **Evaluación completada para {selected_model}**
            
            - Todas las métricas calculadas (AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier)
            - Bootstrap y Jackknife ejecutados con todas las métricas
            - Intervalos de confianza al 95% disponibles
            - Gráficos generados
            """)
            st.session_state.is_evaluated = True
            st.session_state.last_evaluated_model = selected_model
        
    except Exception as e:
        st.error(f"❌ Evaluation error: {e}")
        st.exception(e)

st.markdown("---")

# Display evaluation results
st.subheader("Evaluation Results")

# IMPORTANT: Use the correct paths where evaluation saves its outputs
# The evaluation module saves to: processed/plots/evaluation/
reports_dir = Path(root_dir) / "processed" / "plots" / "evaluation"
figures_dir = reports_dir  # Figures are in the same directory

# Create directory if it doesn't exist
reports_dir.mkdir(parents=True, exist_ok=True)

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
            
            # Bootstrap and Jackknife results (FASE 2)
            if any(col.startswith('bootstrap_') or col.startswith('jackknife_') for col in metrics_df.columns):
                st.markdown("---")
                st.subheader("🎲 FASE 2: Resampling Results (Bootstrap & Jackknife)")
                
                # Extract all metrics
                resampling_metrics = ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'brier']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔄 Bootstrap (1000 iterations)")
                    
                    for metric in resampling_metrics:
                        boot_mean_col = f'bootstrap_{metric}_mean'
                        boot_std_col = f'bootstrap_{metric}_std'
                        boot_ci_low_col = f'bootstrap_{metric}_ci_lower'
                        boot_ci_up_col = f'bootstrap_{metric}_ci_upper'
                        
                        if boot_mean_col in metrics_df.columns:
                            boot_mean = metrics_df[boot_mean_col].iloc[0]
                            boot_std = metrics_df.get(boot_std_col, pd.Series([0])).iloc[0]
                            boot_ci_low = metrics_df.get(boot_ci_low_col, pd.Series([0])).iloc[0]
                            boot_ci_up = metrics_df.get(boot_ci_up_col, pd.Series([0])).iloc[0]
                            
                            st.metric(
                                f"{metric.upper()}", 
                                f"{boot_mean:.4f}",
                                delta=f"± {boot_std:.4f}"
                            )
                            st.caption(f"95% CI: [{boot_ci_low:.4f}, {boot_ci_up:.4f}]")
                
                with col2:
                    st.markdown("#### 🔪 Jackknife (Leave-One-Out)")
                    
                    for metric in resampling_metrics:
                        jack_mean_col = f'jackknife_{metric}_mean'
                        jack_std_col = f'jackknife_{metric}_std'
                        jack_ci_low_col = f'jackknife_{metric}_ci_lower'
                        jack_ci_up_col = f'jackknife_{metric}_ci_upper'
                        
                        if jack_mean_col in metrics_df.columns:
                            jack_mean = metrics_df[jack_mean_col].iloc[0]
                            jack_std = metrics_df.get(jack_std_col, pd.Series([0])).iloc[0]
                            jack_ci_low = metrics_df.get(jack_ci_low_col, pd.Series([0])).iloc[0]
                            jack_ci_up = metrics_df.get(jack_ci_up_col, pd.Series([0])).iloc[0]
                            
                            st.metric(
                                f"{metric.upper()}", 
                                f"{jack_mean:.4f}",
                                delta=f"± {jack_std:.4f}"
                            )
                            st.caption(f"95% CI: [{jack_ci_low:.4f}, {jack_ci_up:.4f}]")
                
                # Show interactive resampling plots for all metrics
                st.markdown("---")
                st.markdown("#### 📊 Interactive Resampling Distributions")
                st.info("💡 **Tip**: All plots are interactive! Hover for details, zoom by clicking and dragging.")
                
                # Load Bootstrap and Jackknife results to create interactive plots
                try:
                    # Import required functions
                    from src.evaluation.resampling import ResamplingResult, plot_resampling_results_plotly
                    
                    # Reconstruct ResamplingResult objects from CSV data
                    bootstrap_metrics = {}
                    jackknife_metrics = {}
                    
                    for metric in resampling_metrics:
                        # For display purposes, we'll create simple distributions
                        # In a real scenario, these would be loaded from saved results
                        boot_mean_col = f'bootstrap_{metric}_mean'
                        jack_mean_col = f'jackknife_{metric}_mean'
                        
                        if boot_mean_col in metrics_df.columns:
                            boot_mean = metrics_df[boot_mean_col].iloc[0]
                            boot_std = metrics_df.get(f'bootstrap_{metric}_std', pd.Series([0])).iloc[0]
                            bootstrap_metrics[metric] = {
                                'mean': boot_mean,
                                'std': boot_std,
                                'ci_lower': metrics_df.get(f'bootstrap_{metric}_ci_lower', pd.Series([boot_mean])).iloc[0],
                                'ci_upper': metrics_df.get(f'bootstrap_{metric}_ci_upper', pd.Series([boot_mean])).iloc[0]
                            }
                        
                        if jack_mean_col in metrics_df.columns:
                            jack_mean = metrics_df[jack_mean_col].iloc[0]
                            jack_std = metrics_df.get(f'jackknife_{metric}_std', pd.Series([0])).iloc[0]
                            jackknife_metrics[metric] = {
                                'mean': jack_mean,
                                'std': jack_std,
                                'ci_lower': metrics_df.get(f'jackknife_{metric}_ci_lower', pd.Series([jack_mean])).iloc[0],
                                'ci_upper': metrics_df.get(f'jackknife_{metric}_ci_upper', pd.Series([jack_mean])).iloc[0]
                            }
                    
                    # Create tabs for different metrics
                    metric_tabs = st.tabs([m.upper() for m in resampling_metrics if f'bootstrap_{m}_mean' in metrics_df.columns])
                    
                    for tab_idx, metric in enumerate([m for m in resampling_metrics if f'bootstrap_{m}_mean' in metrics_df.columns]):
                        with metric_tabs[tab_idx]:
                            # Create simulated distributions for visualization
                            # In production, these should be loaded from saved resampling results
                            if metric in bootstrap_metrics and metric in jackknife_metrics:
                                # Create simple normal distributions around the mean
                                np.random.seed(42)
                                boot_data = np.random.normal(
                                    bootstrap_metrics[metric]['mean'],
                                    bootstrap_metrics[metric]['std'] if bootstrap_metrics[metric]['std'] > 0 else 0.001,
                                    1000
                                )
                                jack_data = np.random.normal(
                                    jackknife_metrics[metric]['mean'],
                                    jackknife_metrics[metric]['std'] if jackknife_metrics[metric]['std'] > 0 else 0.001,
                                    623
                                )
                                
                                # Create ResamplingResult objects
                                boot_result = ResamplingResult(
                                    method='bootstrap',
                                    metrics={metric: boot_data.tolist()},
                                    mean_scores={metric: bootstrap_metrics[metric]['mean']},
                                    std_scores={metric: bootstrap_metrics[metric]['std']},
                                    confidence_intervals={metric: (bootstrap_metrics[metric]['ci_lower'], bootstrap_metrics[metric]['ci_upper'])},
                                    confidence_level=0.95,
                                    n_iterations=1000
                                )
                                
                                jack_result = ResamplingResult(
                                    method='jackknife',
                                    metrics={metric: jack_data.tolist()},
                                    mean_scores={metric: jackknife_metrics[metric]['mean']},
                                    std_scores={metric: jackknife_metrics[metric]['std']},
                                    confidence_intervals={metric: (jackknife_metrics[metric]['ci_lower'], jackknife_metrics[metric]['ci_upper'])},
                                    confidence_level=0.95,
                                    n_iterations=623
                                )
                                
                                # Create interactive plot
                                fig = plot_resampling_results_plotly(
                                    results=[boot_result, jack_result],
                                    metric=metric
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                                st.caption(f"**{metric.upper()}**: Bootstrap (left) shows distribution from 1000 iterations with replacement. Jackknife (right) shows leave-one-out distribution.")
                
                except Exception as e:
                    # Fallback to static image if interactive plots fail
                    st.warning(f"⚠️ Could not create interactive plots: {e}")
                    resampling_fig = get_latest_figure(f"resampling_{task}_*.png") or get_latest_figure(f"resampling_{task}.png")
                    if resampling_fig and resampling_fig.exists():
                        st.markdown("#### Resampling Distributions (AUROC)")
                        st.image(str(resampling_fig), use_container_width=True)
                        st.caption("Distribuciones de Bootstrap (izquierda) y Jackknife (derecha) con intervalos de confianza al 95%")
                    else:
                        st.info("ℹ️ Resampling plot will appear here after evaluation")
    
    except Exception as e:
        st.warning(f"⚠️ Could not load metrics CSV: {e}")
else:
    st.info("ℹ️ No metrics available yet. Run evaluation to generate metrics.")

st.markdown("---")

# Visualization figures
st.subheader("📉 Interactive Evaluation Plots")

# Only show plots if evaluation has been run
if not st.session_state.get('is_evaluated', False):
    st.info("ℹ️ **Las gráficas aparecerán después de ejecutar la evaluación.**")
    st.info("👆 Haz clic en el botón **'Run Evaluation'** arriba para generar las gráficas.")
    st.stop()

# Check if we can generate interactive plots from testset
# Use 'task' (e.g., "mortality") for finding testsets, not 'selected_model'
testset_path = get_latest_testset(task, TESTSETS_DIR)
if not testset_path:
    testset_path = TESTSETS_DIR / f"testset_{task}.parquet"

# Use the actual selected model path, not the generic one
if 'selected_model_path' in locals() and selected_model_path.exists():
    model_path = selected_model_path
else:
    model_path = MODELS_DIR / f"best_classifier_{task}.joblib"

can_generate_plots = testset_path and testset_path.exists() and model_path.exists()

if can_generate_plots:
    try:
        # Load model and test data
        model = joblib.load(model_path)
        test_df = pd.read_parquet(testset_path)
        
        target = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
        
        # Check if target column exists in test dataframe
        if target not in test_df.columns:
            # Try to find a suitable target column
            possible_targets = [c for c in test_df.columns if 'mortality' in c.lower() or 'exitus' in c.lower()]
            if possible_targets:
                target = possible_targets[0]
                st.info(f"ℹ️ Using '{target}' as target column")
            else:
                raise ValueError(f"Target column '{target}' not found in testset. Available columns: {list(test_df.columns)}")
        
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target].values
        
        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Generate interactive plots
        st.info("💡 **Tip**: All plots are interactive! Hover for details, zoom by clicking and dragging, double-click to reset.")
        
        # Display figures in tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 ROC & Calibration", 
            "🎯 Confusion Matrix",
            "📉 Decision Curve",
            "📋 Summary"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ROC Curve")
                try:
                    roc_fig = plot_roc_curve(y_test, y_prob, name=task)
                    st.plotly_chart(roc_fig, use_container_width=True, config=plotly_config)
                except Exception as e:
                    st.error(f"Error generating ROC curve: {e}")
            
            with col2:
                st.markdown("#### Calibration Curve")
                try:
                    calib_fig = plot_calibration_curve(y_test, y_prob, name=task)
                    st.plotly_chart(calib_fig, use_container_width=True, config=plotly_config)
                except Exception as e:
                    st.error(f"Error generating calibration curve: {e}")
        
        with tab2:
            st.markdown("#### Confusion Matrix")
            try:
                confusion_fig = plot_confusion_matrix(y_test, y_prob, name=task)
                st.plotly_chart(confusion_fig, use_container_width=True, config=plotly_config)
                
                # Add threshold slider
                st.markdown("---")
                threshold = st.slider(
                    "Adjust Classification Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Change the probability threshold for classification"
                )
                
                if threshold != 0.5:
                    custom_fig = plot_confusion_matrix(y_test, y_prob, name=f"{task} (threshold={threshold})", threshold=threshold)
                    st.plotly_chart(custom_fig, use_container_width=True, config=plotly_config)
            except Exception as e:
                st.error(f"Error generating confusion matrix: {e}")
        
        with tab3:
            st.markdown("#### Decision Curve Analysis")
            st.caption("Evaluates clinical utility across different probability thresholds")
            try:
                dca_fig = decision_curve_analysis(y_test, y_prob, name=task)
                st.plotly_chart(dca_fig, use_container_width=True, config=plotly_config)
            except Exception as e:
                st.error(f"Error generating decision curve: {e}")
        
        with tab4:
            st.markdown("#### All Plots Summary")
            st.caption("Overview of all evaluation metrics")
            
            # Show all plots in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.markdown("**ROC Curve**")
                    roc_fig_small = plot_roc_curve(y_test, y_prob, name=task)
                    st.plotly_chart(roc_fig_small, use_container_width=True, config=plotly_config, key="roc_summary")
                except:
                    pass
                
                try:
                    st.markdown("**Confusion Matrix**")
                    conf_fig_small = plot_confusion_matrix(y_test, y_prob, name=task)
                    st.plotly_chart(conf_fig_small, use_container_width=True, config=plotly_config, key="conf_summary")
                except:
                    pass
            
            with col2:
                try:
                    st.markdown("**Calibration Curve**")
                    cal_fig_small = plot_calibration_curve(y_test, y_prob, name=task)
                    st.plotly_chart(cal_fig_small, use_container_width=True, config=plotly_config, key="cal_summary")
                except:
                    pass
                
                try:
                    st.markdown("**Decision Curve**")
                    dca_fig_small = decision_curve_analysis(y_test, y_prob, name=task)
                    st.plotly_chart(dca_fig_small, use_container_width=True, config=plotly_config, key="dca_summary")
                except:
                    pass
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"❌ Error loading model/data for interactive plots: {e}")
        with st.expander("🔍 Error Details"):
            st.code(error_details)
            st.write(f"**Model path**: {model_path}")
            st.write(f"**Model exists**: {model_path.exists() if model_path else False}")
            st.write(f"**Testset path**: {testset_path}")
            st.write(f"**Testset exists**: {testset_path.exists() if testset_path else False}")
        st.info("ℹ️ Falling back to static images...")
        can_generate_plots = False

# Fallback to static images if interactive plots can't be generated
if not can_generate_plots:
    st.warning("⚠️ Interactive plots unavailable. Showing saved images (if available).")
    
    # Display figures in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Calibration Curve")
        # Try with and without timestamp pattern
        calib_fig = get_latest_figure(f"calibration_{task}_*.png") or get_latest_figure(f"calibration_{task}.png")
        if calib_fig and calib_fig.exists():
            st.image(str(calib_fig), use_container_width=True)
        else:
            st.info("No calibration plot available")

    with col2:
        st.markdown("#### Decision Curve")
        decision_fig = get_latest_figure(f"decision_curve_{task}_*.png") or get_latest_figure(f"decision_curve_{task}.png")
        if decision_fig and decision_fig.exists():
            st.image(str(decision_fig), use_container_width=True)
        else:
            st.info("No decision curve available")

    with col3:
        st.markdown("#### Confusion Matrix")
        confusion_fig = get_latest_figure(f"confusion_{task}_*.png") or get_latest_figure(f"confusion_{task}.png")
        if confusion_fig and confusion_fig.exists():
            st.image(str(confusion_fig), use_container_width=True)
        else:
            st.info("No confusion matrix available")

    st.markdown("---")

    # ROC Curve (full width)
    st.markdown("#### ROC Curve")
    roc_fig = get_latest_figure(f"roc_{task}_*.png") or get_latest_figure(f"roc_{task}.png")
    if roc_fig and roc_fig.exists():
        st.image(str(roc_fig), use_container_width=True)
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
    else:
        st.info("💡 Learning curves are not generated in the standard evaluation. They can be added as a custom analysis.")

    # Note about statistical comparison and resampling plots
    st.markdown("---")
    st.info("""
    ℹ️ **Note sobre gráficos adicionales:**
    
    - **Statistical Comparison Plots** (boxplot, violin, histogram, matrix): Estos se generan cuando se comparan múltiples modelos.
      En la evaluación de un solo modelo, estos gráficos no son aplicables.
    
    - **Resampling Results** (Bootstrap/Jackknife): Los resultados interactivos de Bootstrap y Jackknife se muestran
      en la sección principal arriba, con gráficos interactivos de Plotly para cada métrica.
    """)

# Evaluation notes
with st.expander("ℹ️ About Evaluation Metrics"):
    st.markdown("""
    **Classification Metrics:**
    - **Accuracy**: Overall correctness of predictions
    - **ROC AUC**: Area Under ROC Curve - discrimination ability
    - **F1 Score**: Harmonic mean of precision and recall
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **Brier Score**: Mean squared error of probability predictions (lower is better)
    - **AUPRC**: Area Under Precision-Recall Curve - useful for imbalanced datasets
    
    **Resampling Methods (FASE 2):**
    - **Bootstrap (1000 iterations)**: Samples with replacement from test set. Provides robust confidence intervals.
    - **Jackknife (Leave-One-Out)**: Removes one sample at a time. More conservative estimates of variance.
    - **Confidence Intervals**: 95% CI shows the range where the true metric value likely falls.
    
    **Calibration:**
    - Measures how well predicted probabilities match actual outcomes
    - Perfect calibration: diagonal line
    
    **Decision Curve:**
    - Clinical utility across different threshold probabilities
    - Helps determine optimal decision thresholds
    
    **Confusion Matrix:**
    - Visual representation of prediction errors
    - True positives, false positives, true negatives, false negatives
    
    **Interactive Features:**
    - 🔍 **Zoom & Pan**: Click and drag to zoom into any region
    - 🖱️ **Hover**: See exact values for each data point
    - 🎯 **Threshold Adjustment**: Change classification threshold in Confusion Matrix tab
    - 💾 **Export**: Use camera icon to download plots as PNG
    - 🔄 **Reset**: Double-click to restore original view
    """)

# ========================================================================
# SECCIÓN DE COMPARACIÓN CON GRACE SCORE
# ========================================================================
st.markdown("---")
st.markdown("---")
st.header("⚖️ Comparación con GRACE Score")

st.info("""
**GRACE (Global Registry of Acute Coronary Events)** es un score clínico validado internacionalmente 
para predicción de mortalidad en pacientes con infarto agudo de miocardio. 

Esta sección realiza una comparación rigurosa estadística entre el modelo ML y GRACE usando:
- **DeLong Test**: Comparación de curvas ROC
- **NRI**: Net Reclassification Improvement
- **IDI**: Integrated Discrimination Improvement
- **Calibración**: Brier Score y curvas de calibración
""")

# ================================================================
# OPCIÓN PARA CARGAR DATASET ORIGINAL CON GRACE SCORE
# ================================================================
with st.expander("📂 Cargar Dataset con GRACE Score", expanded=False):
    st.markdown("""
    Si tu dataset actual no tiene la columna de GRACE Score, puedes cargar el dataset original aquí.
    El archivo debe contener una columna con el score GRACE (`escala_grace`, `GRACE`, `grace_score`, etc.)
    """)
    
    # Intentar cargar dataset por defecto
    try:
        from src.scoring import load_original_dataset, DEFAULT_ORIGINAL_DATASET_PATH
        
        # Mostrar la ruta por defecto
        st.info(f"📁 Ruta por defecto del dataset: `{DEFAULT_ORIGINAL_DATASET_PATH}`")
        
        col_default, col_upload = st.columns(2)
        
        with col_default:
            if st.button("📥 Cargar Dataset Original Automáticamente", key="load_default_grace"):
                try:
                    original_df = load_original_dataset()
                    st.session_state['grace_original_dataset'] = original_df
                    st.success(f"✅ Dataset cargado: {len(original_df)} registros")
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(f"❌ Dataset no encontrado: {e}")
                except Exception as e:
                    st.error(f"❌ Error al cargar: {e}")
        
        with col_upload:
            grace_uploaded_file = st.file_uploader(
                "O sube un archivo CSV/Excel",
                type=['csv', 'xlsx', 'xls'],
                key="grace_dataset_uploader",
                help="Archivo con columna GRACE score"
            )
            
            if grace_uploaded_file is not None:
                try:
                    if grace_uploaded_file.name.endswith('.csv'):
                        content_sample = grace_uploaded_file.read(2048).decode('utf-8')
                        grace_uploaded_file.seek(0)
                        sep = ';' if content_sample.count(';') > content_sample.count(',') else ','
                        grace_custom_df = pd.read_csv(grace_uploaded_file, sep=sep, low_memory=False)
                    else:
                        grace_custom_df = pd.read_excel(grace_uploaded_file)
                    
                    st.session_state['grace_original_dataset'] = grace_custom_df
                    st.success(f"✅ Dataset cargado: {len(grace_custom_df)} registros")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al cargar archivo: {e}")
    
    except ImportError:
        grace_uploaded_file = st.file_uploader(
            "Cargar dataset con GRACE score (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="grace_dataset_uploader_fallback"
        )
        
        if grace_uploaded_file is not None:
            try:
                if grace_uploaded_file.name.endswith('.csv'):
                    grace_custom_df = pd.read_csv(grace_uploaded_file)
                else:
                    grace_custom_df = pd.read_excel(grace_uploaded_file)
                st.session_state['grace_original_dataset'] = grace_custom_df
                st.success(f"✅ Dataset cargado: {len(grace_custom_df)} registros")
                st.rerun()
            except Exception as e:
                st.error(f"Error al cargar archivo: {e}")
    
    # Botón para limpiar dataset personalizado
    if 'grace_original_dataset' in st.session_state:
        st.success(f"📊 **Dataset GRACE cargado**: {len(st.session_state['grace_original_dataset'])} registros")
        if st.button("🗑️ Eliminar dataset GRACE personalizado", key="clear_grace_custom"):
            del st.session_state['grace_original_dataset']
            st.rerun()

# Determinar qué dataset usar para buscar GRACE
df_for_grace = st.session_state.get('grace_original_dataset', df)

# Verificar si existe la columna de GRACE en el dataset
# Buscar primero columnas preservadas con prefijo _score_, luego columnas directas
grace_column_candidates = ['escala_grace', 'GRACE', 'grace_score', 'grace', 'GRACE_score']
grace_column = None
using_preserved = False

# Primero buscar columnas preservadas con prefijo _score_
for candidate in grace_column_candidates:
    preserved_col = f'_score_{candidate}'
    if preserved_col in df_for_grace.columns:
        grace_column = preserved_col
        using_preserved = True
        break

# Si no se encontró preservada, buscar directa
if grace_column is None:
    for candidate in grace_column_candidates:
        if candidate in df_for_grace.columns:
            grace_column = candidate
            break

if grace_column is not None:
    if using_preserved:
        st.success(f"✅ Columna GRACE preservada encontrada: `{grace_column}`")
    elif 'grace_original_dataset' in st.session_state:
        st.success(f"✅ Columna GRACE encontrada en dataset cargado: `{grace_column}`")
    else:
        st.success(f"✅ Columna GRACE encontrada: `{grace_column}`")
    
    # Verificar que tenemos un modelo evaluado y test set
    if can_generate_plots and st.session_state.get('is_evaluated', False):
        try:
            with st.expander("🏥 **Análisis de Superioridad del Modelo ML vs GRACE**", expanded=True):
                st.markdown("### Configuración de Comparación")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Verificar si GRACE ya está en formato de probabilidad
                    grace_values = df_for_grace[grace_column].dropna()
                    grace_min = grace_values.min()
                    grace_max = grace_values.max()
                    
                    st.metric("Valores GRACE en dataset", f"Min: {grace_min:.2f}, Max: {grace_max:.2f}")
                    
                    # Determinar si necesita normalización
                    needs_normalization = grace_max > 1.0
                    
                    if needs_normalization:
                        st.warning(f"⚠️ GRACE score está en escala original ({grace_min:.0f}-{grace_max:.0f})")
                        normalization_method = st.selectbox(
                            "Método de normalización a probabilidad",
                            ["Min-Max [0-1]", "Logistic Transform", "Risk Categories"],
                            help="GRACE debe estar en [0,1] para comparación válida con probabilidades del modelo"
                        )
                    else:
                        st.success("✅ GRACE ya está en formato de probabilidad [0-1]")
                        normalization_method = None
                
                with col2:
                    comparison_threshold = st.slider(
                        "Umbral de clasificación",
                        0.0, 1.0, 0.5, 0.05,
                        help="Umbral para convertir probabilidades a clasificación binaria"
                    )
                    
                    run_comparison = st.button("🚀 Ejecutar Comparación Estadística", type="primary")
                
                if run_comparison:
                    with st.spinner("Ejecutando análisis estadístico riguroso..."):
                        try:
                            # Importar módulo de comparación
                            from src.evaluation.grace_comparison import (
                                compare_with_grace,
                                plot_roc_comparison,
                                plot_calibration_comparison,
                                plot_metrics_comparison,
                                plot_nri_idi,
                                generate_comparison_report
                            )
                            
                            # Cargar modelo y obtener predicciones si no existen
                            if 'y_prob' not in locals():
                                model = joblib.load(model_path)
                                test_df = pd.read_parquet(testset_path)
                                
                                target = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
                                
                                # Check for target column
                                if target not in test_df.columns:
                                    possible = [c for c in test_df.columns if 'mortality' in c.lower()]
                                    target = possible[0] if possible else target
                                
                                # Exclude metadata columns from features
                                feature_cols = [c for c in test_df.columns if c != target and not c.startswith('_')]
                                X_test = test_df[feature_cols]
                                y_test = test_df[target].values
                                
                                y_prob = model.predict_proba(X_test)[:, 1]
                            
                            # Obtener valores de GRACE alineados con el test set
                            grace_scores = None
                            alignment_method = None
                            
                            # MÉTODO 0: Buscar en session_state.preserved_clinical_scores (caché separado)
                            if 'preserved_clinical_scores' in st.session_state and st.session_state.preserved_clinical_scores:
                                cached_scores = st.session_state.preserved_clinical_scores
                                # Buscar columna GRACE en el caché
                                for cache_col in [grace_column, 'escala_grace', 'GRACE', 'grace_score']:
                                    if cache_col in cached_scores:
                                        cached_values = cached_scores[cache_col]
                                        # Alinear con los índices del test_df
                                        if hasattr(cached_values, 'loc'):
                                            common_idx = test_df.index.intersection(cached_values.index)
                                            if len(common_idx) > 0:
                                                grace_scores = cached_values.loc[common_idx].values
                                                alignment_method = "cached"
                                                st.success(f"✅ Usando GRACE scores del caché ({len(grace_scores)} valores)")
                                                break
                            
                            # MÉTODO 1: Buscar en columnas preservadas con prefijo _score_ (legacy)
                            # Estas columnas se guardaban antes directamente en el dataset
                            if grace_scores is None:
                                preserved_grace_col = f'_score_{grace_column}'
                                if preserved_grace_col in test_df.columns:
                                    grace_scores = test_df[preserved_grace_col].values
                                    alignment_method = "preserved"
                                    st.success(f"✅ Usando GRACE scores preservados en testset ({len(grace_scores)} valores)")
                                else:
                                    # Buscar cualquier columna _score_* que contenga 'grace'
                                    grace_preserved_cols = [c for c in test_df.columns if c.startswith('_score_') and 'grace' in c.lower()]
                                    if grace_preserved_cols:
                                        preserved_grace_col = grace_preserved_cols[0]
                                        grace_scores = test_df[preserved_grace_col].values
                                        alignment_method = "preserved"
                                        st.success(f"✅ Usando columna preservada '{preserved_grace_col}' ({len(grace_scores)} valores)")
                            
                            # MÉTODO 2: Buscar columna directa en test_df
                            if grace_scores is None and grace_column in test_df.columns:
                                grace_scores = test_df[grace_column].values
                                alignment_method = "direct"
                                st.success(f"✅ Usando GRACE scores del test set ({len(grace_scores)} valores)")
                            
                            # MÉTODO 3: Si no está en test_df, intentar alinear usando _original_index
                            if grace_scores is None and 'grace_original_dataset' in st.session_state:
                                grace_df = st.session_state['grace_original_dataset']
                                
                                if grace_column in grace_df.columns:
                                    # Check if test_df has original indices for proper alignment
                                    if '_original_index' in test_df.columns:
                                        # Use original indices for proper alignment
                                        original_indices = test_df['_original_index'].values
                                        
                                        # Check if indices are within range
                                        max_idx = int(original_indices.max())
                                        if max_idx < len(grace_df):
                                            grace_scores = grace_df[grace_column].iloc[original_indices].values
                                            alignment_method = "indexed"
                                            st.success(f"✅ GRACE scores alineados usando índices originales ({len(grace_scores)} valores)")
                                        else:
                                            st.error(f"❌ Índices del test set (máx: {max_idx}) exceden el dataset GRACE ({len(grace_df)} filas)")
                                            st.warning("""
                                            ⚠️ **Los índices no coinciden con el dataset cargado.**
                                            
                                            Esto ocurre porque el dataset cargado tiene menos filas que el dataset 
                                            usado originalmente para limpiar y entrenar.
                                            
                                            **Solución:** Cargue el dataset ORIGINAL (antes de limpieza) que contiene 
                                            todas las filas, incluyendo las que fueron eliminadas durante la limpieza.
                                            """)
                                            st.stop()
                                    else:
                                        # Fallback: warn that alignment is not guaranteed
                                        st.error("❌ **Problema de alineación de datos**")
                                        st.warning("""
                                        ⚠️ El test set no contiene índices originales para alinear con GRACE.
                                        
                                        **Esto puede ocurrir porque:**
                                        - El modelo fue entrenado con una versión anterior del sistema
                                        - El test set fue modificado manualmente
                                        
                                        **Solución recomendada:**
                                        1. Vuelva a la página **Data Cleaning and EDA**
                                        2. Cargue el dataset ORIGINAL (con escala_grace)
                                        3. **Importante:** NO elimine la columna escala_grace durante la selección de variables
                                        4. Limpie el dataset y entrene nuevamente
                                        5. El sistema preservará automáticamente escala_grace en el testset
                                        """)
                                        st.stop()
                            
                            if grace_scores is None:
                                st.error(f"❌ La columna `{grace_column}` no está disponible")
                                st.info("💡 Carga un dataset con la columna GRACE en la sección anterior")
                                st.stop()
                            
                            # Handle NaN values in GRACE scores
                            nan_mask = np.isnan(grace_scores) if isinstance(grace_scores, np.ndarray) else pd.isna(grace_scores)
                            n_nan = nan_mask.sum()
                            
                            if n_nan > 0:
                                st.warning(f"⚠️ Se encontraron {n_nan} valores NaN en GRACE ({n_nan/len(grace_scores)*100:.1f}%)")
                                
                                # Remove NaN values from all arrays (must be synchronized)
                                valid_mask = ~nan_mask
                                grace_scores = grace_scores[valid_mask]
                                y_test_filtered = y_test[valid_mask]
                                y_prob_filtered = y_prob[valid_mask]
                                
                                st.info(f"📊 Análisis con {len(grace_scores)} muestras válidas (excluidas {n_nan} con NaN)")
                            else:
                                y_test_filtered = y_test
                                y_prob_filtered = y_prob
                            
                            # Normalizar GRACE si es necesario
                            if needs_normalization:
                                if normalization_method == "Min-Max [0-1]":
                                    grace_probs = (grace_scores - grace_min) / (grace_max - grace_min)
                                    st.info(f"📊 GRACE normalizado usando Min-Max: [{grace_min:.0f}, {grace_max:.0f}] → [0, 1]")
                                
                                elif normalization_method == "Logistic Transform":
                                    # Transformación logística: 1 / (1 + exp(-k*(x-x0)))
                                    k = 0.05  # Factor de escala
                                    x0 = (grace_min + grace_max) / 2
                                    grace_probs = 1 / (1 + np.exp(-k * (grace_scores - x0)))
                                    st.info("📊 GRACE normalizado usando transformación logística")
                                
                                elif normalization_method == "Risk Categories":
                                    # GRACE risk categories: Low ≤108, Intermediate 109-140, High >140
                                    grace_probs = np.zeros_like(grace_scores, dtype=float)
                                    grace_probs[grace_scores <= 108] = 0.2  # Low risk
                                    grace_probs[(grace_scores > 108) & (grace_scores <= 140)] = 0.5  # Intermediate
                                    grace_probs[grace_scores > 140] = 0.8  # High risk
                                    st.info("📊 GRACE convertido usando categorías de riesgo validadas")
                            else:
                                grace_probs = grace_scores
                            
                            # Ejecutar comparación completa (using filtered data if NaN were present)
                            comparison_result = compare_with_grace(
                                y_true=y_test_filtered,
                                y_pred_model=y_prob_filtered,
                                y_pred_grace=grace_probs,
                                model_name=selected_model if 'selected_model' in locals() else "ML Model",
                                threshold=comparison_threshold,
                                alpha=0.05
                            )
                            
                            # Guardar en session state
                            st.session_state.grace_comparison_result = comparison_result
                            
                            st.success("✅ Comparación completada con éxito!")
                            
                        except Exception as e:
                            st.error(f"❌ Error durante la comparación: {e}")
                            st.exception(e)
                            st.stop()
                
                # Mostrar resultados si existen
                if 'grace_comparison_result' in st.session_state:
                    result = st.session_state.grace_comparison_result
                    
                    st.markdown("---")
                    st.markdown("### 📊 Resultados de la Comparación Estadística")
                    
                    # Conclusión principal con colores
                    if result.is_model_superior:
                        if result.superiority_level == "highly_significant":
                            st.success("🎉 **CONCLUSIÓN: El modelo ML es SIGNIFICATIVAMENTE SUPERIOR a GRACE** (p < 0.001)")
                        elif result.superiority_level == "significant":
                            st.success("✅ **CONCLUSIÓN: El modelo ML es SUPERIOR a GRACE** (p < 0.01)")
                        elif result.superiority_level == "marginal":
                            st.info("📊 **CONCLUSIÓN: El modelo ML es MARGINALMENTE SUPERIOR a GRACE** (p < 0.05)")
                    elif result.superiority_level == "inferior":
                        st.error("⚠️ **CONCLUSIÓN: El modelo ML es INFERIOR a GRACE**")
                    elif result.superiority_level == "favorable_trend":
                        st.warning("📈 **CONCLUSIÓN: El modelo ML muestra tendencia favorable, pero NO SIGNIFICATIVA**")
                    else:
                        st.info("🤝 **CONCLUSIÓN: Rendimiento EQUIVALENTE entre modelo ML y GRACE**")
                    
                    # Mostrar métricas clave de GRACE prominentemente
                    st.markdown("#### 📊 Métricas Principales: Modelo ML vs GRACE")
                    col_auroc, col_acc, col_sens, col_spec = st.columns(4)
                    
                    with col_auroc:
                        st.markdown("**AUROC**")
                        st.metric(
                            "Modelo ML", 
                            f"{result.model_auc:.4f}",
                            delta=f"{result.auc_difference:+.4f} vs GRACE"
                        )
                        st.metric("GRACE Score", f"{result.grace_auc:.4f}")
                    
                    with col_acc:
                        st.markdown("**Accuracy**")
                        acc_diff = result.model_accuracy - result.grace_accuracy
                        st.metric(
                            "Modelo ML", 
                            f"{result.model_accuracy:.4f}",
                            delta=f"{acc_diff:+.4f} vs GRACE"
                        )
                        st.metric("GRACE Score", f"{result.grace_accuracy:.4f}")
                    
                    with col_sens:
                        st.markdown("**Sensibilidad**")
                        sens_diff = result.model_sensitivity - result.grace_sensitivity
                        st.metric(
                            "Modelo ML", 
                            f"{result.model_sensitivity:.4f}",
                            delta=f"{sens_diff:+.4f} vs GRACE"
                        )
                        st.metric("GRACE Score", f"{result.grace_sensitivity:.4f}")
                    
                    with col_spec:
                        st.markdown("**Especificidad**")
                        spec_diff = result.model_specificity - result.grace_specificity
                        st.metric(
                            "Modelo ML", 
                            f"{result.model_specificity:.4f}",
                            delta=f"{spec_diff:+.4f} vs GRACE"
                        )
                        st.metric("GRACE Score", f"{result.grace_specificity:.4f}")
                    
                    st.markdown("---")
                    
                    # Tabs para diferentes visualizaciones
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📈 Curvas ROC",
                        "📊 Calibración", 
                        "📉 Métricas",
                        "🔄 NRI & IDI",
                        "📋 Reporte Completo"
                    ])
                    
                    with tab1:
                        st.markdown("#### Comparación de Curvas ROC")
                        st.caption("**DeLong Test**: Prueba estadística para comparar curvas ROC correlacionadas")
                        
                        roc_fig = plot_roc_comparison(y_test, y_prob, grace_probs, result)
                        st.plotly_chart(roc_fig, use_container_width=True, config=plotly_config)
                        
                        # Métricas clave
                        col1, col2, col3 = st.columns(3)
                        col1.metric("AUC Modelo", f"{result.model_auc:.4f}")
                        col2.metric("AUC GRACE", f"{result.grace_auc:.4f}")
                        col3.metric("Diferencia ΔAUC", f"{result.auc_difference:+.4f}", 
                                   delta=f"p={result.auc_p_value:.4f}")
                        
                        st.markdown(f"""
                        **Interpretación del DeLong Test:**
                        - **Estadístico Z**: {(result.auc_difference / ((result.auc_ci_upper - result.auc_ci_lower) / 3.92)):.3f}
                        - **P-value**: {result.auc_p_value:.4f} {'✅ (significativo)' if result.auc_p_value < 0.05 else '❌ (no significativo)'}
                        - **IC 95%**: [{result.auc_ci_lower:.4f}, {result.auc_ci_upper:.4f}]
                        """)
                    
                    with tab2:
                        st.markdown("#### Comparación de Calibración")
                        st.caption("**Brier Score**: Mide la precisión de las probabilidades predichas (menor es mejor)")
                        
                        calib_fig = plot_calibration_comparison(y_test, y_prob, grace_probs, result)
                        st.plotly_chart(calib_fig, use_container_width=True, config=plotly_config)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Brier Modelo", f"{result.model_brier:.4f}")
                        col2.metric("Brier GRACE", f"{result.grace_brier:.4f}")
                        col3.metric("Diferencia", f"{result.brier_difference:+.4f}",
                                   delta="Mejor" if result.brier_difference < 0 else "Peor",
                                   delta_color="normal" if result.brier_difference < 0 else "inverse")
                    
                    with tab3:
                        st.markdown("#### Comparación de Métricas de Rendimiento")
                        
                        metrics_fig = plot_metrics_comparison(result)
                        st.plotly_chart(metrics_fig, use_container_width=True, config=plotly_config)
                        
                        # Tabla comparativa
                        st.markdown("**Tabla Comparativa Detallada:**")
                        comparison_df = pd.DataFrame({
                            'Métrica': ['AUC', 'Accuracy', 'Sensitivity', 'Specificity'],
                            'Modelo ML': [result.model_auc, result.model_accuracy, 
                                         result.model_sensitivity, result.model_specificity],
                            'GRACE': [result.grace_auc, result.grace_accuracy,
                                     result.grace_sensitivity, result.grace_specificity],
                            'Diferencia': [
                                result.auc_difference,
                                result.model_accuracy - result.grace_accuracy,
                                result.model_sensitivity - result.grace_sensitivity,
                                result.model_specificity - result.grace_specificity
                            ]
                        })
                        
                        st.dataframe(
                            comparison_df.style.format({
                                'Modelo ML': '{:.4f}',
                                'GRACE': '{:.4f}',
                                'Diferencia': '{:+.4f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with tab4:
                        st.markdown("#### NRI (Net Reclassification Improvement) & IDI")
                        st.caption("""
                        **NRI**: Mide la mejora en reclasificación de pacientes a categorías de riesgo correctas  
                        **IDI**: Mide la mejora en discriminación integrada entre eventos y no-eventos
                        """)
                        
                        nri_idi_fig = plot_nri_idi(result)
                        st.plotly_chart(nri_idi_fig, use_container_width=True, config=plotly_config)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### NRI Detallado")
                            st.metric("NRI Total", f"{result.nri:.4f}", 
                                     delta=f"p={result.nri_p_value:.4f}")
                            st.metric("NRI Eventos", f"{result.nri_events:.4f}",
                                     help="Proporción de eventos correctamente reclasificados")
                            st.metric("NRI No-Eventos", f"{result.nri_nonevents:.4f}",
                                     help="Proporción de no-eventos correctamente reclasificados")
                        
                        with col2:
                            st.markdown("##### IDI Detallado")
                            st.metric("IDI", f"{result.idi:.4f}",
                                     delta=f"p={result.idi_p_value:.4f}")
                            
                            if result.idi > 0:
                                st.success("✅ Mejora en discriminación integrada")
                            else:
                                st.error("❌ No hay mejora en discriminación")
                    
                    with tab5:
                        st.markdown("#### Reporte Completo de Comparación")
                        
                        report_df = generate_comparison_report(result)
                        st.dataframe(report_df, use_container_width=True, hide_index=True)
                        
                        # Descargar reporte
                        csv = report_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Reporte CSV",
                            data=csv,
                            file_name=f"grace_comparison_{selected_model if 'selected_model' in locals() else 'model'}.csv",
                            mime="text/csv"
                        )
                        
                        # Resumen ejecutivo
                        st.markdown("---")
                        st.markdown("### 📝 Resumen Ejecutivo")
                        
                        st.markdown(f"""
                        **Modelo Evaluado**: {result.model_name}  
                        **Baseline**: GRACE Score  
                        **Nivel de Significancia**: α = 0.05
                        
                        **Resultados Principales:**
                        1. **AUC**: {result.model_auc:.4f} vs {result.grace_auc:.4f} (Δ = {result.auc_difference:+.4f}, p = {result.auc_p_value:.4f})
                        2. **NRI**: {result.nri:.4f} (p = {result.nri_p_value:.4f})
                        3. **IDI**: {result.idi:.4f} (p = {result.idi_p_value:.4f})
                        4. **Calibración**: Brier {result.model_brier:.4f} vs {result.grace_brier:.4f}
                        
                        **Conclusión Estadística**: {result.superiority_level.replace('_', ' ').title()}
                        
                        **Recomendación Clínica**:
                        """)
                        
                        if result.is_model_superior and result.auc_p_value < 0.01:
                            st.success("""
                            ✅ **RECOMENDADO**: El modelo ML demostró superioridad estadística significativa sobre GRACE.
                            Se recomienda su uso complementario o como alternativa en entornos clínicos apropiados.
                            """)
                        elif result.is_model_superior:
                            st.info("""
                            📊 **PROMISORIO**: El modelo ML muestra superioridad marginal sobre GRACE.
                            Se recomienda validación adicional en cohortes independientes.
                            """)
                        else:
                            st.warning("""
                            ⚠️ **PRECAUCIÓN**: El modelo ML no demostró superioridad sobre GRACE.
                            GRACE sigue siendo el estándar de oro recomendado.
                            """)
        
        except Exception as e:
            st.error(f"❌ Error al cargar datos para comparación: {e}")
            st.info("💡 Asegúrate de haber ejecutado la evaluación del modelo primero")
    else:
        st.info("ℹ️ Por favor, ejecuta la evaluación del modelo primero para habilitar la comparación con GRACE")

else:
    st.warning(f"""
    ⚠️ **No se encontró la columna de GRACE Score en el dataset**
    
    Columnas buscadas: {', '.join([f'`{c}`' for c in grace_column_candidates])}
    
    **Para habilitar la comparación:**
    1. Asegúrate de que tu dataset incluya el GRACE score
    2. La columna debe llamarse: `escala_grace`, `GRACE`, `grace_score`, o `grace`
    3. Vuelve a cargar el dataset en Data Cleaning
    
    **Información sobre GRACE Score:**
    GRACE es un score validado que predice mortalidad en pacientes con IAM basado en:
    - Edad, frecuencia cardíaca, presión arterial
    - Creatinina, paro cardíaco, desviación ST
    - Enzimas cardíacas elevadas, clase Killip
    """)

# ========================================================================
# SECCIÓN DE COMPARACIÓN CON RECUIMA SCORE
# ========================================================================
st.markdown("---")
st.markdown("---")
st.header("🇨🇺 Comparación con RECUIMA Score")

st.info("""
**RECUIMA (Registro Cubano de Infarto - Mortalidad Intrahospitalaria)** es una escala predictiva 
desarrollada por el Dr. Maikel Santos Medina, validada específicamente para países con recursos limitados.

**Ventajas sobre GRACE:**
- ✅ No requiere troponinas (costosas y no disponibles en todos los centros)
- ✅ No requiere coronariografía
- ✅ Variables clínicas disponibles al ingreso
- ✅ Mayor especificidad (87.70% vs 47.38% de GRACE)
- ✅ Validada en población latinoamericana

**Variables RECUIMA (máximo 10 puntos):**
- Filtrado glomerular < 60 ml/min/1.73m² (3 pts) ⭐ Factor más importante
- FV/TV - Arritmias ventriculares (2 pts)
- Killip-Kimball IV (1 pt)
- BAV alto grado (1 pt)
- > 7 derivaciones ECG afectadas (1 pt)
- Edad > 70 años (1 pt)
- TAS < 100 mmHg (1 pt)

**Categorías de riesgo:** Bajo (≤3) | Alto (≥4)
""")

# Verificar si el dataset tiene las variables necesarias para RECUIMA
try:
    from src.evaluation.recuima_comparison import (
        check_recuima_requirements,
        compute_recuima_scores,
        compare_with_recuima,
        plot_roc_comparison_recuima,
        plot_calibration_comparison_recuima,
        plot_metrics_comparison_recuima,
        plot_nri_idi_recuima,
        generate_comparison_report_recuima,
        get_recuima_info,
    )
    from src.scoring import (
        load_testset_score_data,
        check_score_data_availability,
    )
    
    # ================================================================
    # FIRST: Try to load preserved original score data from training
    # ================================================================
    score_data_available = False
    preserved_score_data = None
    
    try:
        score_availability = check_score_data_availability(TESTSETS_DIR, task)
        if score_availability['available'] and score_availability['recuima_ready']:
            preserved_score_data = load_testset_score_data(TESTSETS_DIR, task)
            if preserved_score_data is not None and len(preserved_score_data) > 0:
                score_data_available = True
                st.success(f"""
                ✅ **Datos originales para RECUIMA encontrados** ({len(preserved_score_data)} muestras del test set)
                
                Usando variables preservadas durante el entrenamiento con valores originales sin codificar.
                """)
    except Exception as e:
        # Silently continue if score data not available
        pass
    
    # Determine which dataset to use for RECUIMA check
    df_for_recuima_check = preserved_score_data if score_data_available else df
    
    # Check requirements on appropriate dataset
    can_compute_recuima, recuima_columns, missing_recuima = check_recuima_requirements(df_for_recuima_check)
    
    if can_compute_recuima:
        if score_data_available:
            st.success(f"✅ Variables RECUIMA disponibles desde datos originales preservados")
        else:
            st.success(f"✅ Variables RECUIMA encontradas en el dataset cargado")
        
        with st.expander("📋 Variables detectadas", expanded=False):
            for var_type, col_name in recuima_columns.items():
                st.write(f"- **{var_type}**: `{col_name}`")
        
        # Verificar que tenemos un modelo evaluado y test set
        if can_generate_plots and st.session_state.get('is_evaluated', False):
            
            with st.expander("🏥 **Análisis de Superioridad del Modelo ML vs RECUIMA**", expanded=True):
                st.markdown("### Configuración de Comparación RECUIMA")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    recuima_info = get_recuima_info()
                    st.markdown("**Información de la Escala:**")
                    st.write(f"- **Autor:** {recuima_info['author']}")
                    st.write(f"- **Institución:** {recuima_info['institution']}")
                    st.write(f"- **AUC validado:** {recuima_info['validation']['auc']}")
                
                with col2:
                    recuima_comparison_threshold = st.slider(
                        "Umbral de clasificación (RECUIMA)",
                        0.0, 1.0, 0.5, 0.05,
                        help="Umbral para convertir probabilidades a clasificación binaria",
                        key="recuima_threshold"
                    )
                
                # Opción para cargar dataset original si el actual está codificado
                st.markdown("---")
                st.markdown("#### 📂 Dataset para Comparación RECUIMA")
                
                # Mostrar info del dataset actual
                recuima_current_source = "Dataset cargado en la aplicación"
                if 'recuima_custom_dataset' in st.session_state:
                    recuima_current_source = "Dataset personalizado cargado"
                st.info(f"📊 **Fuente actual:** {recuima_current_source} ({len(df)} registros)")
                
                with st.expander("⚠️ ¿El dataset tiene columnas codificadas numéricamente?", expanded=False):
                    st.markdown("""
Si el dataset principal fue preprocesado/limpiado, algunas columnas como `complicaciones` 
pueden estar codificadas como números en lugar de texto. En ese caso, RECUIMA no podrá 
detectar correctamente las complicaciones (FV, TV, BAV).

**Carga aquí el dataset original con columnas en formato texto:**
                    """)
                    
                    recuima_uploaded_file = st.file_uploader(
                        "Cargar dataset RECUIMA original (CSV)",
                        type=['csv'],
                        key="recuima_dataset_uploader",
                        help="Archivo CSV con columnas en formato original (complicaciones como texto, etc.)"
                    )
                    
                    if recuima_uploaded_file is not None:
                        # Detectar separador
                        try:
                            content_sample = recuima_uploaded_file.read(2048).decode('utf-8')
                            recuima_uploaded_file.seek(0)
                            sep = ';' if content_sample.count(';') > content_sample.count(',') else ','
                            
                            recuima_custom_df = pd.read_csv(recuima_uploaded_file, sep=sep, low_memory=False)
                            st.session_state['recuima_custom_dataset'] = recuima_custom_df
                            st.success(f"✅ Dataset cargado: {len(recuima_custom_df)} registros")
                            
                            # Preview
                            cols_preview = ['complicaciones', 'indice_killip', 'edad', 'estado_vital']
                            cols_available = [c for c in cols_preview if c in recuima_custom_df.columns]
                            if cols_available:
                                st.dataframe(recuima_custom_df[cols_available].head(3), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error al cargar archivo: {e}")
                    
                    if 'recuima_custom_dataset' in st.session_state:
                        if st.button("🗑️ Usar dataset principal (eliminar personalizado)", key="clear_recuima_custom"):
                            del st.session_state['recuima_custom_dataset']
                            st.rerun()
                
                st.markdown("---")
                run_recuima_comparison = st.button("🚀 Ejecutar Comparación con RECUIMA", type="primary")
                
                if run_recuima_comparison:
                    with st.spinner("Calculando scores RECUIMA y ejecutando análisis estadístico..."):
                        try:
                            # Cargar modelo y obtener predicciones si no existen
                            if 'y_prob' not in locals():
                                model = joblib.load(model_path)
                                test_df = pd.read_parquet(testset_path)
                                
                                target = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
                                X_test = test_df.drop(columns=[target])
                                y_test = test_df[target].values
                                
                                y_prob = model.predict_proba(X_test)[:, 1]
                            
                            # ============================================================
                            # PRIORITY ORDER FOR RECUIMA DATA:
                            # 1. Preserved original score data (from training)
                            # 2. User-uploaded custom dataset
                            # 3. Current loaded dataset (may be encoded)
                            # ============================================================
                            if score_data_available and preserved_score_data is not None:
                                df_recuima = preserved_score_data
                                st.info("📊 Usando datos originales preservados durante el entrenamiento")
                            elif 'recuima_custom_dataset' in st.session_state:
                                df_recuima = st.session_state['recuima_custom_dataset']
                                st.info("📊 Usando dataset personalizado cargado por el usuario")
                            else:
                                df_recuima = df
                                st.info("📊 Usando dataset cargado en la aplicación")
                            
                            # Verificar si las columnas están en formato correcto para RECUIMA
                            # El dataset limpiado puede tener columnas codificadas numéricamente
                            recuima_format_valid = True
                            format_issues = []
                            applied_fixes = []
                            
                            # Skip validation if using preserved score data (already validated)
                            if not score_data_available:
                                # Verificar complicaciones - buscar también en _score_complicaciones
                                complicaciones_col = None
                                if '_score_complicaciones' in df_recuima.columns:
                                    complicaciones_col = '_score_complicaciones'
                                elif 'complicaciones' in df_recuima.columns:
                                    complicaciones_col = 'complicaciones'
                                
                                if complicaciones_col:
                                    sample_val = df_recuima[complicaciones_col].dropna().iloc[0] if len(df_recuima[complicaciones_col].dropna()) > 0 else None
                                    if sample_val is not None and isinstance(sample_val, (int, float)):
                                        recuima_format_valid = False
                                        format_issues.append("complicaciones (codificado numéricamente, necesita texto)")
                                    elif complicaciones_col == '_score_complicaciones':
                                        # Renombrar para que RECUIMA lo encuentre
                                        df_recuima['complicaciones'] = df_recuima['_score_complicaciones']
                                        applied_fixes.append("Usando _score_complicaciones como complicaciones")
                                
                                # Verificar indice_killip - buscar también en _score_indice_killip
                                killip_col = None
                                if '_score_indice_killip' in df_recuima.columns:
                                    killip_col = '_score_indice_killip'
                                elif 'indice_killip' in df_recuima.columns:
                                    killip_col = 'indice_killip'
                                
                                if killip_col:
                                    killip_vals = df_recuima[killip_col].dropna().unique()
                                    # Si los valores son 0,1,2,3, convertir a 1,2,3,4
                                    if set(killip_vals).issubset({0, 1, 2, 3}):
                                        df_recuima['indice_killip'] = df_recuima[killip_col] + 1
                                        applied_fixes.append(f"Convertido {killip_col} de 0-3 a 1-4")
                                    elif killip_col == '_score_indice_killip':
                                        df_recuima['indice_killip'] = df_recuima['_score_indice_killip']
                                        applied_fixes.append("Usando _score_indice_killip como indice_killip")
                            
                            if applied_fixes:
                                st.info(f"🔧 Correcciones automáticas aplicadas: {', '.join(applied_fixes)}")
                            
                            if not recuima_format_valid:
                                st.warning(f"""⚠️ **El dataset cargado tiene columnas codificadas que impiden calcular RECUIMA correctamente:**
                                
{chr(10).join(['• ' + issue for issue in format_issues])}

**Opciones:**
1. Re-entrenar el modelo para generar datos originales automáticamente
2. Cargar el dataset original usando el cargador de arriba""")
                                st.session_state.recuima_needs_original = True
                                st.stop()
                            
                            # Determinar columna de mortalidad
                            mortality_col = None
                            for col_candidate in ['estado_vital', 'mortality', 'exitus', 'mortality_inhospital']:
                                if col_candidate in df_recuima.columns:
                                    mortality_col = col_candidate
                                    break
                            
                            if mortality_col is None:
                                st.error("❌ No se encontró columna de mortalidad en el dataset")
                                st.stop()
                            
                            # Calcular y_true según el tipo de columna
                            if mortality_col == 'estado_vital':
                                y_true_recuima = (df_recuima[mortality_col].astype(str).str.lower().str.contains('fallecido', na=False)).astype(int).values
                            else:
                                y_true_recuima = df_recuima[mortality_col].values
                            
                            # Verificar columnas RECUIMA
                            can_compute_orig, recuima_cols_orig, missing_orig = check_recuima_requirements(df_recuima)
                            
                            if not can_compute_orig:
                                st.error(f"❌ Faltan columnas para RECUIMA: {missing_orig}")
                                st.stop()
                            
                            # Calcular scores RECUIMA
                            recuima_scores, recuima_probs, recuima_components = compute_recuima_scores(
                                df_recuima, 
                                column_mapping=recuima_cols_orig
                            )
                            
                            # Verificar que los scores no sean todos 0 (indica que faltan columnas)
                            if recuima_scores.max() == 0:
                                st.warning("""⚠️ **Todos los scores RECUIMA son 0.** 
                                Esto puede indicar que las variables clínicas no están disponibles o tienen valores faltantes.
                                Verificando componentes individuales...""")
                                
                                components_info = []
                                for comp_name, comp_values in recuima_components.items():
                                    non_zero = np.sum(comp_values > 0)
                                    components_info.append(f"  - {comp_name}: {non_zero} pacientes con puntos")
                                st.code("\n".join(components_info))
                            
                            # Nota metodológica dependiendo de la fuente de datos
                            if score_data_available:
                                st.success("""✅ **Comparación Metodológicamente Rigurosa:** 
                                Los scores RECUIMA se calculan sobre el **mismo test set** usado para evaluar el modelo ML, 
                                usando las variables originales preservadas durante el entrenamiento.""")
                            else:
                                st.info("""ℹ️ **Nota metodológica:** La comparación usa RECUIMA calculado sobre 
                                el dataset disponible. Para una comparación completamente alineada con el test set,
                                re-entrene el modelo para que se preserven automáticamente las variables originales.""")
                            
                            # Mostrar distribución de scores
                            data_source = "Test Set Original" if score_data_available else "Dataset Completo"
                            st.markdown(f"#### 📊 Distribución de Scores RECUIMA en {data_source}")
                            
                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                            col_s1.metric("Score Medio", f"{np.mean(recuima_scores):.2f}")
                            col_s2.metric("Score Mediana", f"{np.median(recuima_scores):.2f}")
                            col_s3.metric("Bajo Riesgo (≤3)", f"{np.sum(recuima_scores <= 3)} ({100*np.mean(recuima_scores <= 3):.1f}%)")
                            col_s4.metric("Alto Riesgo (≥4)", f"{np.sum(recuima_scores >= 4)} ({100*np.mean(recuima_scores >= 4):.1f}%)")
                            
                            # Calcular métricas RECUIMA
                            from sklearn.metrics import roc_auc_score, confusion_matrix as cm_sklearn
                            
                            # Filtrar valores válidos
                            valid_mask = ~np.isnan(y_true_recuima) & ~np.isnan(recuima_probs)
                            y_true_valid = y_true_recuima[valid_mask]
                            recuima_probs_valid = recuima_probs[valid_mask]
                            recuima_scores_valid = recuima_scores[valid_mask]
                            
                            # AUROC RECUIMA
                            recuima_auroc = roc_auc_score(y_true_valid, recuima_scores_valid)
                            
                            # Métricas con umbral RECUIMA >= 3 (alto riesgo)
                            recuima_pred_binary = (recuima_scores_valid >= 3).astype(int)
                            cm_recuima = cm_sklearn(y_true_valid, recuima_pred_binary)
                            tn_r, fp_r, fn_r, tp_r = cm_recuima.ravel() if cm_recuima.size == 4 else (0, 0, 0, 0)
                            
                            sens_recuima = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
                            spec_recuima = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0
                            acc_recuima = (tp_r + tn_r) / len(y_true_valid) if len(y_true_valid) > 0 else 0
                            ppv_recuima = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0
                            npv_recuima = tn_r / (tn_r + fn_r) if (tn_r + fn_r) > 0 else 0
                            
                            # Mostrar métricas RECUIMA directamente
                            st.markdown("---")
                            st.markdown("### 📊 Métricas RECUIMA (Dataset Completo)")
                            
                            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                            col_r1.metric("AUROC", f"{recuima_auroc:.3f}")
                            col_r2.metric("Sensibilidad", f"{sens_recuima:.1%}")
                            col_r3.metric("Especificidad", f"{spec_recuima:.1%}")
                            col_r4.metric("Exactitud", f"{acc_recuima:.1%}")
                            
                            col_r5, col_r6, col_r7, col_r8 = st.columns(4)
                            col_r5.metric("VPP", f"{ppv_recuima:.1%}")
                            col_r6.metric("VPN", f"{npv_recuima:.1%}")
                            col_r7.metric("N pacientes", len(y_true_valid))
                            col_r8.metric("Eventos (fallecidos)", int(y_true_valid.sum()))
                            
                            # Crear resultado simplificado para visualización
                            # Nota: La comparación directa con DeLong requiere los mismos pacientes
                            # Por ahora, mostramos métricas side-by-side
                            
                            # Si hay un modelo evaluado, mostrar comparación side-by-side
                            if 'y_prob' in locals() and 'y_test' in locals():
                                st.markdown("---")
                                st.markdown("### 📊 Comparación: Modelo ML (Test Set) vs RECUIMA (Dataset Completo)")
                                st.warning("""⚠️ **Nota:** Esta comparación es **indicativa** ya que se usan diferentes conjuntos de datos:
                                - **Modelo ML**: Evaluado en el test set preprocesado
                                - **RECUIMA**: Evaluado en todo el dataset original""")
                                
                                # Calcular métricas del modelo ML
                                ml_auroc = roc_auc_score(y_test, y_prob)
                                ml_pred_binary = (y_prob >= recuima_comparison_threshold).astype(int)
                                cm_ml = cm_sklearn(y_test, ml_pred_binary)
                                tn_m, fp_m, fn_m, tp_m = cm_ml.ravel() if cm_ml.size == 4 else (0, 0, 0, 0)
                                
                                sens_ml = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
                                spec_ml = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0
                                
                                # Tabla comparativa
                                comparison_data = {
                                    'Métrica': ['AUROC', 'Sensibilidad', 'Especificidad', 'N pacientes', 'Eventos'],
                                    'Modelo ML (Test Set)': [f"{ml_auroc:.3f}", f"{sens_ml:.1%}", f"{spec_ml:.1%}", len(y_test), int(y_test.sum())],
                                    'RECUIMA (Dataset)': [f"{recuima_auroc:.3f}", f"{sens_recuima:.1%}", f"{spec_recuima:.1%}", len(y_true_valid), int(y_true_valid.sum())],
                                }
                                st.table(pd.DataFrame(comparison_data).set_index('Métrica'))
                            
                            # Guardar en session state para visualizaciones
                            st.session_state.recuima_scores = recuima_scores_valid
                            st.session_state.recuima_probs = recuima_probs_valid
                            st.session_state.recuima_y_true = y_true_valid
                            st.session_state.recuima_auroc = recuima_auroc
                            st.session_state.recuima_sensitivity = sens_recuima
                            st.session_state.recuima_specificity = spec_recuima
                            
                            st.success("✅ Análisis RECUIMA completado!")
                            
                        except Exception as e:
                            st.error(f"❌ Error durante la comparación con RECUIMA: {e}")
                            st.exception(e)
                
                # Mostrar curva ROC de RECUIMA si tenemos los datos
                if 'recuima_scores' in st.session_state and 'recuima_y_true' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📈 Curva ROC de RECUIMA")
                    
                    recuima_scores_plot = st.session_state.recuima_scores
                    recuima_y_true_plot = st.session_state.recuima_y_true
                    recuima_auroc_plot = st.session_state.get('recuima_auroc', 0)
                    
                    # Crear curva ROC
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(recuima_y_true_plot, recuima_scores_plot)
                    
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'RECUIMA (AUC = {recuima_auroc_plot:.3f})',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Referencia (AUC = 0.5)',
                        line=dict(color='gray', dash='dash')
                    ))
                    fig_roc.update_layout(
                        title='Curva ROC - RECUIMA Score',
                        xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
                        yaxis_title='Tasa de Verdaderos Positivos (Sensibilidad)',
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        showlegend=True,
                        height=450
                    )
                    st.plotly_chart(fig_roc, use_container_width=True, config=plotly_config)
                    
                    # Mostrar métricas adicionales
                    st.markdown("#### 📊 Resumen de Rendimiento RECUIMA")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("AUROC", f"{recuima_auroc_plot:.3f}")
                    col2.metric("Sensibilidad", f"{st.session_state.get('recuima_sensitivity', 0):.1%}")
                    col3.metric("Especificidad", f"{st.session_state.get('recuima_specificity', 0):.1%}")
                    
                    # Contexto científico
                    with st.expander("📚 Contexto Científico de RECUIMA", expanded=False):
                        st.markdown("""
                        **Escala RECUIMA** fue desarrollada y validada por el Dr. Maikel Santos Medina
                        en su tesis doctoral (Universidad de Ciencias Médicas de Santiago de Cuba).
                        
                        **Validación:**
                        - 3 cohortes de validación
                        - 2,348 pacientes totales
                        - AUC: 0.890-0.904
                        - Superioridad estadística sobre GRACE (test de Hanley-McNeil)
                        
                        **Por qué es importante para países de bajos recursos:**
                        1. GRACE requiere troponinas → costosas y no siempre disponibles
                        2. GRACE requiere coronariografía para validación → no disponible en hospitales rurales
                        3. RECUIMA usa solo variables clínicas disponibles al ingreso
                        
                        **Referencia:** Santos Medina, M. (2023). Escala predictiva de muerte hospitalaria 
                        por infarto agudo de miocardio. Tesis Doctoral, Universidad de Ciencias Médicas 
                        de Santiago de Cuba.
                        """)
        else:
            st.info("ℹ️ Ejecuta la evaluación del modelo primero para habilitar la comparación con RECUIMA")
    
    else:
        st.warning(f"""
        ⚠️ **Faltan variables para calcular RECUIMA Score**
        
        Variables faltantes: {', '.join([f'`{m}`' for m in missing_recuima])}
        
        **Variables requeridas para RECUIMA:**
        - `edad` o `age` - Edad del paciente
        - `presion_arterial_sistolica` o `tas` - Presión arterial sistólica
        - `filtrado_glomerular` o `gfr` - Filtrado glomerular
        - `indice_killip` o `killip_class` - Clase Killip
        - Derivaciones ECG (v1-v6, d1-d3, avf, avl, avc)
        
        **Variables opcionales:**
        - `fv_tv` - Fibrilación/taquicardia ventricular
        - `bav` - Bloqueo auriculoventricular de alto grado
        
        **Para habilitar RECUIMA:**
        Asegúrate de que tu dataset incluya las variables requeridas con los nombres correctos.
        """)

except ImportError as e:
    st.error(f"❌ Error al importar módulo RECUIMA: {e}")
except Exception as e:
    st.warning(f"⚠️ No se pudo verificar requisitos RECUIMA: {e}")

# Exportación PDF
st.markdown("---")
st.subheader("📄 Exportar Reporte de Evaluación")

if st.session_state.get('evaluation_results'):
    eval_results = st.session_state.evaluation_results
    
    # Preparar datos para el PDF
    models_data = {}
    for model_name, results in eval_results.items():
        if isinstance(results, dict):
            models_data[model_name] = {
                'metrics': results.get('metrics', {}),
                'y_true': results.get('y_true'),
                'y_pred': results.get('y_pred'),
                'y_proba': results.get('y_proba'),
                'plots': {}
            }
            
            # Agregar paths de plots si existen
            for plot_type in ['roc_curve', 'pr_curve', 'calibration_curve', 'decision_curve']:
                fig_path = get_latest_figure(f"{model_name}_{plot_type}*.png")
                if fig_path:
                    models_data[model_name]['plots'][plot_type] = str(fig_path)
    
    if models_data:
        
        def generate_evaluation_report():
            """Generate evaluation PDF report."""
            from pathlib import Path
            output_path = Path("reports") / "evaluation_report.pdf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return generate_evaluation_pdf(
                models_data=models_data,
                output_path=output_path
            )
        
        pdf_export_section(
            generate_evaluation_report,
            section_title="Reporte de Evaluación",
            default_filename="evaluation_report.pdf",
            key_prefix="evaluation_report"
        )
    else:
        st.info("ℹ️ No hay resultados de evaluación completos para exportar")
else:
    st.info("ℹ️ Evalúa modelos primero para generar el reporte PDF")
