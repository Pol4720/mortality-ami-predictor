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

# Get task from session state
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
                
                # Display in expander
                with st.expander("📋 Ver detalles completos de la evaluación", expanded=False):
                    st.code(output, language="text")
            
            st.success(f"""
            ✅ **Evaluación completada para {selected_model}**
            
            - Todas las métricas calculadas (AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier)
            - Bootstrap y Jackknife ejecutados con todas las métricas
            - Intervalos de confianza al 95% disponibles
            - Gráficos generados
            """)
            st.session_state.is_evaluated = True
        
    except Exception as e:
        st.error(f"❌ Evaluation error: {e}")
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
                
                # Show resampling plot
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

# Check if we can generate interactive plots from testset
testset_path = get_latest_testset(selected_model if 'selected_model' in locals() else 'dtree', TESTSETS_DIR)
if not testset_path:
    testset_path = TESTSETS_DIR / f"testset_{task}.parquet"
    
model_path = MODELS_DIR / f"best_classifier_{task}.joblib"

can_generate_plots = testset_path and testset_path.exists() and model_path.exists()

if can_generate_plots:
    try:
        # Load model and test data
        model = joblib.load(model_path)
        test_df = pd.read_parquet(testset_path)
        
        target = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
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
        st.error(f"❌ Error loading model/data for interactive plots: {e}")
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

    # Statistical comparison plots
    st.markdown("##### Statistical Comparison Plots")
    stat_boxplot = get_latest_figure(f"stat_boxplot_{task}_*.png") or get_latest_figure(f"stat_boxplot_{task}.png")
    stat_violin = get_latest_figure(f"stat_violin_{task}_*.png") or get_latest_figure(f"stat_violin_{task}.png")
    stat_hist = get_latest_figure(f"stat_hist_{task}_*.png") or get_latest_figure(f"stat_hist_{task}.png")
    stat_matrix = get_latest_figure(f"stat_matrix_{task}_*.png") or get_latest_figure(f"stat_matrix_{task}.png")

    stat_cols = st.columns(4)
    with stat_cols[0]:
        if stat_boxplot and stat_boxplot.exists():
            st.image(str(stat_boxplot), caption="Boxplot", width='stretch')
        else:
            st.info("No boxplot available")
    with stat_cols[1]:
        if stat_violin and stat_violin.exists():
            st.image(str(stat_violin), caption="Violin Plot", width='stretch')
        else:
            st.info("No violin plot available")
    with stat_cols[2]:
        if stat_hist and stat_hist.exists():
            st.image(str(stat_hist), caption="Histogram", width='stretch')
        else:
            st.info("No histogram available")
    with stat_cols[3]:
        if stat_matrix and stat_matrix.exists():
            st.image(str(stat_matrix), caption="Comparison Matrix", width='stretch')
        else:
            st.info("No comparison matrix available")

    # Resampling results (Bootstrap/Jackknife)
    st.markdown("##### Resampling Results (Bootstrap/Jackknife)")
    resample_boot = get_latest_figure(f"bootstrap_{task}_*.png") or get_latest_figure(f"bootstrap_{task}.png")
    resample_jack = get_latest_figure(f"jackknife_{task}_*.png") or get_latest_figure(f"jackknife_{task}.png")
    resample_cols = st.columns(2)
    with resample_cols[0]:
        if resample_boot and resample_boot.exists():
            st.image(str(resample_boot), caption="Bootstrap Distribution", width='stretch')
        else:
            st.info("No bootstrap plot available")
    with resample_cols[1]:
        if resample_jack and resample_jack.exists():
            st.image(str(resample_jack), caption="Jackknife Distribution", width='stretch')
        else:
            st.info("No jackknife plot available")

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
    
    **Interactive Features:**
    - 🔍 **Zoom & Pan**: Click and drag to zoom into any region
    - 🖱️ **Hover**: See exact values for each data point
    - 🎯 **Threshold Adjustment**: Change classification threshold in Confusion Matrix tab
    - 💾 **Export**: Use camera icon to download plots as PNG
    - 🔄 **Reset**: Double-click to restore original view
    """)

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
