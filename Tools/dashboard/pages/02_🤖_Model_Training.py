"""Model Training page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd

from app import (
    display_model_list,
    get_state,
    initialize_state,
    set_state,
    sidebar_training_controls,
    train_models_with_progress,
)
from app.config import PLOTS_TRAINING_DIR
from src.data_load import get_latest_plot
from src.training import generate_training_pdf
from src.reporting import pdf_export_section

# Initialize
initialize_state()

# Page config
st.title("🤖 Model Training")
st.markdown("---")

# Check if data has been loaded
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    data_path = st.session_state.get('data_path')
    st.success("✅ Usando datos limpios del proceso de limpieza")
elif raw_data is not None:
    df = raw_data
    data_path = st.session_state.get('data_path')
    st.warning("⚠️ Usando datos crudos (se recomienda limpiar primero)")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Si no hay data_path o el path no existe, crear un archivo temporal
import tempfile
if not data_path or not Path(data_path).exists():
    st.info("ℹ️ Guardando datos en archivo temporal para el entrenamiento...")
    temp_dir = Path(tempfile.gettempdir())
    data_path = temp_dir / "streamlit_training_dataset.csv"
    df.to_csv(data_path, index=False)
    st.session_state.data_path = str(data_path)
    st.success(f"✅ Dataset guardado en: {data_path}")

# Get task from session state
task = st.session_state.get('target_column', 'mortality')
if task == 'exitus':
    task = 'mortality'

# Custom models section
st.sidebar.markdown("---")
st.sidebar.header("🔧 Custom Models")

use_custom_models = st.sidebar.checkbox(
    "Include Custom Models",
    value=False,
    help="Include custom models in training"
)

custom_models_list = []
if use_custom_models:
    from src.models.persistence import list_saved_models
    custom_models_dir = root_dir / "models" / "custom"
    custom_models_dir.mkdir(parents=True, exist_ok=True)
    
    available_custom = list_saved_models(custom_models_dir, include_info=True)
    
    if available_custom:
        st.sidebar.markdown(f"**Available: {len(available_custom)} model(s)**")
        
        custom_models_list = st.sidebar.multiselect(
            "Select Custom Models",
            [m["name"] for m in available_custom],
            help="Select which custom models to include"
        )
        
        if custom_models_list:
            st.sidebar.success(f"✅ {len(custom_models_list)} custom model(s) selected")
    else:
        st.sidebar.info("No custom models available. Upload in Custom Models page.")

# Training settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Training Configuration")



quick, imputer_mode, selected_models = sidebar_training_controls()

# Main content
st.subheader("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.metric("Task", task.capitalize())
    st.metric("Imputation", imputer_mode.capitalize())

with col2:
    st.metric("Quick Mode", "Enabled" if quick else "Disabled")
    st.metric("Models Selected", len(selected_models))

# Display selected models
if selected_models:
    st.info(f"📦 Selected models: {', '.join(selected_models)}")
else:
    st.warning("⚠️ No models selected for training")

st.markdown("---")

# Training section
st.subheader("Train Models")

if not selected_models:
    st.error("❌ Please select at least one model from the sidebar")
else:
    # Initialize training state
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    
    # Show button or training message
    if not st.session_state.is_training:
        start_button = st.button("🚀 Start Training", type="primary", use_container_width=True)
    else:
        st.info("⏳ **Training in progress, please wait...**")
    
    if not st.session_state.is_training and 'start_button' in locals() and start_button:
        # Set training flag
        st.session_state.is_training = True
        
        try:
            # Create containers for progress display
            progress_container = st.empty()
            status_container = st.empty()
            
            # Capture stdout to show progress
            import io
            from contextlib import redirect_stdout
            
            with status_container.container():
                st.markdown("### 📊 Progreso del Entrenamiento")
                progress_area = st.empty()
                
                # Redirect stdout
                output_buffer = io.StringIO()
                
                with redirect_stdout(output_buffer):
                    save_paths, experiment_results = train_models_with_progress(
                        data_path=data_path,
                        task=task,
                        quick=quick,
                        imputer_mode=imputer_mode,
                        selected_models=selected_models,
                    )
                
                # Get the output
                output = output_buffer.getvalue()
                
                # Display in expander
                with st.expander("📋 Ver detalles completos del entrenamiento", expanded=False):
                    st.code(output, language="text")
            
            # Update session state
            set_state("is_trained", True)
            set_state("last_train_task", task)
            set_state("last_train_models", list(save_paths.keys()))
            
            st.success(f"""
            ✅ **Entrenamiento completado exitosamente**
            
            - {len(save_paths)} modelo(s) entrenado(s)
            - Validación cruzada estratificada completada
            - Curvas de aprendizaje generadas
            - Comparación estadística realizada
            - Modelos guardados en `models/`
            """)
            
            # Display saved models
            with st.expander("📁 Ver rutas de modelos guardados"):
                for name, path in save_paths.items():
                    st.code(f"{name}: {path}", language="text")
            
            # Display learning curves if available
            if hasattr(st.session_state, 'learning_curve_paths') and st.session_state.learning_curve_paths:
                st.markdown("---")
                st.subheader("📈 Curvas de Aprendizaje")
                st.info("Las curvas de aprendizaje muestran cómo el rendimiento del modelo mejora con más datos de entrenamiento.")
                
                lc_paths = st.session_state.learning_curve_paths
                lc_results = st.session_state.get('learning_curve_results', {})
                
                # Create tabs for each model
                if len(lc_paths) > 0:
                    tabs = st.tabs([f"📊 {model}" for model in lc_paths.keys()])
                    
                    for tab, (model_name, img_path) in zip(tabs, lc_paths.items()):
                        with tab:
                            # Display image if PNG exists
                            if Path(img_path).exists():
                                st.image(img_path, use_container_width=True)
                            else:
                                # Try HTML version
                                html_path = img_path.replace('.png', '.html')
                                if Path(html_path).exists():
                                    with open(html_path, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    st.components.v1.html(html_content, height=650, scrolling=True)
                            
                            # Display statistics
                            if model_name in lc_results:
                                lc_res = lc_results[model_name]
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    final_train = lc_res.train_scores_mean[-1]
                                    st.metric("Score Final (Train)", f"{final_train:.4f}")
                                
                                with col2:
                                    final_val = lc_res.val_scores_mean[-1]
                                    st.metric("Score Final (Val)", f"{final_val:.4f}")
                                
                                with col3:
                                    gap = abs(final_train - final_val)
                                    st.metric("Gap Train-Val", f"{gap:.4f}")
                                
                                # Interpretation
                                if gap < 0.05:
                                    st.success("✅ **Buen ajuste**: Gap pequeño entre train y validación")
                                elif gap < 0.10:
                                    st.warning("⚠️ **Ligero sobreajuste**: Gap moderado")
                                else:
                                    st.error("🔴 **Sobreajuste significativo**: Gap grande, considerar regularización")
            
            # 🎉 Success! Show balloons
            st.balloons()
            st.success("🎉 **¡Entrenamiento completado exitosamente!**")
            
            # Show statistical comparison if available
            st.markdown("---")
            st.subheader("📊 Comparación Estadística de Modelos")
            
            # Get statistical results
            stat_results = experiment_results.get('statistical_comparison', {})
            
            if stat_results and len(selected_models) > 1:
                st.info("""
                **Análisis Estadístico:**
                - 🧪 Prueba de normalidad (Shapiro-Wilk) para verificar distribución
                - 📊 Test paramétrico (t-Student) si los datos son normales
                - 📈 Test no paramétrico (Mann-Whitney U) si no son normales
                - ⚖️ Determina si las diferencias entre modelos son estadísticamente significativas (p < 0.05)
                """)
                
                # Display comparison matrix
                from src.data_load import get_latest_plot
                matrix_plot = get_latest_plot(PLOTS_TRAINING_DIR, "comparison_matrix")
                
                if matrix_plot and matrix_plot.exists():
                    st.markdown("### Matriz de Comparaciones")
                    if matrix_plot.suffix == '.png':
                        st.image(str(matrix_plot), use_container_width=True)
                    elif matrix_plot.suffix == '.html':
                        with open(matrix_plot, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                
                # Display pairwise comparisons
                st.markdown("### Comparaciones por Pares")
                
                # Create dataframe with results
                comparison_data = []
                for (m1, m2), res in stat_results.items():
                    comparison_data.append({
                        "Modelo 1": m1,
                        "Modelo 2": m2,
                        "Test Usado": res.test_used,
                        "p-value": f"{res.p_value:.4f}",
                        "Significativo (p<0.05)": "✅ SÍ" if res.significant else "❌ NO",
                        "Diferencia de medias": f"{res.mean_diff:.4f}",
                        "Normalidad M1": "✓" if res.normality_p1 > 0.05 else "✗",
                        "Normalidad M2": "✓" if res.normality_p2 > 0.05 else "✗"
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Show individual comparison plots
                    with st.expander("📈 Ver gráficos de comparación individual"):
                        for (m1, m2), res in stat_results.items():
                            comp_plot = get_latest_plot(PLOTS_TRAINING_DIR, f"comparison_{m1}_vs_{m2}")
                            if comp_plot and comp_plot.exists():
                                st.markdown(f"**{m1} vs {m2}**")
                                if comp_plot.suffix == '.png':
                                    st.image(str(comp_plot), use_container_width=True)
                                elif comp_plot.suffix == '.html':
                                    with open(comp_plot, 'r', encoding='utf-8') as f:
                                        st.components.v1.html(f.read(), height=500, scrolling=True)
            elif len(selected_models) == 1:
                st.info("ℹ️ Selecciona al menos 2 modelos para ver la comparación estadística.")
            else:
                st.warning("⚠️ No se encontraron resultados de comparación estadística.")
        
        except FileNotFoundError as e:
            st.error(f"❌ Dataset file not found: {e}")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")
            st.exception(e)
        finally:
            # Reset training flag
            st.session_state.is_training = False

st.markdown("---")

# Display learning curves from previous training if available
if not get_state("is_trained") and hasattr(st.session_state, 'learning_curve_paths'):
    if st.session_state.learning_curve_paths:
        st.subheader("📈 Curvas de Aprendizaje (del último entrenamiento)")
        
        lc_paths = st.session_state.learning_curve_paths
        tabs = st.tabs([f"📊 {model}" for model in lc_paths.keys()])
        
        for tab, (model_name, img_path) in zip(tabs, lc_paths.items()):
            with tab:
                if Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    html_path = img_path.replace('.png', '.html')
                    if Path(html_path).exists():
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=650, scrolling=True)
        
        st.markdown("---")

# Display saved models section
st.subheader("Saved Models")

last_task = get_state("last_train_task")
if last_task and last_task != task:
    st.info(f"ℹ️ Last training was for task: {last_task}")

display_model_list(task)

# Training history/log
with st.expander("ℹ️ Training Notes"):
    st.markdown("""
    ### ⚙️ Configuración del Entrenamiento
    
    **Quick Mode:**
    - ✅ Búsqueda simplificada de hiperparámetros
    - ✅ Menos splits en CV (3×3 = 9 corridas en vez de 10×10 = 100)
    - ✅ Iteración rápida para depuración
    - ⚠️ Recomendado solo para exploración inicial
    
    **Estrategias de Imputación:**
    - **Iterative**: IterativeImputer de sklearn (MICE - Multiple Imputation by Chained Equations)
    - **KNN**: K-Nearest Neighbors imputation (busca valores similares)
    - **Simple**: Imputación básica (media/mediana/moda)
    
    **Tipos de Modelos Disponibles:**
    - 🌳 Decision Trees, Random Forest
    - 🚀 XGBoost (Gradient Boosting)
    - 📈 Logistic Regression
    - 🎯 Support Vector Machine (SVM)
    - 👥 K-Nearest Neighbors (KNN)
    - 📊 Naive Bayes
    
    ### 📋 Pipeline de Experimentación
    
    El **Pipeline Riguroso** implementa el proceso científico completo:
    
    1. **Validación Cruzada Estratificada Repetida**: Se entrena y evalúa cada modelo
       múltiples veces (≥30 corridas) para obtener estimaciones robustas de μ y σ.
       
    2. **Curvas de Aprendizaje**: Diagnostican sobreajuste/subajuste y la necesidad
       de más datos.
       
    3. **Comparación Estadística**: Determina si las diferencias entre modelos son
       estadísticamente significativas usando:
       - Prueba de normalidad (Shapiro-Wilk)
       - Test paramétrico (t-Student) si los datos son normales
       - Test no paramétrico (Mann-Whitney) si no lo son
       
    4. **Evaluación Final en Test Set**: Una vez seleccionado el mejor modelo:
       - Bootstrap (1000 iteraciones con reemplazo)
       - Jackknife (leave-one-out)
       - Intervalos de confianza al 95%
    
    📚 Ver documentación completa en `Tools/docs/EXPERIMENT_PIPELINE.md`
    """)

# Exportación PDF
st.markdown("---")
st.subheader("📄 Exportar Reporte de Entrenamiento")

if st.session_state.get('training_results'):
    
    def generate_training_report():
        """Generate training PDF report."""
        from pathlib import Path
        output_path = Path("reports") / "training_report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get models metadata from trained models or create empty dict
        models_metadata = {}
        if 'trained_models' in st.session_state:
            for model_name, model in st.session_state.trained_models.items():
                # Create basic metadata from model
                from src.models.metadata import ModelMetadata
                models_metadata[model_name] = ModelMetadata(
                    model_name=model_name,
                    model_type=type(model).__name__,
                    hyperparameters=model.get_params() if hasattr(model, 'get_params') else {}
                )
        
        return generate_training_pdf(
            training_results=st.session_state.training_results,
            models_metadata=models_metadata,
            output_path=output_path
        )
    
    pdf_export_section(
        generate_training_report,
        section_title="Reporte de Entrenamiento",
        default_filename="training_report.pdf",
        key_prefix="training_report"
    )
else:
    st.info("ℹ️ Entrena modelos primero para generar el reporte PDF")
