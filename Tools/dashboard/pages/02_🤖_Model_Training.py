"""Model Training page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from app import (
    display_model_list,
    get_state,
    initialize_state,
    set_state,
    sidebar_training_controls,
    train_models_with_progress,
)

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

# Training settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Training Configuration")

# Info about the rigorous pipeline (always active)
st.sidebar.info("""
🎓 **Pipeline Riguroso Activo**

Este dashboard SIEMPRE usa el pipeline académico completo:
• ✅ Validación cruzada estratificada repetida (≥30 corridas)
• ✅ Curvas de aprendizaje
• ✅ Comparación estadística (Shapiro-Wilk, t-test/Mann-Whitney)

La evaluación final (Bootstrap/Jackknife) se hace en el módulo de **Evaluación**.
""")

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
    # Show pipeline info (always rigorous)
    st.info("""
    ### 🎓 Pipeline de Experimentación Riguroso
    
    Este pipeline seguirá las mejores prácticas académicas:
    
    **FASE 1: Train + Validation**
    - ✅ Validación cruzada estratificada repetida (30+ corridas)
    - ✅ Estimación de μ (media) y σ (desviación) por modelo
    - ✅ Curvas de aprendizaje para diagnóstico
    
    **FASE 3: Comparación Estadística**
    - ✅ Prueba de normalidad (Shapiro-Wilk)
    - ✅ Test paramétrico (t-Student) o no paramétrico (Mann-Whitney)
    - ✅ Tamaño del efecto (Cohen's d)
    
    **FASE 2: Test (Estimado Final)**
    - ⚠️ Se realizará en el módulo de **Evaluación**
    - Bootstrap (1000 iteraciones con reemplazo)
    - Jackknife (eliminando 1 elemento)
    - Intervalos de confianza al 95%
    
    📊 Se generarán gráficos y reportes detallados en `models/`
    """)
    
    if st.button("🚀 Start Training", type="primary", width='stretch'):
        try:
            with st.spinner("Training models..."):
                save_paths = train_models_with_progress(
                    data_path=data_path,
                    task=task,
                    quick=quick,
                    imputer_mode=imputer_mode,
                    selected_models=selected_models,
                )
            
            # Update session state
            set_state("is_trained", True)
            set_state("last_train_task", task)
            set_state("last_train_models", list(save_paths.keys()))
            
            st.success(f"✅ Successfully trained {len(save_paths)} model(s)")
            
            # Display saved models
            with st.expander("View saved model paths"):
                for name, path in save_paths.items():
                    st.code(f"{name}: {path}", language="text")
        
        except FileNotFoundError as e:
            st.error(f"❌ Dataset file not found: {e}")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")
            st.exception(e)

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
