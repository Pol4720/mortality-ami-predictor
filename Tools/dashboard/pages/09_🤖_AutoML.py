"""AutoML Page - Automated Machine Learning.

This page provides a comprehensive AutoML interface for:
- Configuring AutoML search parameters
- Running automated model selection
- Viewing leaderboard of evaluated models
- Exporting best models
- Getting intelligent suggestions
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add src to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from app import initialize_state, get_state, set_state
from src.config import CONFIG

# Page config
st.set_page_config(
    page_title="AutoML",
    page_icon="🤖",
    layout="wide"
)

# Initialize state
initialize_state()

# Title
st.title("🤖 AutoML - Aprendizaje Automático")
st.markdown("""
**AutoML** busca automáticamente la mejor arquitectura de modelo y hiperparámetros
para tu dataset. Solo necesitas configurar el tiempo de búsqueda y el sistema
explorará múltiples algoritmos y configuraciones.
""")

st.markdown("---")

# Check data availability
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    st.success("✅ Usando datos limpios del proceso de limpieza")
elif raw_data is not None:
    df = raw_data
    st.warning("⚠️ Usando datos crudos (se recomienda limpiar primero)")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Get target column
target_col = st.session_state.get('target_column', 'mortality_inhospital')
if target_col not in df.columns:
    # Try to find it
    possible_targets = [c for c in df.columns if 'mortal' in c.lower() or 'exitus' in c.lower() or 'target' in c.lower()]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        st.error(f"❌ No se encontró la columna target. Columnas disponibles: {list(df.columns[:10])}")
        st.stop()

# Import AutoML components
try:
    from src.automl import (
        AutoMLClassifier, FLAMLClassifier, AutoMLConfig, AutoMLPreset,
        is_autosklearn_available, is_flaml_available,
        AutoMLSuggestions, analyze_dataset, get_suggestions,
        export_best_model, create_automl_report
    )
    from src.automl.export import create_automl_report
    
    AUTOML_AVAILABLE = is_autosklearn_available() or is_flaml_available()
    BACKEND = "auto-sklearn" if is_autosklearn_available() else ("flaml" if is_flaml_available() else None)
except ImportError as e:
    AUTOML_AVAILABLE = False
    BACKEND = None
    st.error(f"❌ Módulo AutoML no disponible: {e}")
    st.info("Instala las dependencias con: `pip install flaml[automl]` o `pip install auto-sklearn` (Linux)")
    st.stop()

# Display backend info
col1, col2 = st.columns([3, 1])
with col1:
    if BACKEND == "auto-sklearn":
        st.info("🐧 **Backend:** auto-sklearn (Linux)")
    elif BACKEND == "flaml":
        st.info("🪟 **Backend:** FLAML (Cross-platform)")
with col2:
    st.metric("Backend", BACKEND or "None")

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================

st.sidebar.header("⚙️ Configuración AutoML")

# Preset selector
preset = st.sidebar.selectbox(
    "Preset de búsqueda",
    ["quick", "balanced", "high_performance", "custom"],
    index=1,
    help="Configuración predefinida del tiempo y parámetros de búsqueda"
)

preset_descriptions = {
    "quick": "🚀 **Quick** (5 min): Exploración rápida",
    "balanced": "⚖️ **Balanced** (1 hora): Balance tiempo/rendimiento", 
    "high_performance": "🏆 **High Performance** (4 horas): Máximo rendimiento",
    "custom": "🔧 **Custom**: Configuración personalizada",
}
st.sidebar.markdown(preset_descriptions.get(preset, ""))

# Custom settings
if preset == "custom":
    time_budget = st.sidebar.slider(
        "Tiempo de búsqueda (minutos)",
        min_value=1,
        max_value=480,
        value=30,
        step=5,
        help="Tiempo total para la búsqueda de modelos"
    ) * 60  # Convert to seconds
    
    ensemble_size = st.sidebar.slider(
        "Tamaño del ensemble",
        min_value=1,
        max_value=100,
        value=50,
        help="Número de modelos en el ensemble final"
    )
else:
    # Default values based on preset
    preset_configs = {
        "quick": {"time_budget": 300, "ensemble_size": 10},
        "balanced": {"time_budget": 3600, "ensemble_size": 50},
        "high_performance": {"time_budget": 14400, "ensemble_size": 100},
    }
    time_budget = preset_configs[preset]["time_budget"]
    ensemble_size = preset_configs[preset]["ensemble_size"]

# Metric selection
metric = st.sidebar.selectbox(
    "Métrica de optimización",
    ["roc_auc", "accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    index=0,
    help="Métrica a optimizar durante la búsqueda"
)

# Estimator selection (for FLAML)
if BACKEND == "flaml":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Estimadores")
    
    all_estimators = ["lgbm", "xgboost", "rf", "extra_tree", "catboost", "kneighbor", "lrl1", "lrl2"]
    default_estimators = ["lgbm", "xgboost", "rf", "extra_tree"]
    
    selected_estimators = st.sidebar.multiselect(
        "Estimadores a probar",
        all_estimators,
        default=default_estimators,
        help="Selecciona qué algoritmos incluir en la búsqueda"
    )
else:
    selected_estimators = None

# Display config summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Resumen")
st.sidebar.markdown(f"- **Tiempo:** {time_budget // 60} minutos")
st.sidebar.markdown(f"- **Ensemble:** {ensemble_size} modelos")
st.sidebar.markdown(f"- **Métrica:** {metric}")
if selected_estimators:
    st.sidebar.markdown(f"- **Estimadores:** {len(selected_estimators)}")

# ============================================================================
# MAIN CONTENT - Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Entrenar AutoML",
    "📊 Leaderboard",
    "💡 Sugerencias",
    "📥 Exportar"
])

# ============================================================================
# TAB 1: Train AutoML
# ============================================================================

with tab1:
    st.subheader("🚀 Entrenar Modelo AutoML")
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Muestras", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Target", target_col)
    with col4:
        if target_col in df.columns:
            class_dist = df[target_col].value_counts()
            ratio = class_dist.iloc[0] / class_dist.iloc[-1] if len(class_dist) > 1 else 1
            st.metric("Desbalance", f"{ratio:.1f}:1")
    
    st.markdown("---")
    
    # Training progress container
    if 'automl_model' not in st.session_state:
        st.session_state.automl_model = None
    if 'automl_training' not in st.session_state:
        st.session_state.automl_training = False
    
    # Start training button
    if not st.session_state.automl_training:
        col1, col2 = st.columns([1, 3])
        with col1:
            start_btn = st.button("🚀 Iniciar AutoML", type="primary", use_container_width=True)
        with col2:
            st.info(f"⏱️ Tiempo estimado: **{time_budget // 60} minutos**")
        
        if start_btn:
            st.session_state.automl_training = True
            st.rerun()
    
    if st.session_state.automl_training:
        st.warning("⏳ **Entrenamiento en progreso...** No cierres esta página.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()
        
        # Initialize log list in session state
        if 'automl_logs' not in st.session_state:
            st.session_state.automl_logs = []
        
        # Prepare data
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values for AutoML
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "Unknown")
        
        # Progress callback with log capture
        def progress_callback(message: str, progress: float):
            progress_bar.progress(progress)
            status_text.markdown(f"**{message}**")
            # Add to logs
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.automl_logs.append(f"[{timestamp}] {message}")
            # Show last 10 logs
            with log_container.container():
                st.markdown("#### 📋 Log de Entrenamiento")
                log_text = "\n".join(st.session_state.automl_logs[-15:])
                st.code(log_text, language="text")
        
        try:
            start_time = time.time()
            
            # Create AutoML model based on backend
            if BACKEND == "flaml":
                automl = FLAMLClassifier(
                    time_budget=time_budget,
                    metric=metric,
                    estimator_list=selected_estimators,
                    ensemble=ensemble_size > 1,
                    verbose=2,  # Enable verbose logging
                    name="AutoML-Dashboard",
                    progress_callback=progress_callback,
                )
            else:
                automl = AutoMLClassifier(
                    preset=AutoMLPreset(preset) if preset != "custom" else AutoMLPreset.CUSTOM,
                    time_left_for_this_task=time_budget,
                    ensemble_size=ensemble_size,
                    metric=metric,
                    name="AutoML-Dashboard",
                    progress_callback=progress_callback,
                )
            
            # Fit
            status_text.markdown("**🔄 Iniciando búsqueda de modelos...**")
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Iniciando FLAML AutoML...")
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️ Time budget: {time_budget}s")
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Métrica: {metric}")
            if selected_estimators:
                st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Estimadores: {', '.join(selected_estimators)}")
            
            automl.fit(X, y)
            
            elapsed = time.time() - start_time
            
            # Get training logs from model
            if hasattr(automl, 'get_training_logs'):
                training_logs = automl.get_training_logs()
                for log in training_logs[-20:]:  # Last 20 logs
                    if log not in st.session_state.automl_logs:
                        st.session_state.automl_logs.append(log)
            
            # Store results
            st.session_state.automl_model = automl
            st.session_state.automl_training = False
            st.session_state.automl_elapsed = elapsed
            st.session_state.automl_feature_names = feature_cols
            
            progress_bar.progress(1.0)
            status_text.markdown("**✅ ¡Entrenamiento completado!**")
            
            # Final log entry
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Completado: {automl.best_estimator_} (score={-automl.best_loss_:.4f})")
            
            st.success(f"""
            ### ✅ AutoML Completado
            
            - **Tiempo:** {elapsed:.1f} segundos
            - **Backend:** {BACKEND}
            - **Mejor modelo:** {getattr(automl, 'best_estimator_', 'ensemble')}
            - **Mejor score:** {-getattr(automl, 'best_loss_', 0):.4f}
            """)
            
            st.balloons()
            
        except Exception as e:
            st.session_state.automl_training = False
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
            st.error(f"❌ Error durante el entrenamiento: {e}")
            import traceback
            with st.expander("Ver detalles del error"):
                st.code(traceback.format_exc())
    
    # Show results if model exists
    if st.session_state.automl_model is not None and not st.session_state.automl_training:
        st.markdown("---")
        st.subheader("📈 Resultados del Entrenamiento")
        
        automl = st.session_state.automl_model
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if hasattr(automl, 'best_estimator_'):
                st.metric("Mejor Estimador", automl.best_estimator_)
        with col2:
            if hasattr(automl, 'best_loss_'):
                st.metric("Mejor Score", f"{-automl.best_loss_:.4f}")
        with col3:
            if hasattr(automl, 'fit_time_'):
                st.metric("Tiempo Total", f"{automl.fit_time_:.1f}s")
        
        # Best config
        if hasattr(automl, 'best_config_'):
            with st.expander("🔧 Mejor Configuración"):
                st.json(automl.best_config_)
        
        # Training logs
        if 'automl_logs' in st.session_state and st.session_state.automl_logs:
            with st.expander("📋 Logs de Entrenamiento", expanded=False):
                st.code("\n".join(st.session_state.automl_logs), language="text")
                
                # Download logs button
                log_text = "\n".join(st.session_state.automl_logs)
                st.download_button(
                    label="📥 Descargar Logs",
                    data=log_text,
                    file_name=f"automl_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ============================================================================
# TAB 2: Leaderboard
# ============================================================================

with tab2:
    st.subheader("📊 Leaderboard de Modelos")
    
    if st.session_state.automl_model is None:
        st.info("🔍 Entrena un modelo AutoML primero para ver el leaderboard")
    else:
        automl = st.session_state.automl_model
        
        if hasattr(automl, 'get_leaderboard'):
            leaderboard = automl.get_leaderboard()
            if leaderboard is not None and len(leaderboard) > 0:
                st.dataframe(leaderboard, use_container_width=True)
                
                # Visualize scores
                if 'mean_score' in leaderboard.columns:
                    fig = px.bar(
                        leaderboard.head(10),
                        x=leaderboard.index[:10],
                        y='mean_score',
                        title="Top 10 Modelos por Score",
                        labels={'x': 'Modelo', 'mean_score': 'Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de leaderboard disponibles")
        
        # Search history for FLAML
        if hasattr(automl, 'get_search_history'):
            history = automl.get_search_history()
            if len(history) > 0:
                st.markdown("### 🔍 Historial de Búsqueda")
                st.dataframe(history, use_container_width=True)

# ============================================================================
# TAB 3: Suggestions
# ============================================================================

with tab3:
    st.subheader("💡 Sugerencias Inteligentes")
    
    st.markdown("""
    El sistema analiza las características de tu dataset y sugiere técnicas
    que podrían mejorar el rendimiento del modelo.
    """)
    
    if st.button("🔍 Analizar Dataset", type="primary"):
        with st.spinner("Analizando dataset..."):
            # Analyze
            suggestions_engine = AutoMLSuggestions()
            analysis = suggestions_engine.analyze_dataset(df, target_col, task="classification")
            suggestions = suggestions_engine.generate_suggestions()
            
            # Store for display
            st.session_state.dataset_analysis = analysis
            st.session_state.suggestions = suggestions
    
    if 'dataset_analysis' in st.session_state:
        analysis = st.session_state.dataset_analysis
        suggestions = st.session_state.suggestions
        
        # Display analysis
        st.markdown("### 📊 Análisis del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Muestras", analysis.n_samples)
        with col2:
            st.metric("Features Numéricos", analysis.n_numeric_features)
        with col3:
            st.metric("Features Categóricos", analysis.n_categorical_features)
        with col4:
            st.metric("% Datos Faltantes", f"{analysis.missing_percentage:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Desbalanceado", "Sí" if analysis.is_imbalanced else "No")
        with col2:
            st.metric("Ratio de Clases", f"{analysis.imbalance_ratio:.1f}:1")
        
        st.markdown("---")
        
        # Display suggestions
        st.markdown("### 💡 Sugerencias")
        
        if not suggestions:
            st.success("✅ No se detectaron problemas significativos en el dataset")
        else:
            # Group by priority
            from src.automl.suggestions import Priority
            
            high_priority = [s for s in suggestions if s.priority == Priority.HIGH]
            medium_priority = [s for s in suggestions if s.priority == Priority.MEDIUM]
            low_priority = [s for s in suggestions if s.priority == Priority.LOW]
            
            if high_priority:
                st.markdown("#### 🔴 Alta Prioridad")
                for s in high_priority:
                    with st.expander(f"🔴 {s.title}", expanded=True):
                        st.markdown(f"**{s.description}**")
                        st.markdown(f"**Razón:** {s.reason}")
                        st.markdown(f"**Beneficio esperado:** {s.expected_benefit}")
                        if s.module_link:
                            st.markdown(f"**Módulo:** `{s.module_link}`")
                        if s.code_example:
                            st.code(s.code_example, language="python")
            
            if medium_priority:
                st.markdown("#### 🟡 Prioridad Media")
                for s in medium_priority:
                    with st.expander(f"🟡 {s.title}"):
                        st.markdown(f"**{s.description}**")
                        st.markdown(f"**Razón:** {s.reason}")
                        st.markdown(f"**Beneficio esperado:** {s.expected_benefit}")
                        if s.module_link:
                            st.markdown(f"**Módulo:** `{s.module_link}`")
            
            if low_priority:
                st.markdown("#### 🟢 Baja Prioridad")
                for s in low_priority:
                    with st.expander(f"🟢 {s.title}"):
                        st.markdown(f"**{s.description}**")
                        st.markdown(f"**Razón:** {s.reason}")

# ============================================================================
# TAB 4: Export
# ============================================================================

with tab4:
    st.subheader("📥 Exportar Modelo")
    
    if st.session_state.automl_model is None:
        st.info("🔍 Entrena un modelo AutoML primero para exportarlo")
    else:
        automl = st.session_state.automl_model
        
        st.markdown("""
        Exporta el mejor modelo encontrado por AutoML para usarlo en producción
        o en otras partes de la aplicación.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Nombre del modelo",
                value="automl_best_model",
                help="Nombre para el archivo del modelo"
            )
        
        with col2:
            export_dir = st.text_input(
                "Directorio de exportación",
                value=str(Path(CONFIG.models_dir) / "automl"),
                help="Directorio donde guardar el modelo"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Exportar Mejor Modelo", type="primary", use_container_width=True):
                try:
                    output_path = export_best_model(
                        automl_model=automl,
                        output_dir=export_dir,
                        model_name=model_name,
                        include_metadata=True,
                        training_data=df,
                        target_column=target_col,
                    )
                    st.success(f"✅ Modelo exportado: `{output_path}`")
                except Exception as e:
                    st.error(f"❌ Error al exportar: {e}")
        
        with col2:
            if st.button("📄 Generar Reporte", use_container_width=True):
                try:
                    report = create_automl_report(automl)
                    st.text_area("Reporte AutoML", report, height=400)
                    
                    # Download button
                    st.download_button(
                        "📥 Descargar Reporte",
                        report,
                        file_name="automl_report.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error al generar reporte: {e}")
        
        # Feature importance
        if hasattr(automl, 'get_feature_importance'):
            importance = automl.get_feature_importance()
            if importance is not None and len(importance) > 0:
                st.markdown("---")
                st.markdown("### 📊 Importancia de Features")
                
                feature_names = st.session_state.get('automl_feature_names', [f"F{i}" for i in range(len(importance))])
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importance)],
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 20 Features más Importantes"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>AutoML Module</strong> | Mortality AMI Predictor</p>
<p>💡 <em>AutoML busca automáticamente la mejor configuración para tu dataset</em></p>
</div>
""", unsafe_allow_html=True)
