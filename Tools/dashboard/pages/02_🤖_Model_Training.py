"""Model Training page."""
from __future__ import annotations

import io
import sys
import json  # ✅ AÑADIR IMPORT
import tempfile
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sklearn.decomposition import PCA

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
from src.features import ICATransformer

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

# Custom models section (mantener en sidebar)
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

# ==================== CHECKPOINTING SECTION (MOVED TO MAIN AREA) ====================
st.subheader("💾 Checkpointing Configuration")

col1, col2 = st.columns([2, 1])

with col1:
    use_checkpointing = st.checkbox(
        "Enable Checkpointing (Recommended)",
        value=True,
        help="Guarda progreso fold-a-fold. Permite reanudar entrenamiento si se interrumpe."
    )

with col2:
    if use_checkpointing:
        st.success("✅ Activo")
    else:
        st.error("❌ Desactivado")

# Initialize selected_checkpoint_id
selected_checkpoint_id = None

if use_checkpointing:
    st.info("""
    **¿Qué hace el checkpointing?**
    - 💾 Guarda progreso después de cada fold completado
    - ♻️ Permite reanudar entrenamiento si se interrumpe (fallo, cierre accidental, etc.)
    - 🧹 Se limpia automáticamente al completar con éxito
    - 📂 Ubicación: `Tools/dashboard/checkpoints/`
    """)
    
    # Show existing checkpoints
    checkpoint_dir = root_dir / "dashboard" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_files = list(checkpoint_dir.glob(f"training_{task}_*_state.json"))
    
    if checkpoint_files:
        st.markdown("### 📂 Checkpoints Existentes")
        
        # Parse and display checkpoints
        checkpoint_data = []
        for ckpt_file in sorted(checkpoint_files, reverse=True):
            try:
                with open(ckpt_file, 'r') as f:
                    state = json.load(f)
                
                exp_id = state.get('experiment_id', 'unknown')
                created = state.get('created_at', 'N/A')[:19]
                updated = state.get('last_updated', 'N/A')[:19]
                completed = len(state.get('completed_models', []))
                total = len(state.get('models', []))
                
                # Get current model progress
                current_model = state.get('current_model')
                current_folds = 0
                if current_model:
                    current_results = state.get('current_model_results', {}).get(current_model, {})
                    current_folds = len(current_results.get('completed_folds', []))
                
                checkpoint_data.append({
                    'Experiment ID': exp_id,
                    'Creado': created,
                    'Última actualización': updated,
                    'Modelos completados': f"{completed}/{total}",
                    'Modelo actual': current_model or 'N/A',
                    'Folds completados': current_folds,
                    'Archivo': ckpt_file.name
                })
            except Exception as e:
                st.warning(f"⚠️ Error leyendo {ckpt_file.name}: {e}")
        
        if checkpoint_data:
            # Display as dataframe
            checkpoint_df = pd.DataFrame(checkpoint_data)
            st.dataframe(checkpoint_df, use_container_width=True, hide_index=True)
            
            # ✅ CORREGIDO: Allow user to select checkpoint to resume
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_checkpoint_option = st.selectbox(
                    "¿Qué deseas hacer?",
                    ["🆕 Iniciar nuevo entrenamiento"] + [f"♻️ Resumir: {d['Experiment ID']}" for d in checkpoint_data],
                    key="checkpoint_selector"
                )
            
            with col2:
                if st.button("🗑️ Limpiar todos", type="secondary", use_container_width=True, key="clean_all_checkpoints"):
                    try:
                        for ckpt_file in checkpoint_files:
                            # Delete state file
                            ckpt_file.unlink()
                            
                            # Delete associated files
                            exp_id = ckpt_file.stem.replace('_state', '')
                            for f in checkpoint_dir.glob(f"{exp_id}*"):
                                try:
                                    f.unlink()
                                except Exception:
                                    pass
                        
                        st.success("✅ Todos los checkpoints eliminados")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            # ✅ AÑADIR: Botón para eliminar checkpoint individual
            if selected_checkpoint_option.startswith("♻️"):
                selected_checkpoint_id = selected_checkpoint_option.split(": ")[1]
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.success(f"✅ Se reanudará: `{selected_checkpoint_id}`")
                with col2:
                    if st.button("🗑️ Eliminar", type="secondary", use_container_width=True, key="delete_selected"):
                        try:
                            # Find and delete this specific checkpoint
                            for ckpt_file in checkpoint_files:
                                state_exp_id = ckpt_file.stem.replace('_state', '')
                                if state_exp_id == selected_checkpoint_id:
                                    # Delete state file
                                    ckpt_file.unlink()
                                    
                                    # Delete associated files
                                    for f in checkpoint_dir.glob(f"{selected_checkpoint_id}*"):
                                        try:
                                            f.unlink()
                                        except Exception:
                                            pass
                                    break
                            
                            st.success(f"✅ Checkpoint `{selected_checkpoint_id}` eliminado")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            else:
                selected_checkpoint_id = None
                st.info("🆕 Se iniciará un nuevo experimento")
        else:
            selected_checkpoint_id = None
    else:
        st.info("ℹ️ No hay checkpoints existentes para esta tarea")
        selected_checkpoint_id = None
else:
    st.warning("⚠️ **Advertencia:** Sin checkpointing, si el entrenamiento se interrumpe, perderás todo el progreso.")
    selected_checkpoint_id = None

st.markdown("---")

# ==================== FEATURE TRANSFORMATION SELECTOR ====================
st.subheader("🔄 Feature Transformation")

with st.expander("ℹ️ ¿Qué transformación usar?", expanded=False):
    st.markdown("""
    **Opciones de transformación de features:**
    
    - **🔤 Original Features:** Entrena con las variables originales sin transformación.
      - ✅ Interpretabilidad directa de features
      - ✅ No requiere procesamiento adicional
      - ❌ Alta dimensionalidad si hay muchas variables
    
    - **📊 PCA (Principal Component Analysis):** Reducción de dimensionalidad maximizando varianza.
      - ✅ Reduce multicolinealidad
      - ✅ Menor dimensionalidad = entrenamiento más rápido
      - ✅ Componentes ordenados por importancia (varianza)
      - ❌ Pérdida de interpretabilidad directa
      - 💡 Mejor para datos Gaussianos / lineales
    
    - **🧬 ICA (Independent Component Analysis):** Separación de fuentes independientes.
      - ✅ Encuentra patrones no-Gaussianos
      - ✅ Componentes estadísticamente independientes
      - ✅ Útil para separar señales mezcladas
      - ❌ No ordena componentes por importancia
      - 💡 Mejor para datos no-Gaussianos con múltiples fuentes
    
    **El transformer será guardado junto con el modelo para aplicarlo automáticamente en predicciones.**
    """)

transformation_type = st.radio(
    "Selecciona tipo de features para entrenamiento:",
    ["🔤 Original Features", "📊 PCA Components", "🧬 ICA Components"],
    help="El modelo se entrenará con el tipo de features seleccionado"
)

# Initialize transformation session state
if 'transformation_applied' not in st.session_state:
    st.session_state.transformation_applied = False
    st.session_state.transformer = None
    st.session_state.transformed_df = None
    st.session_state.transformation_params = {}

# Configuration based on transformation type
if transformation_type == "📊 PCA Components" or transformation_type == "🧬 ICA Components":
    st.markdown("### ⚙️ Configuración de Transformación")
    
    col1, col2, col3 = st.columns(3)
    
    # Get numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if task in numeric_cols:
        numeric_cols.remove(task)
    
    n_features = len(numeric_cols)
    
    with col1:
        if transformation_type == "📊 PCA Components":
            pca_mode = st.radio(
                "Modo de selección",
                ["Varianza", "Número fijo"],
                help="Varianza: selecciona automáticamente | Número fijo: especifica cantidad"
            )
            
            if pca_mode == "Varianza":
                variance_threshold = st.slider(
                    "Varianza acumulada deseada",
                    0.70, 0.99, 0.95, 0.01,
                    format="%.2f",
                    help="Porcentaje de varianza a capturar"
                )
                n_components = None
            else:
                n_components = st.slider(
                    "Número de componentes",
                    2, min(20, n_features), min(10, n_features),
                    help="Componentes PCA a extraer"
                )
                variance_threshold = None
        
        else:  # ICA
            n_components = st.slider(
                "Número de componentes",
                2, min(20, n_features), min(10, n_features),
                help="Componentes independientes a extraer"
            )
    
    with col2:
        if transformation_type == "🧬 ICA Components":
            ica_algorithm = st.selectbox(
                "Algoritmo ICA",
                ["parallel", "deflation"],
                help="parallel: simultáneo | deflation: secuencial"
            )
            
            ica_fun = st.selectbox(
                "Función de contraste",
                ["logcosh", "exp", "cube"],
                help="logcosh: general | exp: super-Gaussiano | cube: sub-Gaussiano"
            )
    
    with col3:
        standardize = st.checkbox(
            "Estandarizar datos",
            value=True,
            help="Recomendado para PCA/ICA (escala variables)"
        )
        
        if transformation_type == "🧬 ICA Components":
            whiten = st.checkbox(
                "Whitening",
                value=True,
                help="Pre-procesamiento para ICA (recomendado)"
            )
    
    # Apply transformation button
    if st.button("🔄 Aplicar Transformación", type="secondary", use_container_width=True):
        with st.spinner(f"Aplicando {'PCA' if transformation_type == '📊 PCA Components' else 'ICA'}..."):
            try:
                # Prepare data (only numeric columns, drop NaNs)
                df_for_transform = df[numeric_cols].dropna()
                
                if len(df_for_transform) == 0:
                    st.error("❌ No hay datos numéricos válidos para transformar")
                    st.stop()
                
                if transformation_type == "📊 PCA Components":
                    from sklearn.preprocessing import StandardScaler
                    
                    # Standardize if requested
                    if standardize:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(df_for_transform)
                    else:
                        scaler = None
                        X_scaled = df_for_transform.values
                    
                    # Apply PCA
                    if pca_mode == "Varianza":
                        pca = PCA(n_components=variance_threshold, random_state=42)
                    else:
                        pca = PCA(n_components=n_components, random_state=42)
                    
                    X_transformed = pca.fit_transform(X_scaled)
                    
                    # Create DataFrame with component names
                    component_names = [f'PC{i+1}' for i in range(pca.n_components_)]
                    transformed_df = pd.DataFrame(
                        X_transformed,
                        columns=component_names,
                        index=df_for_transform.index
                    )
                    
                    # Add target column back
                    transformed_df[task] = df.loc[df_for_transform.index, task]
                    
                    # Store transformer and params
                    st.session_state.transformer = {'pca': pca, 'scaler': scaler}
                    st.session_state.transformation_params = {
                        'type': 'pca',
                        'n_components': pca.n_components_,
                        'variance_explained': pca.explained_variance_ratio_.sum(),
                        'standardize': standardize,
                        'original_features': numeric_cols
                    }
                    
                    st.success(f"""
                    ✅ **PCA aplicado exitosamente**
                    
                    - Componentes: {pca.n_components_}
                    - Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.2f}%
                    - Variables originales: {len(numeric_cols)}
                    """)
                
                else:  # ICA
                    from src.features import ICATransformer
                    
                    # Apply ICA
                    ica = ICATransformer(
                        n_components=n_components,
                        algorithm=ica_algorithm,
                        fun=ica_fun,
                        whiten=whiten,
                        random_state=42
                    )
                    
                    X_transformed = ica.fit_transform(df_for_transform)
                    transformed_df = X_transformed.copy()
                    
                    # Add target column back
                    transformed_df[task] = df.loc[df_for_transform.index, task]
                    
                    # Store transformer and params
                    st.session_state.transformer = ica
                    kurtosis_vals = [abs(x) for x in ica.kurtosis_]
                    st.session_state.transformation_params = {
                        'type': 'ica',
                        'n_components': n_components,
                        'algorithm': ica_algorithm,
                        'fun': ica_fun,
                        'whiten': whiten,
                        'kurtosis_mean': np.mean(kurtosis_vals),
                        'original_features': numeric_cols
                    }
                    
                    st.success(f"""
                    ✅ **ICA aplicado exitosamente**
                    
                    - Componentes independientes: {n_components}
                    - Kurtosis promedio: {np.mean(kurtosis_vals):.3f}
                    - Variables originales: {len(numeric_cols)}
                    """)
                
                # Mark as applied
                st.session_state.transformation_applied = True
                st.session_state.transformed_df = transformed_df
                
                # Show preview
                st.markdown("#### 📋 Preview de datos transformados")
                st.dataframe(transformed_df.head(10), width='stretch')
                st.info(f"Shape: {transformed_df.shape} | Target: {task}")
                
            except Exception as e:
                st.error(f"❌ Error durante transformación: {e}")
                with st.expander("Ver traceback"):
                    st.exception(e)

# Show transformation status
if transformation_type != "🔤 Original Features":
    if st.session_state.transformation_applied:
        params = st.session_state.transformation_params
        if params['type'] == 'pca':
            st.success(
                f"✅ **PCA activo:** {params['n_components']} componentes | "
                f"Varianza: {params['variance_explained']*100:.2f}%"
            )
        else:  # ICA
            st.success(
                f"✅ **ICA activo:** {params['n_components']} componentes | "
                f"Kurtosis: {params['kurtosis_mean']:.3f}"
            )
    else:
        st.warning("⚠️ Transformación configurada pero no aplicada. Haz clic en '🔄 Aplicar Transformación'.")

# Training section
st.markdown("---")
st.subheader("🚀 Train Models")

if not selected_models:
    st.error("❌ Please select at least one model from the sidebar")
else:
    # Initialize training state
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    
    # Show button or training message
    if not st.session_state.is_training:
        st.markdown("### 🎯 ¿Listo para entrenar?")
        
        if not selected_models:
            st.error("❌ Selecciona al menos un modelo en la barra lateral")
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                start_button = st.button(
                    "🚀 INICIAR ENTRENAMIENTO",
                    type="primary",
                    use_container_width=True,
                    help="Click para comenzar el entrenamiento de modelos"
                )
    else:
        start_button = False
        st.warning("⏳ **Entrenamiento en progreso...**")
        st.info("💾 Checkpointing: " + ("✅ Activo" if use_checkpointing else "❌ Desactivado"))
    
    if start_button and not st.session_state.is_training:
        st.session_state.is_training = True
        st.rerun()
    
    if st.session_state.is_training:
        try:
            st.markdown("---")
            st.info(f"""
            **🎯 Configuración del entrenamiento:**
            - Modelos: {', '.join(selected_models)}
            - Quick Mode: {'✅ Sí' if quick else '❌ No'}
            - Checkpointing: {'✅ Activo' if use_checkpointing else '❌ Desactivado'}
            - Reanudando checkpoint: {'✅ Sí - ' + selected_checkpoint_id if selected_checkpoint_id else '❌ No (nuevo)'}
            """)
            
            # Determine which dataset to use
            if transformation_type != "🔤 Original Features":
                if st.session_state.transformation_applied and st.session_state.transformed_df is not None:
                    df_for_training = st.session_state.transformed_df.copy()
                    st.info(f"ℹ️ Entrenando con **{st.session_state.transformation_params['type'].upper()} components**: {st.session_state.transformation_params['n_components']} features")
                else:
                    st.error("❌ Transformación no aplicada. Haz clic en '🔄 Aplicar Transformación' primero.")
                    st.session_state.is_training = False
                    st.stop()
            else:
                df_for_training = df.copy()
                st.info(f"ℹ️ Entrenando con **features originales**: {len(df.columns)-1} variables")
            
            # Save training dataset
            temp_dir = Path(tempfile.gettempdir())
            training_data_path = temp_dir / f"streamlit_training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_for_training.to_csv(training_data_path, index=False)
            st.success(f"✅ Dataset preparado: {len(df_for_training)} muestras, {len(df_for_training.columns)} columnas")
            
            # Save transformer if using transformation
            transformer_path = None
            if transformation_type != "🔤 Original Features" and st.session_state.transformer is not None:
                transformer_path = temp_dir / f"streamlit_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                joblib.dump(st.session_state.transformer, str(transformer_path))
                st.success(f"✅ Transformer guardado temporalmente")
            
            st.markdown("---")
            st.markdown("### 📊 Progreso del Entrenamiento")
            
            progress_container = st.container()
            
            with progress_container:
                st.info("⏳ Iniciando entrenamiento...")
                
                # ✅ LLAMADA PRINCIPAL con checkpoint_id correcto
                save_paths, experiment_results = train_models_with_progress(
                    data_path=str(training_data_path),
                    task=task,
                    quick=quick,
                    imputer_mode=imputer_mode,
                    selected_models=selected_models,
                    use_checkpointing=use_checkpointing,
                    resume_checkpoint_id=selected_checkpoint_id if use_checkpointing else None,
                )
            
            # Clean up
            try:
                if training_data_path.exists():
                    training_data_path.unlink()
            except Exception:
                pass
            
            # Update session state
            set_state("is_trained", True)
            set_state("last_train_task", task)
            set_state("last_train_models", list(save_paths.keys()))
            st.session_state.training_results = experiment_results
            
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            
            st.markdown("---")
            st.success(f"""
            ✅ **¡Entrenamiento completado exitosamente!**
            
            - ✅ {len(save_paths)} modelo(s) entrenado(s)
            - ✅ Validación cruzada estratificada completada
            - ✅ Curvas de aprendizaje generadas
            - ✅ Comparación estadística realizada
            - ✅ Modelos guardados en `models/`
            """)
            
            with st.expander("📁 Ver rutas de modelos guardados"):
                for name, path in save_paths.items():
                    st.code(path, language="text")
            
            # Save transformer
            if transformer_path is not None and st.session_state.transformer is not None:
                st.markdown("---")
                st.info("💾 **Guardando transformer y actualizando metadata...**")
                
                try:
                    models_dir = root_dir / "models"
                    permanent_transformer_path = models_dir / f"transformer_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                    joblib.dump(st.session_state.transformer, str(permanent_transformer_path))
                    
                    for model_name, model_path in save_paths.items():
                        metadata_path = Path(str(model_path).replace('.joblib', '.metadata.json'))
                        
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            metadata['transformation'] = {
                                'type': st.session_state.transformation_params['type'],
                                'n_components': st.session_state.transformation_params['n_components'],
                                'transformer_path': str(permanent_transformer_path),
                                'original_features': st.session_state.transformation_params['original_features'],
                                'params': st.session_state.transformation_params
                            }
                            
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                    
                    st.success(f"✅ Transformer guardado: {permanent_transformer_path.name}")
                except Exception as e:
                    st.error(f"❌ Error guardando transformer: {e}")
                finally:
                    try:
                        if transformer_path and Path(transformer_path).exists():
                            Path(transformer_path).unlink()
                    except Exception:
                        pass
            
            # Display learning curves
            if hasattr(st.session_state, 'learning_curve_paths') and st.session_state.learning_curve_paths:
                st.markdown("---")
                st.subheader("📈 Curvas de Aprendizaje")
                
                lc_paths = st.session_state.learning_curve_paths
                
                if len(lc_paths) > 0:
                    tabs = st.tabs([f"📊 {model}" for model in lc_paths.keys()])
                    
                    for tab, (model_name, img_path) in zip(tabs, lc_paths.items()):
                        with tab:
                            if Path(img_path).exists():
                                st.image(str(img_path), caption=f"Learning Curve: {model_name}", use_container_width=True)
            
            st.balloons()
            
            # Statistical comparison
            st.markdown("---")
            st.subheader("📊 Comparación Estadística de Modelos")
            
            stat_results = experiment_results.get('statistical_comparison', {})
            
            if stat_results and len(selected_models) > 1:
                st.info("Comparación par-a-par entre todos los modelos entrenados")
                
                for (m1, m2), comparison in stat_results.items():
                    with st.expander(f"🔬 {m1} vs {m2}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Test Usado", comparison.get('test_type', 'N/A'))
                        with col2:
                            p_val = comparison.get('p_value', 0)
                            st.metric("P-value", f"{p_val:.4f}")
                        with col3:
                            is_sig = comparison.get('significant', False)
                            sig_label = "Sí ✅" if is_sig else "No ❌"
                            st.metric("Significativo (α=0.05)", sig_label)
                        
                        st.write(f"**Interpretación:** {comparison.get('interpretation', 'N/A')}")
        
        except FileNotFoundError as e:
            st.error(f"❌ Dataset file not found: {e}")
            st.exception(e)
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {e}")
            st.exception(e)
            
            import traceback
            with st.expander("🔍 Ver detalles técnicos del error"):
                st.code(traceback.format_exc())
        finally:
            st.session_state.is_training = False

st.markdown("---")

# Display saved models
st.subheader("Saved Models")

last_task = get_state("last_train_task")
if last_task and last_task != task:
    st.info(f"ℹ️ Last training was for task: {last_task}")

display_model_list(task)

# Training notes
with st.expander("ℹ️ Training Notes"):
    st.markdown("""
    ### ⚙️ Configuración del Entrenamiento
    
    **Quick Mode:**
    - ✅ Búsqueda simplificada de hiperparámetros
    - ✅ Menos splits en CV (3×3 = 9 corridas en vez de 10×10 = 100)
    - ✅ Iteración rápida para depuración
    - ⚠️ Recomendado solo para exploración inicial
    
    **Checkpointing:**
    - ✅ Guarda progreso después de cada fold
    - ✅ Permite reanudar entrenamiento si se interrumpe
    - ✅ Gestión individual de checkpoints (eliminar selectivamente)
    - ✅ Se limpia automáticamente al completar
    - 📂 Ubicación: `Tools/dashboard/checkpoints/`
    
    **Estrategias de Imputación:**
    - **Iterative**: IterativeImputer de sklearn (MICE)
    - **KNN**: K-Nearest Neighbors imputation
    - **Simple**: Imputación básica (media/mediana/moda)
    
    ### 📋 Pipeline de Experimentación
    
    El **Pipeline Riguroso** implementa el proceso científico completo:
    
    1. **Validación Cruzada Estratificada Repetida**: ≥30 corridas para μ y σ robustos
    2. **Curvas de Aprendizaje**: Diagnostican sobreajuste/subajuste
    3. **Comparación Estadística**: Tests paramétricos/no paramétricos
    4. **Evaluación Final**: Bootstrap + Jackknife con intervalos de confianza
    """)

# PDF export
st.markdown("---")
st.subheader("📄 Exportar Reporte de Entrenamiento")

if st.session_state.get('training_results'):
    def generate_training_report():
        from pathlib import Path
        output_path = Path("reports") / "training_report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_res = st.session_state.training_results
        models_metadata = {}
        cv_results = training_res.get('cv_results', {})
        
        for model_name in cv_results.keys():
            from src.models.metadata import ModelMetadata, PerformanceMetrics, TrainingMetadata, DatasetMetadata
            
            cv_data = cv_results[model_name]
            
            perf_metrics = PerformanceMetrics(
                mean_score=cv_data['mean_score'],
                std_score=cv_data['std_score'],
                min_score=cv_data.get('min_score', 0.0),
                max_score=cv_data.get('max_score', 1.0),
                all_scores=cv_data.get('all_scores', [])
            )
            
            train_metadata = TrainingMetadata(
                training_date=datetime.now().isoformat(),
                training_duration_seconds=0.0,
                cv_strategy="RepeatedStratifiedKFold",
                n_cv_folds=cv_data.get('n_splits', 10),
                n_cv_repeats=cv_data.get('n_repeats', 10),
                total_cv_runs=cv_data.get('n_runs', 100),
                scoring_metric=cv_data.get('scoring', 'roc_auc'),
                preprocessing_config={},
                random_seed=42
            )
            
            models_metadata[model_name] = ModelMetadata(
                model_name=model_name,
                model_type=model_name,
                task="classification",
                model_file_path="",
                dataset=DatasetMetadata(
                    train_set_path="",
                    test_set_path="",
                    train_samples=0,
                    test_samples=0,
                    n_features=0,
                    target_column="",
                    class_distribution_train={},
                    class_distribution_test={},
                    feature_names=[]
                ),
                training=train_metadata,
                hyperparameters={},
                performance=perf_metrics
            )
        
        return generate_training_pdf(
            training_results=training_res,
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
