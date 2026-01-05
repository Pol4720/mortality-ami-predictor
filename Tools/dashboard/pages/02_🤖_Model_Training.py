"""Model Training page."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
import tempfile

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
from src.config import CONFIG

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

# ==================== TARGET VARIABLE SELECTION ====================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Variable Objetivo")

# Get available columns for target selection (binary/numeric columns)
potential_targets = []
for col in df.columns:
    # Check if column could be a valid target (binary or numeric with few unique values)
    if df[col].nunique() <= 10:  # Categorical or binary
        potential_targets.append(col)
    elif pd.api.types.is_numeric_dtype(df[col]):
        potential_targets.append(col)

# Determine default target from session_state or CONFIG
saved_target = st.session_state.get('target_column_name', None)
default_target = None

# Priority: 1) saved from Data Cleaning, 2) CONFIG.target_column, 3) first potential target
if saved_target and saved_target in potential_targets:
    default_target = saved_target
elif CONFIG.target_column in potential_targets:
    default_target = CONFIG.target_column
elif potential_targets:
    default_target = potential_targets[0]

# Create target selector
default_idx = potential_targets.index(default_target) if default_target in potential_targets else 0

target_col = st.sidebar.selectbox(
    "Variable a Predecir",
    potential_targets,
    index=default_idx,
    help="Selecciona la variable objetivo para el entrenamiento. Se recomienda seleccionarla primero en Data Cleaning."
)

# Save selection to session_state
st.session_state.target_column_name = target_col
st.session_state.target_column = target_col

# Determine task name for model organization (used for saving models in folders)
# If it's a known target, use the standard name; otherwise use 'custom'
if target_col == CONFIG.target_column or target_col in ['mortality', 'mortality_inhospital', 'exitus']:
    task = 'mortality'
elif target_col == CONFIG.arrhythmia_column or target_col in ['arrhythmia', 'ventricular_arrhythmia']:
    task = 'arrhythmia'
else:
    # Custom target - use a sanitized version of the column name
    task = target_col.lower().replace(' ', '_')[:20]  # Limit length for folder names

# Show target info
target_info = df[target_col].value_counts()
st.sidebar.markdown(f"**Distribución:**")
for val, count in target_info.head(5).items():
    pct = count / len(df) * 100
    st.sidebar.markdown(f"- `{val}`: {count} ({pct:.1f}%)")

if df[target_col].nunique() > 5:
    st.sidebar.caption(f"... y {df[target_col].nunique() - 5} valores más")

# Custom models section
st.sidebar.markdown("---")
st.sidebar.header("🔧 Custom Models")

use_custom_models = st.sidebar.checkbox(
    "Include Custom Models",
    value=False,
    help="Include custom models defined in Custom Models page"
)

custom_models_list = []
custom_model_classes = {}

if use_custom_models:
    import importlib.util
    import inspect
    import sys
    from src.models.custom_base import BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor
    
    # Buscar archivos .py con definiciones de modelos custom
    code_templates_dir = root_dir / "src" / "models" / "custom"
    code_templates_dir.mkdir(parents=True, exist_ok=True)
    
    available_files = sorted([f for f in code_templates_dir.glob("*.py") if f.name != "__init__.py"])
    
    if available_files:
        st.sidebar.markdown(f"**Available: {len(available_files)} file(s)**")
        
        # Extraer clases de cada archivo
        available_classes = []
        for filepath in available_files:
            try:
                # Use unique module name for each file to avoid conflicts
                module_name = f"custom_models.{filepath.stem}"
                
                # Check if module is already loaded
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module  # Register in sys.modules for pickle
                    spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, (BaseCustomClassifier, BaseCustomRegressor)):
                        if obj not in [BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor]:
                            display_name = f"{name} ({filepath.stem})"
                            available_classes.append(display_name)
                            custom_model_classes[display_name] = obj
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error loading {filepath.name}: {e}")
        
        if available_classes:
            custom_models_list = st.sidebar.multiselect(
                "Select Custom Models",
                available_classes,
                help="Select which custom models to include in training"
            )
            
            if custom_models_list:
                st.sidebar.success(f"✅ {len(custom_models_list)} custom model(s) selected")
        else:
            st.sidebar.warning("⚠️ No valid model classes found in files")
    else:
        st.sidebar.info("📭 No custom models available. Create one in Custom Models page (🔧).")

# ==================== AUTOML SECTION ====================
st.sidebar.markdown("---")
st.sidebar.header("🤖 AutoML")

# Check AutoML availability
try:
    from src.training import is_automl_available
    automl_available = is_automl_available()
except ImportError:
    automl_available = False

if automl_available:
    use_automl = st.sidebar.checkbox(
        "Enable AutoML",
        value=False,
        help="Use automated machine learning to find the best model"
    )
    
    if use_automl:
        # AutoML preset selection
        automl_preset = st.sidebar.selectbox(
            "AutoML Preset",
            ["quick", "balanced", "high_performance"],
            index=0,
            help="quick: 5 min | balanced: 1 hour | high_performance: 4 hours"
        )
        
        # Time budget override (optional)
        automl_time = st.sidebar.number_input(
            "Custom Time Budget (seconds)",
            min_value=60,
            max_value=28800,
            value={"quick": 300, "balanced": 3600, "high_performance": 14400}[automl_preset],
            help="Override the preset time budget"
        )
        
        # Backend info
        try:
            from src.automl import is_flaml_available, is_autosklearn_available
            if is_autosklearn_available():
                backend = "auto-sklearn"
            elif is_flaml_available():
                backend = "FLAML"
            else:
                backend = "Not available"
            st.sidebar.info(f"📦 Backend: **{backend}**")
        except ImportError:
            st.sidebar.info("📦 Backend: FLAML (default)")
        
        # Store in session state
        st.session_state.automl_enabled = True
        st.session_state.automl_preset = automl_preset
        st.session_state.automl_time = automl_time
        
        st.sidebar.success("✅ AutoML enabled")
        st.sidebar.markdown(f"⏱️ Time: {automl_time//60} min")
        
        # Link to full AutoML page
        st.sidebar.markdown("---")
        st.sidebar.info("💡 For advanced AutoML options, visit **🤖 AutoML** page")
    else:
        st.session_state.automl_enabled = False
else:
    st.sidebar.warning("⚠️ AutoML not available")
    st.sidebar.markdown("""
    Install FLAML to enable:
    ```
    pip install flaml[automl]
    ```
    """)
    st.session_state.automl_enabled = False

# ==================== CLASS IMBALANCE HANDLING ====================
st.sidebar.markdown("---")
st.sidebar.header("⚖️ Balanceo de Clases")

# Check imbalanced-learn availability
try:
    from src.preprocessing.imbalance import (
        detect_imbalance,
        get_recommended_strategy,
        ImbalanceStrategy,
        STRATEGY_DESCRIPTIONS,
        is_imblearn_available,
    )
    IMBALANCE_AVAILABLE = is_imblearn_available()
except ImportError:
    IMBALANCE_AVAILABLE = False

# Detect class imbalance
if df is not None and target_col in df.columns:
    y_target = df[target_col]
    try:
        is_imbalanced, imbalance_ratio, class_counts = detect_imbalance(y_target)
        
        # Show imbalance info
        if is_imbalanced:
            st.sidebar.warning(f"⚠️ **Dataset Desbalanceado**")
            st.sidebar.markdown(f"Ratio: **{imbalance_ratio:.1f}:1**")
        else:
            st.sidebar.success(f"✅ Dataset Balanceado (ratio {imbalance_ratio:.1f}:1)")
        
        # Show class distribution bar
        total = sum(class_counts.values())
        for cls, count in sorted(class_counts.items()):
            pct = count / total * 100
            st.sidebar.progress(pct / 100, text=f"Clase {cls}: {count} ({pct:.1f}%)")
        
    except Exception as e:
        is_imbalanced = True  # Assume imbalanced for safety
        imbalance_ratio = 1.0
        st.sidebar.info("ℹ️ No se pudo analizar el desbalance")
else:
    is_imbalanced = True
    imbalance_ratio = 1.0

# Imbalance strategy selector
if IMBALANCE_AVAILABLE:
    # Define available strategies with display names
    strategy_options = {
        "smote": "🔄 SMOTE (Recomendado)",
        "adasyn": "📈 ADASYN (Adaptativo)",
        "borderline_smote": "🎯 Borderline-SMOTE",
        "smote_tomek": "🔄+🧹 SMOTE + Tomek Links",
        "smote_enn": "🔄+✂️ SMOTE + ENN",
        "class_weight": "⚖️ Class Weight (sin resampleo)",
        "random_oversample": "📊 Random Oversampling",
        "none": "❌ Sin Balanceo",
    }
    
    # Get recommended strategy
    try:
        X_features = df.drop(columns=[target_col]) if target_col in df.columns else df
        recommended = get_recommended_strategy(y_target, X_features)
        recommended_key = recommended.value
    except:
        recommended_key = "smote"
    
    # Default to recommended strategy
    default_idx = list(strategy_options.keys()).index(recommended_key) if recommended_key in strategy_options else 0
    
    imbalance_strategy = st.sidebar.selectbox(
        "Estrategia de Balanceo",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=default_idx,
        help="Técnica para manejar el desbalance de clases"
    )
    
    # Show strategy description
    try:
        strategy_enum = ImbalanceStrategy(imbalance_strategy)
        desc = STRATEGY_DESCRIPTIONS.get(strategy_enum, {})
        if desc:
            with st.sidebar.expander("ℹ️ Acerca de esta estrategia"):
                st.markdown(f"**{desc.get('name', '')}**")
                st.markdown(desc.get('description', ''))
                st.markdown(f"✅ **Ventajas:** {desc.get('pros', '')}")
                st.markdown(f"❌ **Desventajas:** {desc.get('cons', '')}")
                st.markdown(f"💡 **Recomendado para:** {desc.get('recommended_for', '')}")
    except:
        pass
    
    # Advanced options for SMOTE variants
    if imbalance_strategy in ['smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn']:
        with st.sidebar.expander("⚙️ Opciones Avanzadas"):
            imbalance_k_neighbors = st.slider(
                "K-Neighbors (SMOTE)",
                min_value=1,
                max_value=15,
                value=5,
                help="Número de vecinos para generación de datos sintéticos"
            )
            
            imbalance_sampling_strategy = st.selectbox(
                "Sampling Strategy",
                ["auto", "minority", "not majority", "all"],
                index=0,
                help="'auto' balancea las clases automáticamente"
            )
    else:
        imbalance_k_neighbors = 5
        imbalance_sampling_strategy = "auto"
    
    # Store in session state
    st.session_state.imbalance_strategy = imbalance_strategy
    st.session_state.imbalance_k_neighbors = imbalance_k_neighbors
    st.session_state.imbalance_sampling_strategy = imbalance_sampling_strategy
    
else:
    st.sidebar.warning("⚠️ imbalanced-learn no instalado")
    st.sidebar.markdown("""
    Para habilitar SMOTE/ADASYN:
    ```
    pip install imbalanced-learn
    ```
    """)
    imbalance_strategy = "none"
    st.session_state.imbalance_strategy = "none"

# Training settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Training Configuration")



quick, imputer_mode, selected_models = sidebar_training_controls()

# Combine standard models with custom models
all_selected_models = list(selected_models) if selected_models else []
if custom_models_list:
    all_selected_models.extend(custom_models_list)

# Main content
st.subheader("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.metric("Task", task.capitalize())
    st.metric("Imputation", imputer_mode.capitalize())

with col2:
    st.metric("Quick Mode", "Enabled" if quick else "Disabled")
    st.metric("Models Selected", len(all_selected_models))

# Display selected models
if all_selected_models:
    if selected_models:
        st.info(f"📦 Standard models: {', '.join(selected_models)}")
    if custom_models_list:
        st.success(f"🔧 Custom models: {', '.join(custom_models_list)}")
else:
    st.warning("⚠️ No models selected for training")

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
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
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
                    st.error("❌ No hay datos válidos después de eliminar NaNs. Aplica imputación primero.")
                    st.stop()
                
                if transformation_type == "📊 PCA Components":
                    # Apply PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    # Standardize if requested
                    if standardize:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(df_for_transform)
                    else:
                        data_scaled = df_for_transform.values
                        scaler = None
                    
                    # Fit PCA
                    if pca_mode == "Varianza":
                        pca = PCA(n_components=variance_threshold, random_state=42)
                    else:
                        pca = PCA(n_components=n_components, random_state=42)
                    
                    components = pca.fit_transform(data_scaled)
                    
                    # Create DataFrame
                    component_names = [f'PC{i+1}' for i in range(pca.n_components_)]
                    transformed_df = pd.DataFrame(
                        components,
                        columns=component_names,
                        index=df_for_transform.index
                    )
                    
                    # Add target back
                    transformed_df[target_col] = df.loc[transformed_df.index, target_col]
                    
                    # Store in session state
                    st.session_state.transformer = {'pca': pca, 'scaler': scaler}
                    st.session_state.transformed_df = transformed_df
                    st.session_state.transformation_applied = True
                    st.session_state.transformation_params = {
                        'type': 'pca',
                        'n_components': pca.n_components_,
                        'variance_explained': sum(pca.explained_variance_ratio_),
                        'standardize': standardize,
                        'feature_names': numeric_cols
                    }
                    
                    st.success(
                        f"✅ PCA aplicado exitosamente: {pca.n_components_} componentes | "
                        f"Varianza explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%"
                    )
                
                else:  # ICA
                    # Apply ICA
                    # Convert boolean whiten to string for newer sklearn versions
                    whiten_param = 'unit-variance' if whiten else False
                    
                    ica = ICATransformer(
                        n_components=n_components,
                        algorithm=ica_algorithm,
                        fun=ica_fun,
                        whiten=whiten_param,
                        max_iter=500,
                        random_state=42
                    )
                    
                    ica.fit(df_for_transform)
                    transformed_df = ica.transform(df_for_transform)
                    
                    # Add target back
                    transformed_df[target_col] = df.loc[transformed_df.index, target_col]
                    
                    # Store in session state
                    st.session_state.transformer = ica
                    st.session_state.transformed_df = transformed_df
                    
                    st.session_state.transformation_params = {
                        'type': 'ica',
                        'n_components': n_components,
                        'algorithm': ica_algorithm,
                        'fun': ica_fun,
                        'whiten': whiten,
                        'kurtosis_mean': float(np.mean(np.abs(ica.result_.component_kurtosis))),
                        'feature_names': numeric_cols
                    }
                    st.session_state.transformation_applied = True
                    
                    st.success(
                        f"✅ ICA aplicado exitosamente: {n_components} componentes independientes | "
                        f"Kurtosis promedio: {np.mean(np.abs(ica.result_.component_kurtosis)):.3f}"
                    )
                
                # Show preview
                st.markdown("#### 📋 Preview de datos transformados")
                st.dataframe(transformed_df.head(10), width='stretch')
                st.info(f"Shape: {transformed_df.shape} | Target column: {target_col}")
                
            except Exception as e:
                st.error(f"❌ Error durante transformación: {e}")
                import traceback
                with st.expander("Ver traceback"):
                    st.code(traceback.format_exc())

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
st.subheader("Train Models")

# Show AutoML status if enabled
if st.session_state.get('automl_enabled', False):
    st.info(f"""
    🤖 **AutoML Enabled**
    - Preset: {st.session_state.get('automl_preset', 'balanced')}
    - Time Budget: {st.session_state.get('automl_time', 3600) // 60} minutes
    - AutoML will search for the best model automatically
    """)

if not all_selected_models and not st.session_state.get('automl_enabled', False):
    st.error("❌ Please select at least one model from the sidebar or enable AutoML")
else:
    # Initialize training state
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    
    # Show button or training message
    if not st.session_state.is_training:
        # Different button text based on AutoML status
        if st.session_state.get('automl_enabled', False):
            button_text = "🤖 Start AutoML Training"
        else:
            button_text = "🚀 Start Training"
        start_button = st.button(button_text, type="primary", use_container_width=True)
    else:
        st.info("⏳ **Training in progress, please wait...**")
    
    if not st.session_state.is_training and 'start_button' in locals() and start_button:
        # Set training flag
        st.session_state.is_training = True
        
        try:
            # Determine which dataset to use based on transformation selection
            if transformation_type != "🔤 Original Features":
                if st.session_state.transformation_applied and st.session_state.transformed_df is not None:
                    df_for_training = st.session_state.transformed_df.copy()
                    params = st.session_state.transformation_params
                    transform_type = "PCA" if params['type'] == 'pca' else "ICA"
                    st.info(
                        f"ℹ️ Entrenando con **{transform_type}**: {params['n_components']} componentes "
                        f"(transformación aplicada a {len(params['feature_names'])} variables originales)"
                    )
                else:
                    st.error(
                        f"❌ Has seleccionado transformación {transformation_type} pero no la has aplicado. "
                        f"Haz clic en '🔄 Aplicar Transformación' primero."
                    )
                    st.session_state.is_training = False
                    st.stop()
            else:
                df_for_training = df.copy()
                st.info(f"ℹ️ Entrenando con **features originales**: {len(df.columns)-1} variables")
            
            # ==================== AUTOML TRAINING ====================
            if st.session_state.get('automl_enabled', False):
                st.markdown("### 🤖 AutoML Training")
                
                # Prepare data
                X = df_for_training.drop(columns=[target_col])
                y = df_for_training[target_col]
                
                # Progress display
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def automl_progress_callback(msg: str, progress: float):
                    progress_bar.progress(progress)
                    status_text.markdown(f"**{msg}**")
                
                try:
                    from src.training import run_automl_experiment_pipeline
                    
                    # Run AutoML
                    automl_results = run_automl_experiment_pipeline(
                        X=X,
                        y=y,
                        preset=st.session_state.get('automl_preset', 'quick'),
                        time_budget=st.session_state.get('automl_time', 300),
                        metric="roc_auc",
                        include_suggestions=True,
                        compare_with_manual=len(all_selected_models) > 0,
                        progress_callback=automl_progress_callback,
                    )
                    
                    # Display results
                    st.success(f"""
                    ✅ **AutoML Completed!**
                    
                    - Best Model: {automl_results.get('best_estimator', 'Unknown')}
                    - Best Score (AUC): {automl_results.get('best_score', 0):.4f}
                    - Training Time: {automl_results.get('training_duration', 0) / 60:.1f} minutes
                    - Backend: {automl_results.get('backend', 'FLAML')}
                    """)
                    
                    # Show leaderboard
                    if 'leaderboard' in automl_results and not automl_results['leaderboard'].empty:
                        st.markdown("### 📊 Model Leaderboard")
                        st.dataframe(automl_results['leaderboard'].head(10), use_container_width=True)
                    
                    # Show suggestions
                    if 'suggestions' in automl_results and automl_results['suggestions']:
                        with st.expander("💡 Dataset Suggestions", expanded=False):
                            for s in automl_results['suggestions'][:5]:
                                priority_icon = "🔴" if s['priority'] == 'high' else "🟡" if s['priority'] == 'medium' else "🟢"
                                st.markdown(f"**{priority_icon} {s['title']}**")
                                st.markdown(f"  {s['description']}")
                                if s.get('module_link'):
                                    st.markdown(f"  → Module: `{s['module_link']}`")
                    
                    # Store results
                    st.session_state.automl_results = automl_results
                    st.session_state.training_results = automl_results
                    set_state("is_trained", True)
                    
                    # Show model path
                    if 'final_model_path' in automl_results:
                        st.info(f"📁 Model saved to: `{automl_results['final_model_path']}`")
                    
                    st.balloons()
                    
                except ImportError as e:
                    st.error(f"❌ AutoML not available: {e}")
                    st.info("Install FLAML: `pip install flaml[automl]`")
                except Exception as e:
                    st.error(f"❌ AutoML Error: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                
                st.session_state.is_training = False
                st.stop()  # Stop here if AutoML was used
            
            # ==================== STANDARD TRAINING ====================
            # Save the DataFrame that will actually be used for training
            # This ensures metadata will reflect the correct features
            temp_dir = Path(tempfile.gettempdir())
            training_data_path = temp_dir / f"streamlit_training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_for_training.to_csv(training_data_path, index=False)
            st.success(f"✅ Dataset para entrenamiento guardado: {len(df_for_training.columns)} columnas (incluyendo target)")
            
            # Save transformer if using transformation
            transformer_path = None
            if transformation_type != "🔤 Original Features" and st.session_state.transformer is not None:
                transformer_path = temp_dir / f"streamlit_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                joblib.dump(st.session_state.transformer, str(transformer_path))
                st.success(f"✅ Transformer guardado temporalmente: {transformer_path.name}")
            
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
                
                # Get imbalance settings from session state
                imb_strategy = st.session_state.get('imbalance_strategy', 'smote')
                imb_k_neighbors = st.session_state.get('imbalance_k_neighbors', 5)
                imb_sampling = st.session_state.get('imbalance_sampling_strategy', 'auto')
                
                with redirect_stdout(output_buffer):
                    save_paths, experiment_results = train_models_with_progress(
                        data_path=str(training_data_path),  # Use the actual training dataset
                        task=task,
                        quick=quick,
                        imputer_mode=imputer_mode,
                        selected_models=all_selected_models,
                        custom_model_classes=custom_model_classes if use_custom_models else {},
                        target_column=target_col,  # Pass explicit target column
                        imbalance_strategy=imb_strategy,
                        imbalance_k_neighbors=imb_k_neighbors,
                        imbalance_sampling_strategy=imb_sampling,
                    )
                
                # Get the output
                output = output_buffer.getvalue()
                
                # Display in expander
                with st.expander("📋 Ver detalles completos del entrenamiento", expanded=False):
                    st.code(output, language="text")
            
            # Clean up temporary training dataset
            try:
                if training_data_path.exists():
                    training_data_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors
            
            # Update session state
            set_state("is_trained", True)
            set_state("last_train_task", task)
            set_state("last_train_models", list(save_paths.keys()))
            
            # Store training results for PDF report
            st.session_state.training_results = experiment_results
            
            # Store trained models references
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            
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
            
            # Save transformer alongside models and update metadata
            if transformer_path is not None and st.session_state.transformer is not None:
                st.markdown("---")
                st.info("💾 **Guardando transformer y actualizando metadata de modelos...**")
                
                try:
                    from pathlib import Path as PathlibPath
                    import json
                    
                    # Save transformer permanently for each model
                    for model_name, model_path in save_paths.items():
                        model_dir = PathlibPath(model_path).parent
                        transformer_save_path = model_dir / f"{model_name}_transformer.joblib"
                        
                        # Copy transformer to model directory
                        joblib.dump(st.session_state.transformer, str(transformer_save_path))
                        
                        # Update model metadata to include transformation info
                        metadata_path = model_dir / f"{model_name}_metadata.json"
                        
                        if metadata_path.exists():
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata_dict = json.load(f)
                            
                            # Add transformation information
                            metadata_dict['transformation'] = {
                                'type': st.session_state.transformation_params['type'],
                                'n_components': st.session_state.transformation_params['n_components'],
                                'transformer_path': str(transformer_save_path),
                                'original_features': st.session_state.transformation_params['feature_names'],
                                'params': st.session_state.transformation_params
                            }
                            
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata_dict, f, indent=2)
                        
                        st.success(f"✅ Transformer guardado para {model_name}: `{transformer_save_path.name}`")
                    
                    st.success(f"""
                    ✅ **Transformers guardados exitosamente**
                    
                    - Tipo: {st.session_state.transformation_params['type'].upper()}
                    - Componentes: {st.session_state.transformation_params['n_components']}
                    - Variables originales transformadas: {len(st.session_state.transformation_params['feature_names'])}
                    - Metadata actualizado para todos los modelos
                    
                    **Los modelos aplicarán automáticamente esta transformación durante la predicción.**
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error guardando transformer: {e}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
                
                finally:
                    # Clean up temporary transformer
                    try:
                        if transformer_path.exists():
                            transformer_path.unlink()
                    except Exception:
                        pass
            
            # Display learning curves if available (INTERACTIVAS con Plotly)
            if hasattr(st.session_state, 'learning_curve_results') and st.session_state.learning_curve_results:
                st.markdown("---")
                st.subheader("📈 Curvas de Aprendizaje")
                
                with st.expander("ℹ️ ¿Cómo interpretar las curvas de aprendizaje?", expanded=False):
                    st.markdown("""
                    **Curvas de aprendizaje** muestran el rendimiento del modelo vs tamaño de entrenamiento:
                    
                    | Patrón | Diagnóstico | Solución |
                    |--------|-------------|----------|
                    | Train alto, Val bajo, Gap grande | **Overfitting** | Más datos, regularización, modelo más simple |
                    | Train bajo, Val bajo, Gap pequeño | **Underfitting** | Modelo más complejo, más features |
                    | Train≈Val, ambos altos, Gap pequeño | **Buen ajuste** | ✅ Modelo adecuado |
                    | Curvas no convergen | **No convergido** | Más datos o epochs |
                    """)
                
                lc_results = st.session_state.learning_curve_results
                
                # Import diagnosis function
                from src.training.learning_curves import plot_learning_curve, diagnose_learning_curve
                
                # Create tabs for each model
                if len(lc_results) > 0:
                    tabs = st.tabs([f"📊 {model}" for model in lc_results.keys()])
                    
                    for tab, (model_name, lc_res) in zip(tabs, lc_results.items()):
                        with tab:
                            # Generate interactive Plotly figure
                            try:
                                fig = plot_learning_curve(lc_res, title=f"Learning Curve: {model_name}")
                                st.plotly_chart(fig, use_container_width=True, key=f"lc_{model_name}")
                            except Exception as e:
                                st.warning(f"No se pudo generar gráfico interactivo: {e}")
                                # Fallback to static image
                                lc_paths = st.session_state.get('learning_curve_paths', {})
                                if model_name in lc_paths and Path(lc_paths[model_name]).exists():
                                    st.image(lc_paths[model_name], use_container_width=True)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            
                            final_train = lc_res.train_scores_mean[-1]
                            final_val = lc_res.val_scores_mean[-1]
                            gap = abs(final_train - final_val)
                            
                            with col1:
                                st.metric("Score Final (Train)", f"{final_train:.4f}")
                            
                            with col2:
                                st.metric("Score Final (Val)", f"{final_val:.4f}")
                            
                            with col3:
                                delta_color = "normal" if gap < 0.05 else ("off" if gap < 0.10 else "inverse")
                                st.metric("Gap Train-Val", f"{gap:.4f}", delta=None)
                            
                            # Diagnosis de underfitting/overfitting
                            st.markdown("##### 🔍 Diagnóstico Automático")
                            diagnosis = diagnose_learning_curve(lc_res)
                            
                            # Show issues
                            if diagnosis['issues']:
                                for issue in diagnosis['issues']:
                                    if 'Good fit' in issue:
                                        st.success(f"✅ {issue}")
                                    elif 'underfitting' in issue.lower() or 'High bias' in issue:
                                        st.error(f"🔴 **{issue}**: El modelo es demasiado simple para capturar los patrones")
                                    elif 'overfitting' in issue.lower() or 'High variance' in issue:
                                        st.warning(f"⚠️ **{issue}**: El modelo memoriza datos en lugar de generalizar")
                                    elif 'not converged' in issue.lower():
                                        st.info(f"ℹ️ **{issue}**: El modelo podría mejorar con más datos")
                                    else:
                                        st.info(f"ℹ️ {issue}")
                            
                            # Show recommendations
                            if diagnosis['recommendations'] and 'performing well' not in ' '.join(diagnosis['recommendations']):
                                with st.expander("💡 Recomendaciones"):
                                    for rec in diagnosis['recommendations']:
                                        st.markdown(f"- {rec}")
                            
                            # Additional stats
                            with st.expander("📊 Estadísticas detalladas"):
                                st.markdown(f"""
                                - **Train mejorando**: {'Sí ✅' if diagnosis['train_improving'] else 'No ❌'}
                                - **Validación mejorando**: {'Sí ✅' if diagnosis['val_improving'] else 'No ❌'}
                                - **Convergido**: {'Sí ✅' if diagnosis['converged'] else 'No (posiblemente necesita más datos)'}
                                - **Score inicial (Train)**: {lc_res.train_scores_mean[0]:.4f}
                                - **Score inicial (Val)**: {lc_res.val_scores_mean[0]:.4f}
                                """)
            
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
                    # Calculate mean difference from model means
                    mean_diff = res.model1_mean - res.model2_mean
                    comparison_data.append({
                        "Modelo 1": m1,
                        "Modelo 2": m2,
                        "Test Usado": res.test_used,
                        "p-value": f"{res.p_value:.4f}",
                        "Significativo (p<0.05)": "✅ SÍ" if res.significant else "❌ NO",
                        "Diferencia de medias": f"{mean_diff:.4f}",
                        "Effect Size (Cohen's d)": f"{res.effect_size:.4f} ({res.effect_size_interpretation})",
                        "Normalidad M1": "✓" if res.model1_is_normal else "✗",
                        "Normalidad M2": "✓" if res.model2_is_normal else "✗"
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

# Display learning curves from previous training if available (INTERACTIVO)
if not get_state("is_trained") and hasattr(st.session_state, 'learning_curve_results'):
    lc_results = st.session_state.learning_curve_results
    if lc_results:
        st.subheader("📈 Curvas de Aprendizaje (del último entrenamiento)")
        
        from src.training.learning_curves import plot_learning_curve, diagnose_learning_curve
        
        tabs = st.tabs([f"📊 {model}" for model in lc_results.keys()])
        
        for tab, (model_name, lc_res) in zip(tabs, lc_results.items()):
            with tab:
                try:
                    fig = plot_learning_curve(lc_res, title=f"Learning Curve: {model_name}")
                    st.plotly_chart(fig, use_container_width=True, key=f"lc_prev_{model_name}")
                    
                    # Quick diagnosis
                    diagnosis = diagnose_learning_curve(lc_res)
                    for issue in diagnosis['issues']:
                        if 'Good fit' in issue:
                            st.success(f"✅ {issue}")
                        elif 'underfitting' in issue.lower():
                            st.error(f"🔴 {issue}")
                        elif 'overfitting' in issue.lower():
                            st.warning(f"⚠️ {issue}")
                except Exception:
                    # Fallback to static
                    lc_paths = st.session_state.get('learning_curve_paths', {})
                    if model_name in lc_paths and Path(lc_paths[model_name]).exists():
                        st.image(lc_paths[model_name], use_container_width=True)
        
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
        
        # Get training results
        training_res = st.session_state.training_results
        
        # Extract models metadata from cv_results
        models_metadata = {}
        cv_results = training_res.get('cv_results', {})
        
        for model_name in cv_results.keys():
            # Create basic metadata from CV results
            from src.models.metadata import ModelMetadata, PerformanceMetrics, TrainingMetadata, DatasetMetadata
            
            cv_data = cv_results[model_name]
            
            # Performance metrics
            perf_metrics = PerformanceMetrics(
                mean_score=cv_data['mean_score'],
                std_score=cv_data['std_score'],
                min_score=cv_data.get('min_score', 0.0),
                max_score=cv_data.get('max_score', 1.0),
                all_scores=cv_data.get('all_scores', [])
            )
            
            # Basic training metadata
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
            
            # Create metadata
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
