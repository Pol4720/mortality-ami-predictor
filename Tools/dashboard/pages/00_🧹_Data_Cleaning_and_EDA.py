"""Página de Streamlit para Preparación de Datos y Análisis Exploratorio.

Esta página proporciona una interfaz completa para:
- Limpieza de datos con opciones configurables
- Análisis exploratorio univariado, bivariado y multivariado
- Visualizaciones interactivas
- Generación de reportes de calidad
- Exportación de datos limpios y análisis
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import dashboard config for absolute paths
from dashboard.app.config import CLEANED_DATASETS_DIR, PLOTS_EDA_DIR

# Import src modules
from src.config import CONFIG
from src.cleaning import CleaningConfig, DataCleaner, quick_clean
from src.eda import EDAAnalyzer, quick_eda
from src.eda import generate_univariate_pdf, generate_bivariate_pdf, generate_multivariate_pdf
from src.reporting import pdf_export_section

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Preparación de Datos y EDA - AMI Mortality Predictor",
    layout="wide",
    page_icon="🧹"
)


def init_session_state():
    """Inicializa el estado de la sesión."""
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'cleaning_config' not in st.session_state:
        st.session_state.cleaning_config = CleaningConfig()


def load_data_page():
    """Sección de carga de datos."""
    st.header("📂 Carga de Datos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Opción 1: Cargar desde ruta
        data_path = st.text_input(
            "Ruta del dataset",
            value=CONFIG.dataset_path,
            help="Ruta al archivo CSV o Excel del dataset"
        )
        
        if st.button("Cargar dataset desde ruta", type="primary"):
            try:
                if data_path.endswith('.csv'):
                    # Try multiple encodings for CSV files
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                    df = None
                    last_error = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(data_path, encoding=encoding)
                            break
                        except (UnicodeDecodeError, LookupError) as e:
                            last_error = e
                            continue
                    
                    if df is None:
                        raise RuntimeError(
                            f"No se pudo decodificar el CSV con ninguna codificación. Error: {last_error}"
                        )
                elif data_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_path)
                else:
                    st.error("Formato no soportado. Use CSV o Excel.")
                    return
                
                st.session_state.raw_data = df
                st.session_state.data_path = data_path  # Guardar la ruta del dataset
                st.success(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al cargar datos: {e}")
    
    with col2:
        # Opción 2: Subir archivo
        uploaded = st.file_uploader(
            "O sube un archivo",
            type=['csv', 'xlsx', 'xls'],
            help="Arrastra y suelta un archivo CSV o Excel"
        )
        
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    # Try multiple encodings for CSV files
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                    df = None
                    last_error = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(uploaded, encoding=encoding)
                            break
                        except (UnicodeDecodeError, LookupError) as e:
                            last_error = e
                            continue
                    
                    if df is None:
                        raise RuntimeError(
                            f"No se pudo decodificar el CSV. Error: {last_error}"
                        )
                else:
                    df = pd.read_excel(uploaded)
                
                st.session_state.raw_data = df
                
                # Guardar archivo subido en temporal para uso posterior
                import tempfile
                temp_dir = Path(tempfile.gettempdir())
                temp_path = temp_dir / f"uploaded_{uploaded.name}"
                if uploaded.name.endswith('.csv'):
                    df.to_csv(temp_path, index=False)
                else:
                    df.to_excel(temp_path, index=False)
                st.session_state.data_path = str(temp_path)
                
                st.success(f"✅ Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al leer archivo: {e}")
    
    # Opción 3: Cargar dataset limpio existente
    st.markdown("---")
    st.subheader("Cargar dataset limpio existente")
    
    # Use absolute path from dashboard config
    cleaned_dir = CLEANED_DATASETS_DIR
    if cleaned_dir.exists():
        cleaned_files = sorted(cleaned_dir.glob("cleaned_dataset_*.csv"), 
                              key=lambda p: p.stat().st_mtime, reverse=True)
        
        if cleaned_files:
            file_options = {f.name: str(f) for f in cleaned_files[:10]}
            selected_file = st.selectbox("Datasets limpios disponibles", list(file_options.keys()))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cargar dataset limpio"):
                    try:
                        # Try multiple encodings for CSV files
                        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                        df = None
                        last_error = None
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_options[selected_file], encoding=encoding)
                                break
                            except (UnicodeDecodeError, LookupError) as e:
                                last_error = e
                                continue
                        
                        if df is None:
                            raise RuntimeError(
                                f"No se pudo decodificar el CSV. Error: {last_error}"
                            )
                        
                        st.session_state.cleaned_data = df
                        st.session_state.data_path = str(file_options[selected_file])
                        st.success(f"✅ Dataset limpio cargado: {df.shape}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with col2:
                if st.button("Cargar metadatos asociados"):
                    try:
                        metadata_file = Path(CONFIG.metadata_path)
                        if metadata_file.exists():
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            st.success("✅ Metadatos cargados")
                            with st.expander("Ver metadatos"):
                                st.json(metadata)
                        else:
                            st.warning("No se encontró archivo de metadatos")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # Vista previa de datos
    if st.session_state.raw_data is not None:
        st.markdown("---")
        st.subheader("Vista previa de datos crudos")
        
        df = st.session_state.raw_data
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", f"{df.shape[0]:,}")
        col2.metric("Columnas", f"{df.shape[1]:,}")
        col3.metric("Valores faltantes", f"{df.isna().sum().sum():,}")
        col4.metric("% Completitud", f"{(1 - df.isna().mean().mean()) * 100:.1f}%")
        
        st.dataframe(df.head(100), width='stretch', height=300)


def variable_selection_page():
    """Sección para descartar variables irrelevantes antes de la limpieza."""
    st.header("🎯 Selección de Variables")
    
    if st.session_state.raw_data is None:
        st.warning("⚠️ Primero carga un dataset en la pestaña 'Carga de Datos'")
        return
    
    df = st.session_state.raw_data.copy()
    
    # Inicializar variables en session_state
    if 'variables_to_keep' not in st.session_state:
        st.session_state.variables_to_keep = set(df.columns.tolist())
    
    if 'variables_to_drop' not in st.session_state:
        st.session_state.variables_to_drop = set()
    
    # Información general
    st.info("""
    **👉 Instrucciones:** Selecciona las variables que deseas **mantener** para el análisis y limpieza. 
    Las variables no seleccionadas serán descartadas antes de iniciar la limpieza de datos.
    """)
    
    # Estadísticas de variables
    col1, col2, col3, col4 = st.columns(4)
    total_vars = len(df.columns)
    vars_to_keep = len(st.session_state.variables_to_keep)
    vars_to_drop = len(st.session_state.variables_to_drop)
    
    col1.metric("📊 Variables Totales", total_vars)
    col2.metric("✅ Variables Seleccionadas", vars_to_keep, delta=None)
    col3.metric("🗑️ Variables a Descartar", vars_to_drop, delta=None, delta_color="inverse")
    col4.metric("📈 % Seleccionadas", f"{(vars_to_keep/total_vars*100):.1f}%")
    
    st.markdown("---")
    
    # Tabs para diferentes métodos de selección
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Selección Visual", 
        "🔍 Búsqueda y Filtrado",
        "📊 Análisis de Calidad",
        "💾 Aplicar Cambios"
    ])
    
    # Tab 1: Selección visual con multiselect mejorado
    with tab1:
        st.subheader("Selección Visual de Variables")
        
        # Mostrar información de las variables
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Selecciona las variables a mantener:**")
            
            # Multiselect con todas las variables
            selected_vars = st.multiselect(
                "Variables disponibles",
                options=sorted(df.columns.tolist()),
                default=sorted(list(st.session_state.variables_to_keep)),
                help="Selecciona múltiples variables usando Ctrl/Cmd + Click",
                key="multiselect_vars"
            )
            
            # Actualizar selección
            if st.button("🔄 Actualizar Selección", key="update_visual"):
                st.session_state.variables_to_keep = set(selected_vars)
                st.session_state.variables_to_drop = set(df.columns) - set(selected_vars)
                st.success(f"✅ Selección actualizada: {len(selected_vars)} variables seleccionadas")
                st.rerun()
        
        with col2:
            st.markdown("**Acciones rápidas:**")
            
            if st.button("✅ Seleccionar todas", key="select_all", use_container_width=True):
                st.session_state.variables_to_keep = set(df.columns.tolist())
                st.session_state.variables_to_drop = set()
                st.rerun()
            
            if st.button("❌ Deseleccionar todas", key="deselect_all", use_container_width=True):
                st.session_state.variables_to_keep = set()
                st.session_state.variables_to_drop = set(df.columns.tolist())
                st.rerun()
            
            if st.button("🔄 Invertir selección", key="invert_selection", use_container_width=True):
                old_keep = st.session_state.variables_to_keep.copy()
                st.session_state.variables_to_keep = st.session_state.variables_to_drop.copy()
                st.session_state.variables_to_drop = old_keep
                st.rerun()
    
    # Tab 2: Búsqueda y filtrado
    with tab2:
        st.subheader("Búsqueda y Filtrado Avanzado")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Búsqueda por nombre
            search_term = st.text_input(
                "🔍 Buscar variables por nombre",
                placeholder="Ejemplo: edad, presion, colesterol...",
                help="Busca variables que contengan el texto ingresado"
            )
            
            if search_term:
                matching_vars = [col for col in df.columns if search_term.lower() in col.lower()]
                st.info(f"📊 Se encontraron {len(matching_vars)} variables que coinciden")
                
                if matching_vars:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("✅ Seleccionar coincidencias", key="select_matching"):
                            st.session_state.variables_to_keep.update(matching_vars)
                            st.session_state.variables_to_drop -= set(matching_vars)
                            st.success(f"✅ {len(matching_vars)} variables añadidas a la selección")
                            st.rerun()
                    
                    with col_b:
                        if st.button("❌ Descartar coincidencias", key="drop_matching"):
                            st.session_state.variables_to_drop.update(matching_vars)
                            st.session_state.variables_to_keep -= set(matching_vars)
                            st.warning(f"🗑️ {len(matching_vars)} variables marcadas para descarte")
                            st.rerun()
                    
                    # Mostrar variables encontradas
                    with st.expander("Ver variables encontradas", expanded=True):
                        for var in matching_vars:
                            status = "✅" if var in st.session_state.variables_to_keep else "❌"
                            st.text(f"{status} {var}")
        
        with col2:
            # Filtro por tipo de dato
            st.markdown("**Filtrar por tipo:**")
            
            var_types = st.multiselect(
                "Tipo de dato",
                ["Numérico", "Categórico", "Datetime", "Booleano"],
                default=[],
                key="type_filter"
            )
            
            if var_types:
                filtered_vars = []
                
                if "Numérico" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=[np.number]).columns.tolist())
                if "Categórico" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=['object', 'category']).columns.tolist())
                if "Datetime" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=['datetime64']).columns.tolist())
                if "Booleano" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=['bool']).columns.tolist())
                
                filtered_vars = list(set(filtered_vars))
                st.info(f"📊 {len(filtered_vars)} variables del tipo seleccionado")
                
                if st.button("✅ Seleccionar por tipo", key="select_by_type", use_container_width=True):
                    st.session_state.variables_to_keep.update(filtered_vars)
                    st.session_state.variables_to_drop -= set(filtered_vars)
                    st.rerun()
                
                if st.button("❌ Descartar por tipo", key="drop_by_type", use_container_width=True):
                    st.session_state.variables_to_drop.update(filtered_vars)
                    st.session_state.variables_to_keep -= set(filtered_vars)
                    st.rerun()
    
    # Tab 3: Análisis de calidad
    with tab3:
        st.subheader("Análisis de Calidad de Variables")
        
        # Calcular métricas de calidad
        quality_metrics = []
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            unique_count = df[col].nunique()
            unique_pct = unique_count / len(df) * 100
            dtype = str(df[col].dtype)
            is_selected = col in st.session_state.variables_to_keep
            
            # Para numéricas, calcular varianza
            if pd.api.types.is_numeric_dtype(df[col]):
                variance = df[col].var()
                is_constant = variance == 0 or unique_count == 1
            else:
                variance = None
                is_constant = unique_count == 1
            
            quality_metrics.append({
                'Variable': col,
                'Tipo': dtype,
                'Valores Únicos': unique_count,
                '% Únicos': f"{unique_pct:.1f}",
                '% Faltantes': f"{missing_pct:.1f}",
                'Constante': '⚠️' if is_constant else '✓',
                'Estado': '✅ Seleccionada' if is_selected else '❌ Descartada'
            })
        
        quality_df = pd.DataFrame(quality_metrics)
        
        # Filtros de calidad
        st.markdown("**Filtros de calidad automática:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_missing = st.slider(
                "% Máximo de faltantes",
                0, 100, 95,
                help="Descartar variables con más de este % de valores faltantes"
            )
        
        with col2:
            include_constants = st.checkbox(
                "Mantener constantes",
                value=False,
                help="Si está desactivado, descarta variables con un único valor"
            )
        
        with col3:
            min_unique_pct = st.slider(
                "% Mínimo de valores únicos",
                0.0, 100.0, 0.0,
                help="Descartar variables con menos de este % de valores únicos"
            )
        
        if st.button("🎯 Aplicar Filtros de Calidad", key="apply_quality_filters", type="primary"):
            vars_filtered = []
            
            for _, row in quality_df.iterrows():
                var_name = row['Variable']
                missing_pct = float(row['% Faltantes'])
                unique_pct = float(row['% Únicos'])
                is_constant = row['Constante'] == '⚠️'
                
                # Aplicar filtros
                keep_var = True
                
                if missing_pct > max_missing:
                    keep_var = False
                
                if is_constant and not include_constants:
                    keep_var = False
                
                if unique_pct < min_unique_pct:
                    keep_var = False
                
                if keep_var:
                    vars_filtered.append(var_name)
            
            st.session_state.variables_to_keep = set(vars_filtered)
            st.session_state.variables_to_drop = set(df.columns) - set(vars_filtered)
            
            st.success(f"✅ Filtros aplicados: {len(vars_filtered)} variables seleccionadas")
            st.rerun()
        
        # Mostrar tabla de calidad
        st.markdown("---")
        st.markdown("**Tabla de Calidad de Variables:**")
        
        # Añadir filtro de vista
        view_filter = st.radio(
            "Mostrar:",
            ["Todas", "Solo Seleccionadas", "Solo Descartadas"],
            horizontal=True,
            key="quality_view_filter"
        )
        
        if view_filter == "Solo Seleccionadas":
            display_df = quality_df[quality_df['Estado'] == '✅ Seleccionada']
        elif view_filter == "Solo Descartadas":
            display_df = quality_df[quality_df['Estado'] == '❌ Descartada']
        else:
            display_df = quality_df
        
        # Mostrar con colores
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Descargar reporte
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Reporte de Calidad",
            data=csv,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Tab 4: Aplicar cambios
    with tab4:
        st.subheader("Aplicar Cambios y Continuar")
        
        # Resumen de cambios
        st.markdown("### 📋 Resumen de Cambios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Variables Seleccionadas:**")
            if st.session_state.variables_to_keep:
                for var in sorted(st.session_state.variables_to_keep):
                    st.text(f"✓ {var}")
            else:
                st.warning("⚠️ No hay variables seleccionadas")
        
        with col2:
            st.markdown("**❌ Variables a Descartar:**")
            if st.session_state.variables_to_drop:
                for var in sorted(st.session_state.variables_to_drop):
                    st.text(f"✗ {var}")
            else:
                st.info("ℹ️ No se descartarán variables")
        
        st.markdown("---")
        
        # Confirmación y aplicación
        if st.session_state.variables_to_drop:
            st.warning(f"⚠️ Se descartarán **{len(st.session_state.variables_to_drop)}** variables del dataset")
        else:
            st.info("ℹ️ No se realizarán cambios en las variables")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Aplicar y Continuar", type="primary", use_container_width=True):
                if len(st.session_state.variables_to_keep) == 0:
                    st.error("❌ Debes seleccionar al menos una variable")
                else:
                    # Aplicar los cambios al dataframe
                    df_filtered = df[sorted(st.session_state.variables_to_keep)].copy()
                    st.session_state.raw_data = df_filtered
                    
                    st.success(f"✅ Variables aplicadas: {df_filtered.shape[1]} columnas seleccionadas")
                    st.balloons()
                    
                    # Mostrar resultado
                    st.markdown("---")
                    st.markdown("**Vista previa del dataset filtrado:**")
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Filas", f"{df_filtered.shape[0]:,}")
                    col_b.metric("Columnas", f"{df_filtered.shape[1]:,}")
                    
                    st.dataframe(df_filtered.head(20), use_container_width=True, height=300)
        
        with col2:
            if st.button("🔄 Restablecer Todo", use_container_width=True):
                st.session_state.variables_to_keep = set(df.columns.tolist())
                st.session_state.variables_to_drop = set()
                st.info("ℹ️ Selección restablecida a todas las variables")
                st.rerun()
        
        with col3:
            if st.button("💾 Guardar Selección", use_container_width=True):
                try:
                    # Guardar configuración de variables
                    selection_config = {
                        'variables_to_keep': sorted(list(st.session_state.variables_to_keep)),
                        'variables_to_drop': sorted(list(st.session_state.variables_to_drop)),
                        'timestamp': datetime.now().isoformat(),
                        'total_variables': len(df.columns)
                    }
                    
                    config_dir = Path(CONFIG.preprocessing_config_path).parent
                    config_dir.mkdir(parents=True, exist_ok=True)
                    
                    selection_path = config_dir / "variable_selection.json"
                    
                    with open(selection_path, 'w', encoding='utf-8') as f:
                        json.dump(selection_config, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"✅ Selección guardada en {selection_path}")
                except Exception as e:
                    st.error(f"❌ Error al guardar: {e}")


def data_cleaning_page():
    """Sección de limpieza de datos."""
    st.header("🧹 Limpieza de Datos")
    
    # Usar datos limpios si existen, sino usar datos crudos
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.info("📊 Usando datos crudos para limpieza")
    elif st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios existentes (se pueden re-procesar)")
    else:
        st.warning("⚠️ Primero carga un dataset en la pestaña 'Carga de Datos'")
        return
    
    # Configuración de limpieza en sidebar expandido
    with st.expander("⚙️ Configuración de Limpieza", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imputación de Valores Faltantes")
            
            numeric_imputation = st.selectbox(
                "Método para numéricas",
                ["mean", "median", "knn", "forward", "backward", "constant"],
                index=1,
                help="Media, mediana, KNN, relleno hacia adelante/atrás, constante"
            )
            
            if numeric_imputation == "knn":
                knn_neighbors = st.slider("Vecinos KNN", 1, 20, 5)
            else:
                knn_neighbors = 5
            
            if numeric_imputation == "constant":
                constant_fill_numeric = st.number_input("Valor constante (numérico)", value=0.0)
            else:
                constant_fill_numeric = 0.0
            
            categorical_imputation = st.selectbox(
                "Método para categóricas",
                ["mode", "constant", "forward", "backward"],
                index=0,
                help="Moda, constante, relleno hacia adelante/atrás"
            )
            
            if categorical_imputation == "constant":
                constant_fill_categorical = st.text_input("Valor constante (categórico)", "missing")
            else:
                constant_fill_categorical = "missing"
        
        with col2:
            st.subheader("Detección y Tratamiento de Outliers")
            
            outlier_method = st.selectbox(
                "Método de detección",
                ["iqr", "zscore", "none"],
                index=0,
                help="IQR (rango intercuartílico), Z-score, o ninguno"
            )
            
            if outlier_method == "iqr":
                iqr_multiplier = st.slider("Multiplicador IQR", 1.0, 3.0, 1.5, 0.1)
                zscore_threshold = 3.0
            elif outlier_method == "zscore":
                zscore_threshold = st.slider("Umbral Z-score", 2.0, 4.0, 3.0, 0.1)
                iqr_multiplier = 1.5
            else:
                iqr_multiplier = 1.5
                zscore_threshold = 3.0
            
            outlier_treatment = st.selectbox(
                "Tratamiento de outliers",
                ["cap", "remove", "none"],
                index=0,
                help="Limitar a rangos válidos, eliminar (marcar como NaN), o no tratar"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Codificación de Categóricas")
            
            categorical_encoding = st.selectbox(
                "Tipo de codificación",
                ["label", "onehot", "ordinal", "none"],
                index=0,
                help="Label encoding, one-hot, ordinal (requiere orden), o ninguno"
            )
        
        with col4:
            st.subheader("Discretización (Opcional)")
            
            discretization_strategy = st.selectbox(
                "Estrategia de discretización",
                ["none", "quantile", "uniform", "custom"],
                index=0,
                help="Ninguna, cuantiles, uniforme, o bins personalizados"
            )
            
            if discretization_strategy in ["quantile", "uniform"]:
                discretization_bins = st.slider("Número de bins", 2, 10, 5)
            else:
                discretization_bins = 5
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Opciones Generales")
            drop_duplicates = st.checkbox("Eliminar duplicados", value=True)
            drop_fully_missing = st.checkbox("Eliminar columnas totalmente vacías", value=True)
        
        with col6:
            st.write("")  # Espaciado
            drop_constant = st.checkbox("Eliminar columnas constantes", value=True)
            constant_threshold = st.slider("Umbral constante (%)", 50, 100, 95) / 100
    
    # Configuración personalizada por variable
    st.markdown("---")
    with st.expander("🎯 Configuración Personalizada por Variable (Opcional)", expanded=False):
        st.markdown("""
        Aquí puedes configurar estrategias específicas de imputación y discretización 
        para variables individuales. Si no se especifica, se usa la configuración global.
        """)
        
        # Identificar columnas numéricas y categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Tabs para imputación y discretización
        tab_impute, tab_discretize = st.tabs(["💉 Imputación Personalizada", "📊 Discretización Personalizada"])
        
        with tab_impute:
            st.subheader("Configurar imputación por variable")
            
            # Inicializar diccionarios en session_state
            if 'custom_imputation' not in st.session_state:
                st.session_state.custom_imputation = {}
            if 'custom_constant_values' not in st.session_state:
                st.session_state.custom_constant_values = {}
            
            # Seleccionar variable para configurar
            col1, col2 = st.columns([2, 1])
            
            with col1:
                var_to_config = st.selectbox(
                    "Selecciona variable para configurar",
                    [""] + df.columns.tolist(),
                    key="impute_var_select"
                )
            
            with col2:
                if st.button("🗑️ Limpiar todas las configuraciones", key="clear_impute_config"):
                    st.session_state.custom_imputation = {}
                    st.session_state.custom_constant_values = {}
                    st.rerun()
            
            if var_to_config:
                is_numeric = var_to_config in numeric_cols
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Estrategias según tipo
                    if is_numeric:
                        strategies = ["mean", "median", "knn", "forward", "backward", "constant_numeric"]
                    else:
                        strategies = ["mode", "forward", "backward", "constant_categorical"]
                    
                    current_strategy = st.session_state.custom_imputation.get(var_to_config, "")
                    selected_strategy = st.selectbox(
                        f"Estrategia para {var_to_config}",
                        ["(usar global)"] + strategies,
                        index=strategies.index(current_strategy) + 1 if current_strategy in strategies else 0,
                        key=f"strategy_{var_to_config}"
                    )
                
                with col2:
                    # Valor constante si aplica
                    if selected_strategy in ["constant_numeric", "constant_categorical"]:
                        if is_numeric:
                            constant_val = st.number_input(
                                "Valor constante",
                                value=st.session_state.custom_constant_values.get(var_to_config, 0.0),
                                key=f"const_{var_to_config}"
                            )
                        else:
                            constant_val = st.text_input(
                                "Valor constante",
                                value=st.session_state.custom_constant_values.get(var_to_config, "missing"),
                                key=f"const_{var_to_config}"
                            )
                
                # Botones de acción
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Aplicar configuración", key=f"apply_{var_to_config}"):
                        if selected_strategy != "(usar global)":
                            st.session_state.custom_imputation[var_to_config] = selected_strategy
                            if selected_strategy in ["constant_numeric", "constant_categorical"]:
                                st.session_state.custom_constant_values[var_to_config] = constant_val
                            st.success(f"✅ Configuración guardada para {var_to_config}")
                        else:
                            # Remover configuración personalizada
                            if var_to_config in st.session_state.custom_imputation:
                                del st.session_state.custom_imputation[var_to_config]
                            if var_to_config in st.session_state.custom_constant_values:
                                del st.session_state.custom_constant_values[var_to_config]
                            st.info(f"ℹ️ {var_to_config} usará la configuración global")
                
                with col2:
                    if st.button("🗑️ Eliminar configuración", key=f"remove_{var_to_config}"):
                        if var_to_config in st.session_state.custom_imputation:
                            del st.session_state.custom_imputation[var_to_config]
                        if var_to_config in st.session_state.custom_constant_values:
                            del st.session_state.custom_constant_values[var_to_config]
                        st.info(f"ℹ️ Configuración eliminada para {var_to_config}")
            
            # Mostrar configuraciones actuales
            if st.session_state.custom_imputation:
                st.markdown("---")
                st.markdown("**Configuraciones personalizadas activas:**")
                config_df = pd.DataFrame([
                    {
                        'Variable': var,
                        'Estrategia': strategy,
                        'Valor Constante': st.session_state.custom_constant_values.get(var, '-')
                    }
                    for var, strategy in st.session_state.custom_imputation.items()
                ])
                st.dataframe(config_df, width='stretch', hide_index=True)
        
        with tab_discretize:
            st.subheader("Configurar discretización por variable")
            
            # Inicializar diccionarios en session_state
            if 'custom_discretization' not in st.session_state:
                st.session_state.custom_discretization = {}
            if 'custom_discretization_bins' not in st.session_state:
                st.session_state.custom_discretization_bins = {}
            
            # Seleccionar variable para configurar
            col1, col2 = st.columns([2, 1])
            
            with col1:
                var_to_disc = st.selectbox(
                    "Selecciona variable numérica para discretizar",
                    [""] + numeric_cols,
                    key="disc_var_select"
                )
            
            with col2:
                if st.button("🗑️ Limpiar todas las configuraciones", key="clear_disc_config"):
                    st.session_state.custom_discretization = {}
                    st.session_state.custom_discretization_bins = {}
                    st.rerun()
            
            if var_to_disc:
                col1, col2 = st.columns(2)
                
                with col1:
                    strategies = ["none", "uniform", "quantile", "kmeans", "custom"]
                    current_strategy = st.session_state.custom_discretization.get(var_to_disc, "")
                    selected_disc_strategy = st.selectbox(
                        f"Estrategia para {var_to_disc}",
                        ["(usar global)"] + strategies,
                        index=strategies.index(current_strategy) + 1 if current_strategy in strategies else 0,
                        key=f"disc_strategy_{var_to_disc}"
                    )
                
                with col2:
                    if selected_disc_strategy not in ["(usar global)", "none", "custom"]:
                        n_bins = st.number_input(
                            "Número de bins",
                            min_value=2,
                            max_value=20,
                            value=st.session_state.custom_discretization_bins.get(var_to_disc, 5),
                            key=f"bins_{var_to_disc}"
                        )
                
                # Botones de acción
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Aplicar configuración", key=f"apply_disc_{var_to_disc}"):
                        if selected_disc_strategy != "(usar global)":
                            st.session_state.custom_discretization[var_to_disc] = selected_disc_strategy
                            if selected_disc_strategy not in ["none", "custom"]:
                                st.session_state.custom_discretization_bins[var_to_disc] = n_bins
                            st.success(f"✅ Discretización configurada para {var_to_disc}")
                        else:
                            # Remover configuración personalizada
                            if var_to_disc in st.session_state.custom_discretization:
                                del st.session_state.custom_discretization[var_to_disc]
                            if var_to_disc in st.session_state.custom_discretization_bins:
                                del st.session_state.custom_discretization_bins[var_to_disc]
                            st.info(f"ℹ️ {var_to_disc} usará la configuración global")
                
                with col2:
                    if st.button("🗑️ Eliminar configuración", key=f"remove_disc_{var_to_disc}"):
                        if var_to_disc in st.session_state.custom_discretization:
                            del st.session_state.custom_discretization[var_to_disc]
                        if var_to_disc in st.session_state.custom_discretization_bins:
                            del st.session_state.custom_discretization_bins[var_to_disc]
                        st.info(f"ℹ️ Configuración eliminada para {var_to_disc}")
            
            # Mostrar configuraciones actuales
            if st.session_state.custom_discretization:
                st.markdown("---")
                st.markdown("**Configuraciones de discretización activas:**")
                config_df = pd.DataFrame([
                    {
                        'Variable': var,
                        'Estrategia': strategy,
                        'Bins': st.session_state.custom_discretization_bins.get(var, '-')
                    }
                    for var, strategy in st.session_state.custom_discretization.items()
                ])
                st.dataframe(config_df, width='stretch', hide_index=True)
    
    # Crear configuración
    config = CleaningConfig(
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        knn_neighbors=knn_neighbors,
        constant_fill_numeric=constant_fill_numeric,
        constant_fill_categorical=constant_fill_categorical,
        custom_imputation_strategies=st.session_state.get('custom_imputation', {}),
        custom_constant_values=st.session_state.get('custom_constant_values', {}),
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
        outlier_treatment=outlier_treatment,
        categorical_encoding=categorical_encoding,
        discretization_strategy=discretization_strategy,
        discretization_bins=discretization_bins,
        custom_discretization_strategies=st.session_state.get('custom_discretization', {}),
        custom_discretization_bins=st.session_state.get('custom_discretization_bins', {}),
        drop_duplicates=drop_duplicates,
        drop_fully_missing=drop_fully_missing,
        drop_constant=drop_constant,
        constant_threshold=constant_threshold,
    )
    
    st.session_state.cleaning_config = config
    
    # Botón para aplicar limpieza
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        target_column = st.selectbox(
            "Columna objetivo (no se modifica)",
            [""] + df.columns.tolist(),
            help="Selecciona la variable objetivo si existe"
        )
        target_column = target_column if target_column else None
    
    with col2:
        if st.button("🚀 Aplicar Limpieza", type="primary", width='stretch'):
            with st.spinner("Limpiando datos..."):
                try:
                    cleaner = DataCleaner(config)
                    df_clean = cleaner.fit_transform(df, target_column=target_column)
                    
                    st.session_state.cleaned_data = df_clean
                    st.session_state.cleaner = cleaner
                    
                    st.success("✅ Limpieza completada!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error durante la limpieza: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col3:
        if st.button("💾 Guardar Configuración", width='stretch'):
            try:
                config_path = Path(CONFIG.preprocessing_config_path)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
                
                st.success(f"✅ Configuración guardada en {config_path}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Mostrar resultados
    if st.session_state.cleaned_data is not None:
        st.markdown("---")
        st.subheader("📊 Resultados de Limpieza")
        
        df_clean = st.session_state.cleaned_data
        cleaner = st.session_state.cleaner
        
        # Métricas comparativas
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Filas",
            f"{df_clean.shape[0]:,}",
            f"{df_clean.shape[0] - df.shape[0]:+,}"
        )
        col2.metric(
            "Columnas",
            f"{df_clean.shape[1]:,}",
            f"{df_clean.shape[1] - df.shape[1]:+,}"
        )
        col3.metric(
            "Valores faltantes",
            f"{df_clean.isna().sum().sum():,}",
            f"{df_clean.isna().sum().sum() - df.isna().sum().sum():+,}"
        )
        col4.metric(
            "% Completitud",
            f"{(1 - df_clean.isna().mean().mean()) * 100:.1f}%",
            f"{((1 - df_clean.isna().mean().mean()) - (1 - df.isna().mean().mean())) * 100:+.1f}%"
        )
        
        # Tabs con detalles
        tab1, tab2, tab3 = st.tabs(["Datos Limpios", "Reporte de Calidad", "Metadatos"])
        
        with tab1:
            st.dataframe(df_clean.head(100), width='stretch', height=400)
            
            # Botones de descarga
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV limpio",
                    data=csv_data,
                    file_name=f"cleaned_dataset_{timestamp}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            with col2:
                if st.button("💾 Guardar en Cleaned Datasets", width='stretch'):
                    try:
                        # Use absolute path from dashboard config
                        cleaned_dir = CLEANED_DATASETS_DIR
                        cleaned_dir.mkdir(parents=True, exist_ok=True)
                        
                        save_path = cleaned_dir / f"cleaned_dataset_{timestamp}.csv"
                        df_clean.to_csv(save_path, index=False)
                        
                        st.success(f"✅ Guardado en: {save_path}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with tab2:
            if cleaner:
                report = cleaner.get_cleaning_report()
                
                st.subheader("Resumen de Operaciones")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Variables procesadas", report.get('variables_cleaned', 0))
                col2.metric("Variables con outliers", report.get('variables_with_outliers', 0))
                col3.metric("Variables imputadas", report.get('variables_imputed', 0))
                
                col1, col2 = st.columns(2)
                col1.metric("Variables codificadas", report.get('variables_encoded', 0))
                col2.metric("Variables discretizadas", report.get('variables_discretized', 0))
                
                # Problemas de calidad
                if report['quality_issues']:
                    st.subheader("⚠️ Alertas de Calidad")
                    
                    for var, flags in report['quality_issues'].items():
                        with st.expander(f"Variable: {var}"):
                            for flag in flags:
                                if 'vacia' in flag or 'constante' in flag:
                                    st.error(f"🔴 {flag}")
                                elif 'outliers' in flag or 'missing' in flag:
                                    st.warning(f"🟡 {flag}")
                                else:
                                    st.info(f"🟢 {flag}")
        
        with tab3:
            if cleaner and cleaner.metadata:
                st.subheader("Metadatos de Variables")
                
                # Tabla resumen
                metadata_rows = []
                for name, meta in cleaner.metadata.items():
                    metadata_rows.append({
                        'Variable': name,
                        'Tipo': meta.cleaned_type,
                        'Missing Original (%)': f"{meta.missing_percent_original:.1f}",
                        'Método Imputación': meta.imputation_method or '-',
                        'Codificación': meta.encoding_type or '-',
                        'Outliers Tratados': meta.outliers_treated
                    })
                
                df_meta = pd.DataFrame(metadata_rows)
                st.dataframe(df_meta, width='stretch', height=400)
                
                # Guardar metadatos
                if st.button("💾 Guardar Metadatos como JSON"):
                    try:
                        metadata_path = Path(CONFIG.metadata_path)
                        metadata_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        cleaner.save_metadata(metadata_path)
                        st.success(f"✅ Metadatos guardados en: {metadata_path}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")


def univariate_analysis_page():
    """Sección de análisis univariado."""
    st.header("📈 Análisis Univariado")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios")
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.warning("⚠️ Usando datos crudos (no limpios)")
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    # Crear o recuperar analyzer
    if st.session_state.analyzer is None or st.session_state.analyzer.df.shape != df.shape:
        with st.spinner("Inicializando analizador..."):
            analyzer = EDAAnalyzer(df)
            analyzer.analyze_univariate()
            st.session_state.analyzer = analyzer
    else:
        analyzer = st.session_state.analyzer
    
    # Seleccionar variable
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_var = st.selectbox(
            "Selecciona una variable para analizar",
            df.columns.tolist(),
            help="Elige una variable para ver estadísticas y visualizaciones"
        )
    
    with col2:
        if st.button("🔄 Reanalizar", width='stretch'):
            with st.spinner("Analizando..."):
                analyzer.analyze_univariate([selected_var])
                st.success("✅ Análisis actualizado")
    
    if selected_var not in analyzer.univariate_results:
        analyzer.analyze_univariate([selected_var])
    
    stats = analyzer.univariate_results[selected_var]
    
    # Mostrar estadísticas
    st.markdown("---")
    st.subheader(f"Estadísticas de: **{selected_var}**")
    
    if stats.variable_type == 'numerical':
        # Estadísticas numéricas
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Media", f"{stats.mean:.2f}" if stats.mean else "N/A")
        col2.metric("Mediana", f"{stats.median:.2f}" if stats.median else "N/A")
        col3.metric("Desv. Estándar", f"{stats.std:.2f}" if stats.std else "N/A")
        col4.metric("Missing (%)", f"{stats.missing_percent:.1f}%")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Mínimo", f"{stats.min:.2f}" if stats.min else "N/A")
        col2.metric("Q1 (25%)", f"{stats.q25:.2f}" if stats.q25 else "N/A")
        col3.metric("Q3 (75%)", f"{stats.q75:.2f}" if stats.q75 else "N/A")
        col4.metric("Máximo", f"{stats.max:.2f}" if stats.max else "N/A")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Asimetría (Skewness)", f"{stats.skewness:.2f}" if stats.skewness else "N/A")
        col2.metric("Curtosis (Kurtosis)", f"{stats.kurtosis:.2f}" if stats.kurtosis else "N/A")
        col3.metric("Conteo", f"{stats.count:,}" if stats.count else "0")
        
        # Interpretación de skewness
        if stats.skewness is not None:
            if abs(stats.skewness) < 0.5:
                skew_msg = "✅ Distribución aproximadamente simétrica"
                skew_color = "green"
            elif abs(stats.skewness) < 1:
                skew_msg = "⚠️ Distribución moderadamente asimétrica"
                skew_color = "orange"
            else:
                skew_msg = "🔴 Distribución altamente asimétrica"
                skew_color = "red"
            
            st.markdown(f":{skew_color}[{skew_msg}]")
        
        # Visualizaciones
        st.markdown("---")
        st.subheader("Visualizaciones")
        
        tab1, tab2, tab3 = st.tabs(["Histograma + KDE", "Boxplot", "Violin Plot"])
        
        with tab1:
            fig = analyzer.plot_distribution(selected_var, plot_type='histogram')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = analyzer.plot_distribution(selected_var, plot_type='box')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = analyzer.plot_distribution(selected_var, plot_type='violin')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Estadísticas categóricas
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Categorías únicas", f"{stats.n_categories:,}" if stats.n_categories else "0")
        col2.metric("Moda", str(stats.mode) if stats.mode else "N/A")
        col3.metric("Frecuencia moda", f"{stats.mode_frequency:,}" if stats.mode_frequency else "0")
        col4.metric("Missing (%)", f"{stats.missing_percent:.1f}%")
        
        # Tabla de frecuencias
        st.markdown("---")
        st.subheader("Tabla de Frecuencias")
        
        if stats.category_counts:
            freq_df = pd.DataFrame([
                {'Categoría': k, 'Frecuencia': v, 'Porcentaje': f"{v/stats.count*100:.1f}%"}
                for k, v in sorted(stats.category_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(freq_df, width='stretch', height=300)
        
        # Visualizaciones
        st.markdown("---")
        st.subheader("Visualizaciones")
        
        tab1, tab2 = st.tabs(["Gráfico de Barras", "Gráfico Circular"])
        
        with tab1:
            fig = analyzer.plot_distribution(selected_var, plot_type='bar')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = analyzer.plot_distribution(selected_var, plot_type='pie')
            st.plotly_chart(fig, use_container_width=True)
    
    # Exportación PDF
    st.markdown("---")
    
    def generate_univariate_report():
        """Generate univariate analysis PDF report."""
        from pathlib import Path
        output_path = Path("reports") / "univariate_eda.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return generate_univariate_pdf(
            df=df,
            numerical_cols=analyzer.numeric_cols,
            categorical_cols=analyzer.categorical_cols,
            output_path=output_path
        )
    
    pdf_export_section(
        generate_univariate_report,
        section_title="Análisis Univariado",
        default_filename="univariate_eda.pdf",
        key_prefix="univariate_eda"
    )


def bivariate_analysis_page():
    """Sección de análisis bivariado."""
    st.header("📊 Análisis Bivariado")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios")
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.warning("⚠️ Usando datos crudos (no limpios)")
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    # Crear o recuperar analyzer
    if st.session_state.analyzer is None or st.session_state.analyzer.df.shape != df.shape:
        with st.spinner("Inicializando analizador..."):
            analyzer = EDAAnalyzer(df)
            st.session_state.analyzer = analyzer
    else:
        analyzer = st.session_state.analyzer
        # Forzar recreación si bivariate_results es una lista (versión antigua)
        if isinstance(analyzer.bivariate_results, list):
            with st.spinner("Actualizando analizador..."):
                analyzer = EDAAnalyzer(df)
                st.session_state.analyzer = analyzer
    
    # Seleccionar variables
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox(
            "Variable 1",
            df.columns.tolist(),
            help="Primera variable para análisis"
        )
    
    with col2:
        var2 = st.selectbox(
            "Variable 2",
            [col for col in df.columns if col != var1],
            help="Segunda variable para análisis"
        )
    
    if var1 == var2:
        st.warning("⚠️ Selecciona variables diferentes")
        return
    
    # Analizar y mostrar resultados
    if st.button("🔍 Analizar Relación", type="primary"):
        with st.spinner("Analizando relación bivariada..."):
            result = analyzer.analyze_bivariate(var1, var2)
            
            # Mostrar resultados inmediatamente después del análisis
            key = f"{var1}_vs_{var2}"
            if key in analyzer.bivariate_results:
                result = analyzer.bivariate_results[key]
                
                st.markdown("---")
                st.subheader(f"Relación: **{var1}** vs **{var2}**")
                
                # Mostrar métricas según tipo de relación
                if result.relationship_type == "num-num":
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Correlación de Pearson", f"{result.pearson_corr:.3f}" if result.pearson_corr else "N/A")
                    col2.metric("Correlación de Spearman", f"{result.spearman_corr:.3f}" if result.spearman_corr else "N/A")
                    col3.metric("p-value", f"{result.pearson_pvalue:.4f}" if result.pearson_pvalue else "N/A")
                    
                    # Interpretación
                    if result.pearson_corr and abs(result.pearson_corr) > 0.7:
                        st.success("✅ Correlación fuerte detectada")
                    elif result.pearson_corr and abs(result.pearson_corr) > 0.4:
                        st.info("📊 Correlación moderada")
                    else:
                        st.warning("⚠️ Correlación débil o no significativa")
                    
                    # Visualización
                    try:
                        fig = analyzer.plot_scatter(var1, var2, add_trendline=True)
                    except (ImportError, ModuleNotFoundError):
                        # Si statsmodels no está instalado, mostrar sin trendline
                        fig = analyzer.plot_scatter(var1, var2, add_trendline=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif result.relationship_type == "cat-cat":
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Chi-cuadrado", f"{result.chi2_statistic:.2f}" if result.chi2_statistic else "N/A")
                    col2.metric("p-value", f"{result.chi2_pvalue:.4f}" if result.chi2_pvalue else "N/A")
                    col3.metric("Cramér's V", f"{result.cramers_v:.3f}" if result.cramers_v else "N/A")
                    
                    if result.chi2_pvalue and result.chi2_pvalue < 0.05:
                        st.success("✅ Relación estadísticamente significativa (p < 0.05)")
                    else:
                        st.info("📊 No hay evidencia de relación significativa")
                    
                    # Tabla de contingencia
                    st.subheader("Tabla de Contingencia")
                    contingency = pd.crosstab(df[var1], df[var2])
                    st.dataframe(contingency, width='stretch')
                
                else:  # num-cat (ANOVA)
                    col1, col2 = st.columns(2)
                    
                    col1.metric("F-statistic (ANOVA)", f"{result.anova_f:.3f}" if result.anova_f else "N/A")
                    col2.metric("p-value", f"{result.anova_pvalue:.4f}" if result.anova_pvalue else "N/A")
                    
                    if result.anova_pvalue and result.anova_pvalue < 0.05:
                        st.success("✅ Diferencias significativas entre grupos (p < 0.05)")
                    else:
                        st.info("📊 No hay evidencia de diferencias significativas")
                    
                    # Estadísticas por grupo
                    st.subheader("Estadísticas por Grupo")
                    # Determinar cuál es numérica y cuál categórica
                    num_var = var1 if var1 in analyzer.numeric_cols else var2
                    cat_var = var2 if var1 in analyzer.numeric_cols else var1
                    group_stats = df.groupby(cat_var)[num_var].describe()
                    st.dataframe(group_stats, width='stretch')
    
    # Exportación PDF
    st.markdown("---")
    if analyzer.bivariate_results:
        
        def generate_bivariate_report():
            """Generate bivariate analysis PDF report."""
            from pathlib import Path
            output_path = Path("reports") / "bivariate_eda.pdf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return generate_bivariate_pdf(
                df=df,
                numerical_cols=analyzer.numeric_cols,
                output_path=output_path,
                correlation_threshold=0.3
            )
        
        pdf_export_section(
            generate_bivariate_report,
            section_title="Análisis Bivariado",
            default_filename="bivariate_eda.pdf",
            key_prefix="bivariate_eda"
        )


def multivariate_analysis_page():
    """Sección de análisis multivariado."""
    st.header("🔬 Análisis Multivariado")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios")
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.warning("⚠️ Usando datos crudos (no limpios)")
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    # Crear o recuperar analyzer
    if st.session_state.analyzer is None or st.session_state.analyzer.df.shape != df.shape:
        with st.spinner("Inicializando analizador..."):
            analyzer = EDAAnalyzer(df)
            st.session_state.analyzer = analyzer
    else:
        analyzer = st.session_state.analyzer
    
    # Tabs para diferentes análisis
    tabs = st.tabs(["📊 Matriz de Correlación", "🎯 PCA (Análisis de Componentes Principales)"])
    
    with tabs[0]:
        st.subheader("Matriz de Correlación")
        
        if len(analyzer.numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para análisis de correlación")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                corr_method = st.selectbox(
                    "Método de correlación",
                    ["pearson", "spearman", "kendall"],
                    help="Pearson: lineal, Spearman: monotónica, Kendall: ordinal"
                )
            
            with col2:
                min_corr = st.slider(
                    "Filtrar correlaciones < ",
                    0.0, 0.9, 0.0, 0.05,
                    help="Mostrar solo correlaciones mayores a este umbral"
                )
            
            if st.button("📊 Calcular Matriz de Correlación", type="primary"):
                with st.spinner("Calculando correlaciones..."):
                    try:
                        # Calcular matriz de correlación usando el analyzer
                        corr_matrix = analyzer.analyze_multivariate(method=corr_method)
                        
                        # Visualización
                        fig = analyzer.plot_correlation_matrix(method=corr_method)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabla de correlaciones más altas
                        st.subheader("Top Correlaciones")
                        
                        # Extraer pares de correlación
                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                var1 = corr_matrix.columns[i]
                                var2 = corr_matrix.columns[j]
                                corr_val = corr_matrix.iloc[i, j]
                                
                                if not pd.isna(corr_val) and abs(corr_val) >= min_corr:
                                    corr_pairs.append({
                                        'Variable 1': var1,
                                        'Variable 2': var2,
                                        'Correlación': corr_val,
                                        'Correlación Abs': abs(corr_val)
                                    })
                        
                        if corr_pairs:
                            corr_df = pd.DataFrame(corr_pairs)
                            corr_df = corr_df.sort_values('Correlación Abs', ascending=False)
                            corr_df = corr_df.drop('Correlación Abs', axis=1)
                            
                            st.dataframe(corr_df.head(20), width='stretch')
                        else:
                            st.info("No se encontraron correlaciones significativas con el filtro aplicado")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    with tabs[1]:
        st.subheader("Análisis de Componentes Principales (PCA)")
        
        if len(analyzer.numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para PCA")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pca_mode = st.radio(
                    "Modo de selección de componentes",
                    ["Automático (por varianza)", "Manual"],
                    help="Automático: selecciona componentes hasta alcanzar umbral de varianza"
                )
            
            with col2:
                if pca_mode == "Automático (por varianza)":
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
                        2, min(20, len(analyzer.numeric_cols)), 5
                    )
                    variance_threshold = 0.95
            
            with col3:
                scale_data = st.checkbox(
                    "Estandarizar datos",
                    value=True,
                    help="Recomendado cuando variables tienen diferentes escalas"
                )
            
            if st.button("🚀 Ejecutar PCA", type="primary"):
                with st.spinner("Ejecutando Análisis de Componentes Principales..."):
                    try:
                        # Validar que hay suficientes variables numéricas
                        if len(analyzer.numeric_cols) < 2:
                            st.error("❌ Se requieren al menos 2 variables numéricas para PCA")
                            st.stop()
                        
                        # Verificar cuántas filas completas hay
                        df_for_pca = df[analyzer.numeric_cols].dropna()
                        if len(df_for_pca) == 0:
                            st.error(
                                "❌ **No se puede ejecutar PCA: Todas las filas tienen valores faltantes**\n\n"
                                "El Análisis de Componentes Principales requiere datos completos en las variables numéricas."
                            )
                            
                            st.info(
                                "**📋 Solución recomendada:**\n\n"
                                "1. Ve a la sección **'🧹 Limpieza de Datos'** (arriba en esta misma página)\n"
                                "2. Configura la **imputación de valores faltantes** para las variables numéricas\n"
                                "3. Haz clic en **'🚀 Aplicar Limpieza'**\n"
                                "4. Regresa a esta sección para ejecutar PCA con los datos limpios"
                            )
                            
                            # Mostrar información sobre valores faltantes
                            missing_info = df[analyzer.numeric_cols].isnull().sum()
                            missing_pct = (missing_info / len(df) * 100).round(2)
                            missing_df = pd.DataFrame({
                                'Variable': missing_info.index,
                                'Valores Faltantes': missing_info.values,
                                'Porcentaje (%)': missing_pct.values
                            })
                            missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values(
                                'Valores Faltantes', ascending=False
                            )
                            
                            if len(missing_df) > 0:
                                st.warning("**⚠️ Variables numéricas con valores faltantes:**")
                                st.dataframe(missing_df, width='stretch', height=300)
                            
                            st.stop()
                        
                        if len(df_for_pca) < 2:
                            st.warning(
                                f"⚠️ **Datos insuficientes para PCA**\n\n"
                                f"Solo hay {len(df_for_pca)} fila(s) sin valores faltantes. "
                                "Se requieren al menos 2 observaciones completas.\n\n"
                                "**Solución:** Aplica imputación de valores faltantes para aumentar el número de filas válidas."
                            )
                            st.stop()
                        
                        pca_results = analyzer.perform_pca(
                            n_components=n_components,
                            variance_threshold=variance_threshold,
                            scale=scale_data
                        )
                        
                        st.success(f"✅ PCA completado: {pca_results.n_components} componentes")
                        
                        # Métricas
                        st.markdown("---")
                        st.subheader("Resultados de PCA")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric(
                            "Componentes Principales",
                            pca_results.n_components
                        )
                        col2.metric(
                            "Varianza Explicada",
                            f"{sum(pca_results.explained_variance_ratio) * 100:.1f}%"
                        )
                        col3.metric(
                            "Variables Originales",
                            len(pca_results.feature_names)
                        )
                        
                        # Tabla de varianza por componente
                        st.subheader("Varianza Explicada por Componente")
                        
                        variance_df = pd.DataFrame({
                            'Componente': [f'PC{i+1}' for i in range(pca_results.n_components)],
                            'Varianza Individual (%)': [v * 100 for v in pca_results.explained_variance_ratio],
                            'Varianza Acumulada (%)': [v * 100 for v in pca_results.cumulative_variance]
                        })
                        
                        st.dataframe(variance_df, width='stretch')
                        
                        # Visualizaciones
                        st.markdown("---")
                        
                        tab1, tab2, tab3 = st.tabs([
                            "Gráfico de Scree",
                            "Biplot",
                            "Importancia de Features"
                        ])
                        
                        with tab1:
                            st.subheader("Gráfico de Scree (Varianza Explicada)")
                            fig = analyzer.plot_pca_scree()
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            st.subheader("Biplot de PCA")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pc_x = st.selectbox("PC para eje X", range(1, pca_results.n_components + 1), index=0)
                            
                            with col2:
                                pc_y = st.selectbox("PC para eje Y", range(1, pca_results.n_components + 1), index=min(1, pca_results.n_components - 1))
                            
                            with col3:
                                n_features_show = st.slider("Features a mostrar", 5, 20, 10)
                            
                            if pc_x != pc_y:
                                fig = analyzer.plot_pca_biplot(pc_x=pc_x, pc_y=pc_y, n_features=n_features_show)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ Selecciona diferentes componentes para X e Y")
                        
                        with tab3:
                            st.subheader("Importancia de Features en Primeros Componentes")
                            
                            n_comp_importance = st.slider(
                                "Componentes a considerar",
                                1, min(5, pca_results.n_components), 
                                min(3, pca_results.n_components)
                            )
                            
                            importance_df = analyzer.get_feature_importance_pca(n_components=n_comp_importance)
                            
                            st.dataframe(importance_df, width='stretch', height=400)
                            
                            # Gráfico de barras
                            import plotly.express as px
                            fig = px.bar(
                                importance_df.head(20).reset_index(),
                                x='importance',
                                y='index',
                                orientation='h',
                                title='Top 20 Features por Importancia',
                                labels={'index': 'Feature', 'importance': 'Importancia'}
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Opción de guardar resultados transformados
                        st.markdown("---")
                        if st.button("💾 Guardar Datos Transformados (PCA)"):
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                # Use absolute path from dashboard config
                                pca_data_path = CLEANED_DATASETS_DIR / f"pca_transformed_{timestamp}.csv"
                                pca_data_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                pca_results.transformed_data.to_csv(pca_data_path, index=False)
                                st.success(f"✅ Datos PCA guardados en: {pca_data_path}")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                        
                    except ValueError as ve:
                        # Errores específicos de validación
                        st.error(f"❌ Error de validación: {ve}")
                        st.info(
                            "💡 **Sugerencias:**\n"
                            "- Asegúrate de haber aplicado limpieza de datos primero\n"
                            "- Verifica que las variables numéricas seleccionadas tengan datos válidos\n"
                            "- Aplica imputación de valores faltantes si es necesario"
                        )
                    except Exception as e:
                        st.error(f"❌ Error durante PCA: {e}")
                        with st.expander("Ver detalles del error"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # Exportación PDF
    st.markdown("---")
    
    def generate_multivariate_report():
        """Generate multivariate analysis PDF report."""
        from pathlib import Path
        output_path = Path("reports") / "multivariate_eda.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return generate_multivariate_pdf(
            df=df,
            numerical_cols=analyzer.numeric_cols,
            output_path=output_path,
            n_components=min(5, len(analyzer.numeric_cols)) if len(analyzer.numeric_cols) > 1 else None
        )
    
    pdf_export_section(
        generate_multivariate_report,
        section_title="Análisis Multivariado",
        default_filename="multivariate_eda.pdf",
        key_prefix="multivariate_eda"
    )


def quality_report_page():
    """Página de reporte de calidad de datos."""
    st.header("📋 Reporte de Calidad de Datos")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.success("✅ Analizando datos limpios")
        is_cleaned = True
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.info("📊 Analizando datos crudos")
        is_cleaned = False
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    st.markdown("---")
    
    # Resumen general
    st.subheader("📊 Resumen General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total de Filas", f"{df.shape[0]:,}")
    col2.metric("Total de Columnas", f"{df.shape[1]:,}")
    col3.metric("Celdas Totales", f"{df.shape[0] * df.shape[1]:,}")
    
    total_missing = df.isna().sum().sum()
    missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
    col4.metric("Valores Faltantes", f"{total_missing:,} ({missing_pct:.1f}%)")
    
    # Tipos de variables
    st.markdown("---")
    st.subheader("🏷️ Tipos de Variables")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Variables Numéricas", len(numeric_cols))
        if numeric_cols:
            with st.expander("Ver lista"):
                st.write(numeric_cols)
    
    with col2:
        st.metric("Variables Categóricas", len(categorical_cols))
        if categorical_cols:
            with st.expander("Ver lista"):
                st.write(categorical_cols)
    
    # Análisis de valores faltantes
    st.markdown("---")
    st.subheader("❓ Análisis de Valores Faltantes")
    
    missing_df = pd.DataFrame({
        'Variable': df.columns,
        'Missing Count': df.isna().sum().values,
        'Missing %': (df.isna().sum() / len(df) * 100).values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing_df) > 0:
        # Gráfico de barras
        import plotly.express as px
        fig = px.bar(
            missing_df.head(20),
            x='Missing %',
            y='Variable',
            orientation='h',
            title='Top 20 Variables con Valores Faltantes',
            color='Missing %',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.dataframe(missing_df, width='stretch', height=300)
        
        # Alertas
        critical_missing = missing_df[missing_df['Missing %'] > 50]
        if len(critical_missing) > 0:
            st.error(f"🔴 **CRÍTICO**: {len(critical_missing)} variables tienen >50% de valores faltantes")
            with st.expander("Ver variables críticas"):
                st.dataframe(critical_missing, width='stretch')
    else:
        st.success("✅ ¡No hay valores faltantes en el dataset!")
    
    # Análisis de duplicados
    st.markdown("---")
    st.subheader("🔄 Análisis de Duplicados")
    
    n_duplicates = df.duplicated().sum()
    pct_duplicates = (n_duplicates / len(df)) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Filas Duplicadas", f"{n_duplicates:,}")
    col2.metric("Porcentaje", f"{pct_duplicates:.2f}%")
    
    if n_duplicates > 0:
        st.warning(f"⚠️ Se encontraron {n_duplicates} filas duplicadas")
    else:
        st.success("✅ No hay filas duplicadas")
    
    # Análisis de cardinalidad
    st.markdown("---")
    st.subheader("🎯 Cardinalidad de Variables")
    
    cardinality_df = pd.DataFrame({
        'Variable': df.columns,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Cardinality %': [(df[col].nunique() / len(df)) * 100 for col in df.columns]
    })
    cardinality_df = cardinality_df.sort_values('Unique Values', ascending=False)
    
    # Identificar problemas
    high_card = cardinality_df[cardinality_df['Cardinality %'] > 95]
    low_card = cardinality_df[cardinality_df['Unique Values'] == 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(high_card) > 0:
            st.warning(f"⚠️ {len(high_card)} variables con alta cardinalidad (>95%)")
            with st.expander("Ver variables"):
                st.dataframe(high_card, width='stretch')
    
    with col2:
        if len(low_card) > 0:
            st.error(f"🔴 {len(low_card)} variables constantes (1 valor único)")
            with st.expander("Ver variables"):
                st.dataframe(low_card, width='stretch')
    
    st.dataframe(cardinality_df, width='stretch', height=300)
    
    # Análisis de outliers (solo numéricas)
    if len(numeric_cols) > 0:
        st.markdown("---")
        st.subheader("📊 Análisis de Outliers (Método IQR)")
        
        outlier_summary = []
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                n_outliers = ((series < lower) | (series > upper)).sum()
                pct_outliers = (n_outliers / len(series)) * 100
                
                outlier_summary.append({
                    'Variable': col,
                    'Outliers Count': n_outliers,
                    'Outliers %': pct_outliers
                })
        
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df[outlier_df['Outliers Count'] > 0].sort_values('Outliers %', ascending=False)
        
        if len(outlier_df) > 0:
            st.dataframe(outlier_df, width='stretch', height=300)
            
            # Gráfico
            import plotly.express as px
            fig = px.bar(
                outlier_df.head(15),
                x='Outliers %',
                y='Variable',
                orientation='h',
                title='Variables con Mayor Porcentaje de Outliers',
                color='Outliers %',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No se detectaron outliers significativos (método IQR)")
    
    # Reporte final
    st.markdown("---")
    st.subheader("📝 Resumen de Calidad")
    
    quality_score = 100
    issues = []
    
    if missing_pct > 10:
        quality_score -= 20
        issues.append(f"🔴 Alto porcentaje de valores faltantes ({missing_pct:.1f}%)")
    elif missing_pct > 5:
        quality_score -= 10
        issues.append(f"🟡 Porcentaje moderado de valores faltantes ({missing_pct:.1f}%)")
    
    if pct_duplicates > 5:
        quality_score -= 15
        issues.append(f"🔴 Alto porcentaje de duplicados ({pct_duplicates:.1f}%)")
    elif pct_duplicates > 0:
        quality_score -= 5
        issues.append(f"🟡 Hay filas duplicadas ({pct_duplicates:.1f}%)")
    
    if len(low_card) > 0:
        quality_score -= 10
        issues.append(f"🔴 {len(low_card)} variables constantes detectadas")
    
    if len(high_card) > 5:
        quality_score -= 10
        issues.append(f"🟡 {len(high_card)} variables con alta cardinalidad")
    
    # Mostrar puntuación
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if quality_score >= 90:
            st.success(f"### 🌟 Puntuación de Calidad: {quality_score}/100")
        elif quality_score >= 70:
            st.warning(f"### ⚠️ Puntuación de Calidad: {quality_score}/100")
        else:
            st.error(f"### 🔴 Puntuación de Calidad: {quality_score}/100")
    
    with col2:
        if issues:
            st.markdown("**Problemas detectados:**")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No se detectaron problemas significativos de calidad")
    
    # Botón de descarga de reporte
    st.markdown("---")
    if st.button("📥 Descargar Reporte Completo (JSON)"):
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_type': 'cleaned' if is_cleaned else 'raw',
            'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
            'missing_summary': missing_df.to_dict('records') if len(missing_df) > 0 else [],
            'duplicates': {'count': int(n_duplicates), 'percentage': float(pct_duplicates)},
            'cardinality': cardinality_df.to_dict('records'),
            'quality_score': quality_score,
            'issues': issues
        }
        
        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        st.download_button(
            "Descargar JSON",
            data=report_json,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Función principal de la aplicación."""
    init_session_state()
    
    # Título y descripción
    st.title("🧹 Preparación de Datos y Análisis Exploratorio")
    st.markdown("""
    **Sistema completo de limpieza y análisis de datos para el predictor de mortalidad por IAM**
    
    Utiliza las pestañas siguientes para:
    - Cargar datos desde múltiples fuentes
    - Seleccionar variables relevantes para el análisis
    - Limpiar y preprocesar datos con configuraciones avanzadas
    - Realizar análisis exploratorio completo
    - Generar reportes de calidad
    """)
    
    # Tabs principales
    tabs = st.tabs([
        "📂 Carga de Datos",
        "🎯 Selección de Variables",
        "🧹 Limpieza de Datos",
        "📈 Análisis Univariado",
        "📊 Análisis Bivariado",
        "🔬 Análisis Multivariado",
        "📋 Reporte de Calidad"
    ])
    
    with tabs[0]:
        load_data_page()
    
    with tabs[1]:
        variable_selection_page()
    
    with tabs[2]:
        data_cleaning_page()
    
    with tabs[3]:
        univariate_analysis_page()
    
    with tabs[4]:
        bivariate_analysis_page()
    
    with tabs[5]:
        multivariate_analysis_page()
    
    with tabs[6]:
        quality_report_page()


if __name__ == "__main__":
    main()
