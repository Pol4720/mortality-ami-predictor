"""Data Overview and Exploration page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from app import (
    display_data_audit,
    display_dataframe_info,
    display_dataset_preview,
    initialize_state,
)
from src.config import CONFIG
from src.features import safe_feature_columns

# Initialize
initialize_state()

# Page config
st.title("📊 Data Overview & Exploration")
st.markdown("---")

# Check if data has been loaded in Data Cleaning page
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    st.success("✅ Usando datos limpios del proceso de limpieza")
elif raw_data is not None:
    df = raw_data
    st.warning("⚠️ Usando datos crudos (se recomienda limpiar primero en la página anterior)")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Display basic info
display_dataframe_info(df)
# Display basic info
display_dataframe_info(df)

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📋 Preview", "🔍 Quality Audit", "📈 Statistics"])

with tab1:
    n_rows = st.slider("Number of rows to display", 5, 50, 10)
    display_dataset_preview(df, n_rows=n_rows)

with tab2:
    # Determine target column from session state
    target = st.session_state.get('target_column', CONFIG.target_column)
    
    # Get feature columns
    if target and target in df.columns:
        feature_cols = safe_feature_columns(df, [target])
    else:
        feature_cols = list(df.columns)
    
    st.info(f"Analyzing {len(feature_cols)} feature columns (excluding target: {target})")
    display_data_audit(df, feature_cols)

with tab3:
    st.subheader("📊 Column Statistics")
    
    # Select columns to analyze
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to view statistics",
        all_cols,
        default=all_cols[:5] if len(all_cols) > 5 else all_cols
    )
    
    if selected_cols:
        st.dataframe(
            df[selected_cols].describe(),
            width='stretch'
        )
        
        # Missing values visualization
        st.subheader("Missing Values")
        missing_df = df[selected_cols].isnull().sum().to_frame("Missing Count")
        missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
        
        st.dataframe(missing_df, width='stretch')
    else:
        st.info("Select at least one column to view statistics")
