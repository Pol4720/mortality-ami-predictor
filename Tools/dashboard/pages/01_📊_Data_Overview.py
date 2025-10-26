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
    get_state,
    initialize_state,
    load_data,
    sidebar_data_controls,
)
from src.config import CONFIG
from src.features import safe_feature_columns

# Initialize
initialize_state()

# Page config
st.title("📊 Data Overview & Exploration")
st.markdown("---")

# Sidebar controls
data_path, task = sidebar_data_controls()

if not data_path:
    st.warning("⚠️ Please provide a dataset path in the sidebar")
    st.stop()

# Load data
try:
    with st.spinner("Loading dataset..."):
        df = load_data(data_path)
    
    st.success(f"✅ Dataset loaded successfully")
    
    # Display basic info
    display_dataframe_info(df)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Preview", "🔍 Quality Audit", "📈 Statistics"])
    
    with tab1:
        n_rows = st.slider("Number of rows to display", 5, 50, 10)
        display_dataset_preview(df, n_rows=n_rows)
    
    with tab2:
        # Determine target column
        if task == "mortality":
            target = CONFIG.target_column
        else:
            target = CONFIG.arrhythmia_column
        
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
                use_container_width=True
            )
            
            # Missing values visualization
            st.subheader("Missing Values")
            missing_df = df[selected_cols].isnull().sum().to_frame("Missing Count")
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
            
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.info("Select at least one column to view statistics")

except FileNotFoundError:
    st.error(f"❌ Dataset file not found: {data_path}")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.exception(e)
