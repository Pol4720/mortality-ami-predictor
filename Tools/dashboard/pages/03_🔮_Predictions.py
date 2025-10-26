"""Predictions page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app import (
    get_state,
    initialize_state,
    list_saved_models,
)
from src.config import CONFIG

# Initialize
initialize_state()

# Page config
st.title("🔮 Make Predictions")
st.markdown("---")

# Check if data has been loaded
if st.session_state.cleaned_data is not None:
    df = st.session_state.cleaned_data
    st.success("✅ Usando datos limpios")
elif st.session_state.raw_data is not None:
    df = st.session_state.raw_data
    st.warning("⚠️ Usando datos crudos")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Get task from session state
task = st.session_state.get('target_column', 'mortality')
if task == 'exitus':
    task = 'mortality'

# Model selection
st.sidebar.markdown("---")
st.sidebar.header("🎯 Model Selection")

saved_models = list_saved_models(task)

if not saved_models:
    st.error(f"❌ No trained models found for task '{task}'. Please train models first.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(saved_models.keys()),
    help="Select a trained model for predictions"
)

model_path = saved_models[selected_model_name]
st.sidebar.success(f"✅ Using: {selected_model_name}")

# Load model
try:
    model = joblib.load(model_path)
    st.success(f"✅ Model loaded: {selected_model_name}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Get feature columns
target_col = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
feature_cols = [c for c in df.columns if c not in {CONFIG.target_column, CONFIG.arrhythmia_column}]

st.markdown("---")

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions"])

with tab1:
    st.subheader("Single Row Prediction")
    
    row_index = st.number_input(
        "Select Row Index",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        help="Choose a row from the dataset to predict"
    )
    
    X_row = df.loc[[row_index], feature_cols].copy()
    
    # Display original values
    with st.expander("📋 Original Row Values"):
        st.dataframe(X_row.T, use_container_width=True)
    
    # What-if analysis for numeric features
    st.subheader("What-If Analysis")
    st.caption("Adjust numeric features to see how predictions change")
    
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if numeric_cols:
        with st.form("whatif_form"):
            cols_per_row = 2
            for i in range(0, len(numeric_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                    with cols[j]:
                        # Get current value
                        raw_val = df.loc[row_index, col_name]
                        val = float(raw_val) if pd.notna(raw_val) else 0.0
                        
                        # Compute robust bounds
                        s = pd.to_numeric(df[col_name], errors="coerce")
                        s = s[np.isfinite(s)]
                        
                        if len(s) == 0:
                            low, high = val - 1.0, val + 1.0
                        else:
                            try:
                                low = float(np.nanpercentile(s, 5))
                                high = float(np.nanpercentile(s, 95))
                            except Exception:
                                low, high = float(s.min()), float(s.max())
                        
                        # Ensure finite bounds
                        if not np.isfinite(low):
                            low = val - 1.0
                        if not np.isfinite(high):
                            high = val + 1.0
                        
                        # Ensure valid range
                        if not (high > low):
                            center = val if np.isfinite(val) else (float(s.median()) if len(s) else 0.0)
                            pad = abs(center) * 0.1 + 1.0
                            low, high = center - pad, center + pad
                        
                        # Clamp value
                        if val < low:
                            val = low
                        elif val > high:
                            val = high
                        
                        X_row.loc[row_index, col_name] = st.slider(
                            col_name,
                            min_value=low,
                            max_value=high,
                            value=val,
                            key=f"slider_{col_name}"
                        )
            
            predict_btn = st.form_submit_button("🎯 Predict", use_container_width=True)
    else:
        predict_btn = st.button("🎯 Predict", use_container_width=True)
    
    # Make prediction
    if predict_btn or not numeric_cols:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_row)[:, 1][0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Probability", f"{prob:.4f}")
                with col2:
                    risk_level = "High" if prob > 0.5 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Visual probability bar
                st.progress(prob)
            else:
                pred = model.predict(X_row)[0]
                st.metric("Prediction", str(pred))
        
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

with tab2:
    st.subheader("Batch Predictions")
    st.caption("Upload a CSV file for batch predictions")
    
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Try multiple encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            upload_df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    upload_df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except (UnicodeDecodeError, LookupError) as e:
                    last_error = e
                    continue
            
            if upload_df is None:
                raise RuntimeError(
                    f"No se pudo decodificar el CSV. Error: {last_error}"
                )
            
            st.success(f"✅ Uploaded {len(upload_df)} rows")
            
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(upload_df.head(10), use_container_width=True)
            
            if st.button("🎯 Predict All Rows", type="primary"):
                with st.spinner("Making predictions..."):
                    # Make predictions
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(upload_df)[:, 1]
                        output_df = upload_df.copy()
                        output_df["prediction_probability"] = probs
                        output_df["prediction_class"] = (probs > 0.5).astype(int)
                    else:
                        preds = model.predict(upload_df)
                        output_df = upload_df.copy()
                        output_df["prediction"] = preds
                    
                    st.success(f"✅ Predictions completed for {len(output_df)} rows")
                    
                    # Display results
                    st.dataframe(output_df, use_container_width=True)
                    
                    # Download button
                    csv_data = output_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv_data,
                        file_name=f"predictions_{task}_{selected_model_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")
            st.exception(e)
