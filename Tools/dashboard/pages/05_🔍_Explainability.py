"""Explainability and SHAP Analysis page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import streamlit as st

from app import (
    initialize_state,
    list_saved_models,
    load_data,
    sidebar_data_controls,
)
from src.config import CONFIG

# Initialize
initialize_state()

# Page config
st.title("🔍 Model Explainability")
st.markdown("---")

# Sidebar controls
data_path, task = sidebar_data_controls()

if not data_path:
    st.warning("⚠️ Please provide a dataset path in the sidebar")
    st.stop()

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
    help="Select a trained model for explainability analysis"
)

model_path = saved_models[selected_model_name]

# SHAP settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ SHAP Settings")

n_samples = st.sidebar.slider(
    "Number of samples",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Number of samples to use for SHAP analysis"
)

# Load model and data
try:
    model = joblib.load(model_path)
    st.success(f"✅ Model loaded: {selected_model_name}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Get feature columns
target_col = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
feature_cols = [c for c in df.columns if c not in {CONFIG.target_column, CONFIG.arrhythmia_column}]

# Sample data for SHAP
sample_df = df[feature_cols].iloc[:n_samples]

st.markdown("---")

# Check if SHAP is available
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    st.error("❌ SHAP not installed. Please install it with: `pip install shap`")
    st.stop()

# SHAP analysis
st.subheader("SHAP Analysis")
st.caption(f"Using {len(sample_df)} samples for explainability")

if st.button("🚀 Compute SHAP Values", type="primary", use_container_width=True):
    try:
        with st.spinner("Computing SHAP values... This may take a moment"):
            # Create explainer
            if hasattr(model, "predict_proba"):
                explainer = shap.Explainer(model.predict_proba, sample_df)
            else:
                explainer = shap.Explainer(model.predict, sample_df)
            
            # Compute SHAP values
            shap_values = explainer(sample_df)
            
            # Store in session state
            st.session_state.shap_explainer = explainer
            st.session_state.shap_values = shap_values
            
            st.success("✅ SHAP values computed successfully!")
    
    except Exception as e:
        st.error(f"❌ Error computing SHAP values: {e}")
        st.exception(e)

# Display SHAP plots if available
if "shap_values" in st.session_state and st.session_state.shap_values is not None:
    shap_values = st.session_state.shap_values
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Beeswarm Plot",
        "📈 Bar Plot",
        "🎯 Force Plot",
        "🌊 Waterfall Plot"
    ])
    
    with tab1:
        st.markdown("### Beeswarm Plot")
        st.caption("Shows the distribution of SHAP values for each feature across all samples")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        except Exception as e:
            st.error(f"Error creating beeswarm plot: {e}")
    
    with tab2:
        st.markdown("### Feature Importance (Bar Plot)")
        st.caption("Mean absolute SHAP value for each feature")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        except Exception as e:
            st.error(f"Error creating bar plot: {e}")
    
    with tab3:
        st.markdown("### Force Plot")
        st.caption("Visualize predictions for individual samples")
        
        sample_idx = st.slider(
            "Select sample index",
            min_value=0,
            max_value=len(shap_values) - 1,
            value=0
        )
        
        try:
            # Force plot for single prediction
            st.pyplot(
                shap.plots.force(
                    shap_values[sample_idx],
                    matplotlib=True,
                    show=False
                )
            )
        
        except Exception as e:
            st.error(f"Error creating force plot: {e}")
            # Try alternative visualization
            try:
                st.write("SHAP values for this sample:")
                import pandas as pd
                shap_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "SHAP Value": shap_values.values[sample_idx]
                }).sort_values("SHAP Value", key=abs, ascending=False)
                st.dataframe(shap_df, use_container_width=True)
            except Exception:
                pass
    
    with tab4:
        st.markdown("### Waterfall Plot")
        st.caption("Shows how each feature contributes to push the model output from the base value")
        
        sample_idx = st.slider(
            "Select sample index",
            min_value=0,
            max_value=len(shap_values) - 1,
            value=0,
            key="waterfall_slider"
        )
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_values[sample_idx], show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        except Exception as e:
            st.error(f"Error creating waterfall plot: {e}")
    
    # Additional analysis
    st.markdown("---")
    
    with st.expander("🔢 Feature Importance Rankings"):
        try:
            import pandas as pd
            import numpy as np
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Mean |SHAP|": mean_shap
            }).sort_values("Mean |SHAP|", ascending=False)
            
            st.dataframe(
                importance_df.style.format({"Mean |SHAP|": "{:.6f}"}),
                use_container_width=True,
                hide_index=True
            )
        
        except Exception as e:
            st.error(f"Error computing feature importance: {e}")
    
    with st.expander("ℹ️ About SHAP"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**
        
        SHAP values explain the contribution of each feature to individual predictions:
        
        - **Positive SHAP value**: Feature pushes prediction higher
        - **Negative SHAP value**: Feature pushes prediction lower
        - **Magnitude**: Importance of the feature for that prediction
        
        **Visualizations:**
        - **Beeswarm**: Overall feature importance and value distribution
        - **Bar**: Simple feature importance ranking
        - **Force**: Individual prediction explanation
        - **Waterfall**: Step-by-step contribution breakdown
        
        SHAP values are based on game theory and provide consistent, locally accurate explanations.
        """)

else:
    st.info("👆 Click 'Compute SHAP Values' to generate explainability visualizations")

