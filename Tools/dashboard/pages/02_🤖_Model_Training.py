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
    sidebar_data_controls,
    sidebar_training_controls,
    train_models_with_progress,
)

# Initialize
initialize_state()

# Page config
st.title("🤖 Model Training")
st.markdown("---")

# Sidebar controls
data_path, task = sidebar_data_controls()

if not data_path:
    st.warning("⚠️ Please provide a dataset path in the sidebar")
    st.stop()

# Training settings
st.sidebar.markdown("---")
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
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
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
    **Quick Mode:**
    - Uses simplified hyperparameter search
    - Faster iteration for debugging
    - Recommended for initial exploration
    
    **Imputation Strategies:**
    - **Iterative**: Uses sklearn's IterativeImputer (MICE)
    - **KNN**: K-Nearest Neighbors imputation
    - **Simple**: Mean/median/mode imputation
    
    **Model Types:**
    - Decision Trees, Random Forest, XGBoost
    - Logistic Regression, SVM
    - KNN, Naive Bayes
    """)
