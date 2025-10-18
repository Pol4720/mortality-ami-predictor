from __future__ import annotations

import streamlit as st


def get_state():
    if "ami" not in st.session_state:
        st.session_state["ami"] = {
            "model_path": "models/best_classifier_mortality.joblib",
            "task": "mortality",
            "last_train_msg": "",
        }
    return st.session_state["ami"]
