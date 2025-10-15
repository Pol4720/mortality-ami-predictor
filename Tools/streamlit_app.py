from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.data import load_dataset, summarize_dataframe

st.set_page_config(page_title="AMI Mortality/Arrhythmia Dashboard", layout="wide")


def load_model(model_path: str):
    return joblib.load(model_path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=str, default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--model", type=str, default=os.path.join("models", "best_classifier_mortality.joblib"))
    args, _ = parser.parse_known_args()

    st.title("In-Hospital Mortality / Ventricular Arrhythmia Predictor")

    data_path = st.text_input("Dataset path (DATASET_PATH)", value=args.data or "")
    model_path = st.text_input("Model path", value=args.model)

    if not data_path:
        st.warning("Please provide DATASET_PATH")
        st.stop()

    df = load_dataset(data_path)
    st.subheader("Dataset overview")
    st.dataframe(df.head())

    # Summary tabs
    tabs = st.tabs(["Summary", "Predict", "Explainability", "Compare Models"])

    with tabs[0]:
        summaries = summarize_dataframe(df, CONFIG)
        st.write("Shape:", df.shape)
        st.write("Missingness:")
        st.dataframe(summaries["missing"].head(30))
        if CONFIG.target_column in df.columns:
            st.write("Class balance:")
            st.dataframe(summaries.get(f"class_balance_{CONFIG.target_column}"))

    with tabs[1]:
        st.write("Run predictions on a selected row or uploaded CSV")
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            st.error("Model not found. Train the models first.")
            st.stop()

        # Select a row
        row_index = st.number_input("Row index", min_value=0, max_value=max(0, len(df) - 1), value=0)
        cols = [c for c in df.columns if c not in {CONFIG.target_column, CONFIG.arrhythmia_column}]
        X_row = df.loc[[row_index], cols]
        # What-if sliders for numeric columns
        with st.expander("What-if analysis (numeric features)"):
            for c in [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]:
                val = float(df.loc[row_index, c]) if pd.notna(df.loc[row_index, c]) else 0.0
                low = float(np.nanpercentile(df[c], 5)) if pd.api.types.is_numeric_dtype(df[c]) else val
                high = float(np.nanpercentile(df[c], 95)) if pd.api.types.is_numeric_dtype(df[c]) else val
                X_row.loc[row_index, c] = st.slider(c, min_value=low, max_value=high, value=val)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_row)[:, 1][0]
            st.metric("Predicted probability", f"{prob:.3f}")
        else:
            pred = model.predict(X_row)[0]
            st.metric("Prediction", str(pred))

        uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"]) 
        if uploaded is not None:
            up_df = pd.read_csv(uploaded)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(up_df)[:, 1]
                out = up_df.copy(); out["prediction_proba"] = probs
            else:
                preds = model.predict(up_df)
                out = up_df.copy(); out["prediction"] = preds
            st.download_button("Download predictions", data=out.to_csv(index=False), file_name="predictions.csv")

    with tabs[2]:
        st.write("Global and local explanations will be rendered using SHAP if available.")
        try:
            import importlib
            shap = importlib.import_module("shap")
            explainer = shap.Explainer(model.predict_proba, df[cols].iloc[:200])
            sv = explainer(df[cols].iloc[:200])
            st.pyplot(shap.plots.beeswarm(sv, show=False))
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")

    with tabs[3]:
        st.write("Compare models (placeholder). If MLflow is used, you can load runs and display a metrics table here.")


if __name__ == "__main__":
    main()
