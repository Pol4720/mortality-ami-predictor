from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from app.state import get_state
from app.ui_utils import sidebar_controls, run_training, run_evaluation, show_dataset_overview
from src.config import CONFIG
from src.data import load_dataset

st.set_page_config(page_title="AMI Mortality/Arrhythmia Dashboard", layout="wide")


def load_model(model_path: str):
    return joblib.load(model_path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=str, default=os.environ.get("DATASET_PATH"))
    args, _ = parser.parse_known_args()

    st.title("In-Hospital Mortality / Ventricular Arrhythmia Predictor")

    # Sidebar controls
    data_path, task, quick, imputer_mode = sidebar_controls()
    if not data_path:
        data_path = args.data or ""
    if not data_path:
        st.warning("Proporciona DATASET_PATH en la barra lateral")
        st.stop()

    state = get_state()
    state["task"] = task
    model_path = state.get("model_path", f"models/best_classifier_{task}.joblib")

    # Load data and overview
    df = load_dataset(data_path)
    show_dataset_overview(df)

    # Actions row
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Entrenar modelos"):
            msg = run_training(data_path, task, quick, imputer_mode)
            state["last_train_msg"] = msg
            st.success(msg)
    with c2:
        if st.button("Evaluar en test hold-out"):
            msg = run_evaluation(data_path, task)
            st.success(msg)
    with c3:
        model_path = st.text_input("Ruta del modelo", value=model_path)
        state["model_path"] = model_path

    tabs = st.tabs(["Predicción", "Explicabilidad", "Comparación", "Métricas"])

    with tabs[0]:
        st.write("Predice en una fila seleccionada o sube un CSV")
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            st.info("Modelo no encontrado. Haz clic en 'Entrenar modelos' para generarlo.")
            st.stop()

        cols = [c for c in df.columns if c not in {CONFIG.target_column, CONFIG.arrhythmia_column}]
        row_index = st.number_input("Índice de fila", min_value=0, max_value=max(0, len(df) - 1), value=0)
        X_row = df.loc[[row_index], cols]
        with st.expander("What-if (numéricas)"):
            for c in [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]:
                val = float(df.loc[row_index, c]) if pd.notna(df.loc[row_index, c]) else 0.0
                low = float(np.nanpercentile(df[c], 5)) if pd.api.types.is_numeric_dtype(df[c]) else val
                high = float(np.nanpercentile(df[c], 95)) if pd.api.types.is_numeric_dtype(df[c]) else val
                X_row.loc[row_index, c] = st.slider(c, min_value=low, max_value=high, value=val)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_row)[:, 1][0]
            st.metric("Probabilidad predicha", f"{prob:.3f}")
        else:
            pred = model.predict(X_row)[0]
            st.metric("Predicción", str(pred))

        uploaded = st.file_uploader("Subir CSV para predicción por lotes", type=["csv"]) 
        if uploaded is not None:
            up_df = pd.read_csv(uploaded)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(up_df)[:, 1]
                out = up_df.copy(); out["prediction_proba"] = probs
            else:
                preds = model.predict(up_df)
                out = up_df.copy(); out["prediction"] = preds
            st.download_button("Descargar predicciones", data=out.to_csv(index=False), file_name="predicciones.csv")

    with tabs[1]:
        st.write("Explicabilidad global/local con SHAP (si está instalado)")
        try:
            import importlib
            shap = importlib.import_module("shap")
            explainer = shap.Explainer(model.predict_proba, df[cols].iloc[:200])
            sv = explainer(df[cols].iloc[:200])
            st.pyplot(shap.plots.beeswarm(sv, show=False))
        except Exception as e:
            st.info(f"SHAP no disponible: {e}")

    with tabs[2]:
        st.write("Comparación de modelos (pendiente): puedes integrar MLflow para listar y comparar runs.")

    with tabs[3]:
        st.subheader("Métricas del modelo")
        reports_dir = Path("reports")
        figures_dir = reports_dir / "figures"

        # Mostrar CSV de métricas finales si existe
        try:
            metrics_files = sorted(reports_dir.glob(f"final_metrics_{task}_*.csv"), key=lambda p: p.stat().st_mtime)
        except Exception:
            metrics_files = []
        if metrics_files:
            try:
                dfm = pd.read_csv(metrics_files[-1])
                st.dataframe(dfm)
            except Exception as e:
                st.warning(f"No se pudo leer el CSV de métricas: {e}")
        else:
            st.info("Aún no hay métricas guardadas. Ejecuta la evaluación.")

        cols = st.columns(3)
        # Cargar y mostrar figuras si existen
        def latest(glob_pattern: str):
            try:
                files = sorted(figures_dir.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
                return files[-1] if files else None
            except Exception:
                return None

        calib = latest(f"calibration_{task}_*.png")
        dcurve = latest(f"decision_curve_{task}_*.png")
        cmatrix = latest(f"confusion_{task}_*.png")
        with cols[0]:
            st.caption("Curva de calibración")
            if calib and calib.exists():
                st.image(str(calib), use_column_width=True)
            else:
                st.write("Sin imagen")
        with cols[1]:
            st.caption("Decision Curve")
            if dcurve and dcurve.exists():
                st.image(str(dcurve), use_column_width=True)
            else:
                st.write("Sin imagen")
        with cols[2]:
            st.caption("Matriz de confusión")
            if cmatrix and cmatrix.exists():
                st.image(str(cmatrix), use_column_width=True)
            else:
                st.write("Sin imagen")


if __name__ == "__main__":
    main()
