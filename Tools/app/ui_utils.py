from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from src.data_load import load_dataset, summarize_dataframe, data_audit
from src.config import CONFIG
from src.training import fit_and_save_best_classifier, fit_and_save_selected_classifiers
from src.evaluation import evaluate_main
from pathlib import Path


def sidebar_controls():
    st.sidebar.header("Configuración")
    # Show logo if present
    try:
        from pathlib import Path
        assets_dir = Path(__file__).parent / "assets"
        for pat in ("logo.png", "logo.jpg", "logo.jpeg", "logo.ico"):
            cand = assets_dir / pat
            if cand.exists():
                st.sidebar.image(str(cand), use_container_width=True)
                break
    except Exception:
        pass
    # Default dataset path: relative from current working directory to the specified Excel file
    try:
        default_abs = r"C:\Users\HP\Desktop\ML\Proyecto\mortality-ami-predictor\DATA\recuima-020425.xlsx"
        default_rel = os.path.relpath(default_abs, start=os.getcwd()) if os.path.isabs(default_abs) else default_abs
    except Exception:
        default_rel = "DATA\\recuima-020425.xlsx"
    data_path = st.sidebar.text_input("DATASET_PATH", value=os.environ.get("DATASET_PATH", default_rel))
    task = st.sidebar.selectbox("Tarea", ["mortality", "arrhythmia"], index=0)
    quick = st.sidebar.checkbox("Modo rápido (debug)", value=True)
    imputer_mode = st.sidebar.selectbox("Imputación", ["iterative", "knn", "simple"], index=0)
    # Model selection
    from src.models import make_classifiers
    model_keys = list(make_classifiers().keys())
    selected_models = st.sidebar.multiselect("Modelos a entrenar/evaluar", model_keys, default=model_keys)
    return data_path, task, quick, imputer_mode, selected_models


def run_training(data_path: str, task: str, quick: bool, imputer_mode: str, selected_models: list[str]) -> str:
    # Custom training with progress callback
    df = load_dataset(data_path)
    from src.features import safe_feature_columns
    from src.data_load import train_test_split
    if task == "mortality":
        target = CONFIG.target_column
    else:
        target = CONFIG.arrhythmia_column
    # Keep both train and test to persist test set for evaluation
    train_df, test_df = train_test_split(df, stratify_target=target)
    X = train_df[safe_feature_columns(train_df, [target])]
    y = train_df[target]

    progress = st.progress(0.0)
    status = st.empty()

    def cb(msg: str, frac: float):
        status.info(msg)
        progress.progress(min(max(frac, 0.0), 1.0))

    if selected_models:
        save_paths = fit_and_save_selected_classifiers(
            X, y,
            selected_models=selected_models,
            quick=quick,
            task_name=task,
            imputer_mode=imputer_mode,
            progress_callback=cb,
        )
        status.success("Modelos guardados: " + ", ".join([f"{k}: {v}" for k, v in save_paths.items()]))
    else:
        path, _ = fit_and_save_best_classifier(X, y, quick=quick, task_name=task, imputer_mode=imputer_mode, progress_callback=cb)
        status.success(f"Modelo guardado en {path}")
    progress.progress(1.0)
    # Persist test set so evaluate_main can load it
    try:
        models_dir = Path(__file__).parents[1] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        test_path = models_dir / f"testset_{task}.parquet"
        test_df.to_parquet(test_path)
    except Exception as e:
        status.warning(f"No se pudo guardar el test set: {e}")
    return f"Entrenamiento completado para {task}."


def run_evaluation(data_path: str, task: str) -> str:
    with st.spinner("Evaluando modelo en test hold-out..."):
        evaluate_main(["--data", data_path, "--task", task])
    return f"Evaluación completada para {task}."


def list_saved_models(task: str) -> dict[str, str]:
    models_dir = Path(__file__).parents[1] / "models"
    mapping = {}
    if models_dir.exists():
        for p in models_dir.glob(f"model_{task}_*.joblib"):
            name = p.stem.replace(f"model_{task}_", "")
            mapping[name] = str(p)
        best = models_dir / f"best_classifier_{task}.joblib"
        if best.exists():
            mapping["best"] = str(best)
    return mapping


def show_dataset_overview(df: pd.DataFrame):
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())
    summaries = summarize_dataframe(df)
    with st.expander("Missingness"):
        st.dataframe(summaries["missing"].head(50))
    with st.expander("Describe"):
        st.dataframe(summaries["describe"].head(50))
    with st.expander("Dtypes"):
        st.dataframe(summaries["dtypes"])
    # Early audit focused on features
    target_candidates = [CONFIG.target_column, CONFIG.arrhythmia_column]
    target = next((t for t in target_candidates if t and t in df.columns), None)
    feature_cols = df.columns if target is None else [c for c in df.columns if c != target]
    audit = data_audit(df, list(feature_cols))
    with st.expander("Auditoría temprana de datos (NaNs y constantes)"):
        st.write("Primeras filas de las features:")
        st.dataframe(audit["head"])
        st.write("Top columnas por fracción de NaN:")
        st.dataframe(audit["nan_summary"])
        if audit["full_missing"]:
            st.warning(f"Columnas completamente vacías: {', '.join(audit['full_missing'][:20])}")
        if audit["mostly_missing"]:
            st.info(f"Columnas con >80% NaN: {', '.join(audit['mostly_missing'][:20])}")
        if audit["constant"]:
            st.info(f"Columnas constantes o casi constantes: {', '.join(audit['constant'][:20])}")
