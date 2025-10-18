"""Training script: nested CV, tuning, imbalance handling, and tracking."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple, List, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency guards
    SMOTE = None  # type: ignore
    ImbPipeline = None  # type: ignore

from .config import CONFIG, ProjectConfig, validate_config, RANDOM_SEED
from .data import load_dataset, train_test_split, select_feature_target
from .features import safe_feature_columns
from .preprocess import build_preprocess_pipelines
from .models import make_classifiers, make_regressors, make_kmeans

# Optional: MLflow
use_mlflow = os.environ.get("EXPERIMENT_TRACKER", "mlflow").lower() == "mlflow"
if use_mlflow:
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(os.environ.get("TRACKING_URI"))
    except Exception:
        use_mlflow = False


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def fit_and_save_best_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    quick: bool = False,
    task_name: str = "mortality",
    imputer_mode: str = "iterative",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, object]:
    """Train multiple classifiers using nested CV, pick best by outer AUROC, refit and save.

    Returns:
        (save_path, fitted_model)
    """
    # Preprocess builder (fitted to get feature names only; cloned in CV)
    pre, _ = build_preprocess_pipelines(X, imputer_mode=imputer_mode)
    # If columns were dropped internally, inform via progress callback
    try:
        # infer columns selected by ColumnTransformer
        selected = pre.transformers_[0][2] + pre.transformers_[1][2]
        dropped = [c for c in X.columns if c not in selected]
        if dropped and progress_callback:
            progress_callback(f"Columnas descartadas (vacías/constantes): {', '.join(map(str, dropped[:20]))}", 0.05)
    except Exception:
        pass
    models = make_classifiers()
    # Safety: drop NN model from grid entirely unless explicitly requested via env var
    if "nn" in models and os.environ.get("ENABLE_TORCH_MODEL", "0") != "1":
        models.pop("nn", None)

    outer_splits = 3 if not quick else 2
    inner_splits = 3 if not quick else 2
    inner_iter = 10 if not quick else 4

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)

    best_overall = {"name": None, "score": -np.inf, "params": None}

    if use_mlflow:
        try:
            mlflow.set_experiment(f"{task_name}_classification")
        except Exception:
            pass

    # Warm start: if a best model already exists for this task, reuse it
    save_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    if os.path.exists(save_path):
        if progress_callback:
            progress_callback("Modelo existente encontrado. Cargando sin reentrenar...", 1.0)
        model = joblib.load(save_path)
        return save_path, model

    total_models = len(models)
    model_index = 0
    for name, (clf, grid) in models.items():
        outer_scores: List[float] = []
        model_index += 1
        if progress_callback:
            progress_callback(f"Entrenando modelo: {name}", (model_index - 1) / max(1, total_models))
        for fold_idx, (tr_idx, va_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Build fresh pipeline per fold
            if ImbPipeline is not None and SMOTE is not None:
                pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=RANDOM_SEED)), ("clf", clf)])
            else:
                pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

            search = RandomizedSearchCV(
                pipe,
                param_distributions={**{f"clf__{k}": v for k, v in grid.items()}},
                n_iter=inner_iter,
                cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
                scoring="roc_auc",
                n_jobs=-1,
                random_state=RANDOM_SEED + fold_idx,
                refit=True,
                verbose=0,
            )
            search.fit(X_tr, y_tr)
            prob_va = search.best_estimator_.predict_proba(X_va)[:, 1]
            auroc = roc_auc_score(y_va, prob_va)
            outer_scores.append(auroc)
            if progress_callback:
                # progress within model based on fold completion
                frac = (model_index - 1 + (fold_idx + 1) / outer_splits) / max(1, total_models)
                progress_callback(f"{name}: fold {fold_idx+1}/{outer_splits} AUROC={auroc:.3f}", frac)

            if use_mlflow:
                try:
                    with mlflow.start_run(run_name=f"{name}_outer{fold_idx}"):
                        mlflow.log_metric("outer_auroc", auroc)
                        mlflow.log_params({k: v for k, v in search.best_params_.items()})
                except Exception:
                    pass

        mean_outer = float(np.mean(outer_scores)) if outer_scores else -np.inf

        if mean_outer > best_overall["score"]:
            best_overall.update({"name": name, "score": mean_outer, "params": None})

    if best_overall["name"] is None:
        raise RuntimeError("No classifier was successfully evaluated.")

    # Refit best model on full training data with a slightly larger search
    best_name = best_overall["name"]
    clf, grid = models[best_name]
    if ImbPipeline is not None and SMOTE is not None:
        final_pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=RANDOM_SEED)), ("clf", clf)])
    else:
        final_pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    final_search = RandomizedSearchCV(
        final_pipe,
        param_distributions={**{f"clf__{k}": v for k, v in grid.items()}},
        n_iter=(15 if not quick else 6),
        cv=StratifiedKFold(n_splits=(3 if not quick else 2), shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
        verbose=0,
    )
    final_search.fit(X, y)
    if progress_callback:
        progress_callback("Refit final del mejor modelo", 1.0)

    best_model = final_search.best_estimator_
    save_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    joblib.dump(best_model, save_path)
    return save_path, best_model


def fit_and_save_selected_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: List[str],
    quick: bool = False,
    task_name: str = "mortality",
    imputer_mode: str = "iterative",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, str]:
    """Train only selected classifiers; save each refit model and a summary.

    Returns a dict model_name -> save_path. Also saves the best overall as best_classifier_{task}.joblib
    """
    pre, _ = build_preprocess_pipelines(X, imputer_mode=imputer_mode)
    all_models = make_classifiers()
    models = {k: v for k, v in all_models.items() if k in selected_models}
    if not models:
        raise ValueError("No hay modelos seleccionados para entrenar.")

    outer_splits = 3 if not quick else 2
    inner_splits = 3 if not quick else 2
    inner_iter = 10 if not quick else 4
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)

    model_scores: Dict[str, float] = {}
    total = len(models)
    for idx, (name, (clf, grid)) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(f"Evaluando (CV externa) modelo: {name}", (idx - 1) / max(1, total))
        scores: List[float] = []
        for fold_idx, (tr, va) in enumerate(outer_cv.split(X, y)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=RANDOM_SEED)), ("clf", clf)]) if (ImbPipeline and SMOTE) else Pipeline(steps=[("pre", pre), ("clf", clf)])
            search = RandomizedSearchCV(
                pipe,
                param_distributions={f"clf__{k}": v for k, v in grid.items()},
                n_iter=inner_iter,
                cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
                scoring="roc_auc",
                n_jobs=-1,
                random_state=RANDOM_SEED + fold_idx,
                refit=True,
                verbose=0,
            )
            search.fit(X_tr, y_tr)
            prob_va = search.best_estimator_.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, prob_va))
            if progress_callback:
                frac = (idx - 1 + (fold_idx + 1) / outer_splits) / max(1, total)
                progress_callback(f"{name}: fold {fold_idx+1}/{outer_splits} AUROC={scores[-1]:.3f}", frac)
        model_scores[name] = float(np.mean(scores)) if scores else -np.inf

    # Refit and save each selected model on full training data
    save_paths: Dict[str, str] = {}
    for idx, (name, (clf, grid)) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(f"Refit final y guardado: {name}", (idx - 1) / max(1, total))
        pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=RANDOM_SEED)), ("clf", clf)]) if (ImbPipeline and SMOTE) else Pipeline(steps=[("pre", pre), ("clf", clf)])
        final_search = RandomizedSearchCV(
            pipe,
            param_distributions={f"clf__{k}": v for k, v in grid.items()},
            n_iter=(15 if not quick else 6),
            cv=StratifiedKFold(n_splits=(3 if not quick else 2), shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            refit=True,
            verbose=0,
        )
        final_search.fit(X, y)
        model = final_search.best_estimator_
        path = os.path.join(MODELS_DIR, f"model_{task_name}_{name}.joblib")
        joblib.dump(model, path)
        save_paths[name] = path

    # Save summary CSV
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pd.DataFrame([{"model": k, "outer_mean_auroc": v} for k, v in model_scores.items()]).to_csv(
        os.path.join(reports_dir, f"training_summary_{task_name}.csv"), index=False
    )

    # Also save best-of-selected as legacy best_classifier_{task}.joblib
    best_name = max(model_scores.items(), key=lambda kv: kv[1])[0]
    best_model = joblib.load(save_paths[best_name])
    best_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    joblib.dump(best_model, best_path)
    if progress_callback:
        progress_callback(f"Mejor modelo: {best_name}", 1.0)
    return save_paths


def train_main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Train models for mortality/arrhythmia/regression.")
    parser.add_argument("--data", type=str, default=os.environ.get("DATASET_PATH"), help="Path to dataset file")
    parser.add_argument("--task", type=str, choices=["mortality", "arrhythmia", "regression"], default="mortality")
    parser.add_argument("--quick", action="store_true", help="Run quick mode for debugging")
    args = parser.parse_args(argv)

    if not args.data:
        raise ValueError("DATASET_PATH not provided. Use --data or set env var.")

    df = load_dataset(args.data)

    # Choose target
    if args.task == "mortality":
        target = CONFIG.target_column
    elif args.task == "arrhythmia":
        target = CONFIG.arrhythmia_column or "ventricular_arrhythmia"
    else:
        target = CONFIG.regression_target
        if not target or target not in df.columns:
            print("No regression target available; skipping.")
            return

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    # Select features
    feat_cols = safe_feature_columns(df, target_cols=[target])
    X = df[feat_cols]
    y = df[target]

    # Split train/test
    train_df, test_df = train_test_split(df, stratify_target=target)
    X_train, y_train = train_df[feat_cols], train_df[target]
    X_test, y_test = test_df[feat_cols], test_df[target]

    if args.task in {"mortality", "arrhythmia"}:
        path, model = fit_and_save_best_classifier(X_train, y_train, quick=args.quick, task_name=args.task)
        # Save test set for later evaluation
        test_path = os.path.join(MODELS_DIR, f"testset_{args.task}.parquet")
        test_df.to_parquet(test_path)
        print(f"Saved best model to {path} and test set to {test_path}")
    else:
        # Regression baseline
        from sklearn.metrics import mean_squared_error
        from .models import make_regressors
        pre, _ = build_preprocess_pipelines(X_train)
        name, (reg, grid) = next(iter(make_regressors().items()))
        pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        save_path = os.path.join(MODELS_DIR, f"baseline_regression.joblib")
        joblib.dump(pipe, save_path)
        print(f"Regression baseline RMSE={rmse:.4f}; saved to {save_path}")


if __name__ == "__main__":
    train_main()
