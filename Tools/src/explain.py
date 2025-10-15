"""Explainability utilities: SHAP global and local explanations, PDP."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def shap_global_summary(model, X: pd.DataFrame, name: str) -> str:
    import importlib
    shap = importlib.import_module("shap")
    try:
        explainer = shap.Explainer(model.predict_proba, X, algorithm="auto")
        shap_values = explainer(X)
        shap.plots.beeswarm(shap_values, show=False)
    except Exception:
        # Fallback: use TreeExplainer if it is a tree model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
    path = os.path.join(FIG_DIR, f"shap_summary_{name}.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def shap_local_force(model, X: pd.DataFrame, row_index: int, name: str) -> Optional[str]:
    import importlib
    shap = importlib.import_module("shap")
    try:
        explainer = shap.Explainer(model.predict_proba, X)
        sv = explainer(X.iloc[[row_index]])
        path = os.path.join(FIG_DIR, f"shap_force_{name}_{row_index}.html")
        shap.save_html(path, shap.plots.force(sv[0], matplotlib=False))
        return path
    except Exception:
        return None


essential_features_cache: Optional[List[str]] = None

def partial_dependence_plot(model, X: pd.DataFrame, features: List[str], name: str) -> str:
    disp = partial_dependence(model, X, features=features, kind="average")
    fig = disp.plot()
    path = os.path.join(FIG_DIR, f"pdp_{name}.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def permutation_importance_plot(model, X: pd.DataFrame, y: pd.Series, name: str) -> str:
    """Compute and save permutation importance bar plot."""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)[:30]
    plt.figure(figsize=(8, 6))
    importances[::-1].plot(kind="barh")
    plt.title("Permutation importance (top 30)")
    path = os.path.join(FIG_DIR, f"perm_importance_{name}.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path
