"""Preprocessing pipelines for numeric and categorical data with optional selection."""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def build_preprocess_pipelines(
    X: pd.DataFrame,
    use_iterative_imputer: bool = True,
    scale_numeric: bool = True,
    categorical_encoding: str = "onehot",
    feature_selection: bool = False,
    selection_estimator: Optional[object] = None,
) -> Tuple[Pipeline, List[str]]:
    """Build a ColumnTransformer-based preprocessing pipeline.

    Args:
        X: Input feature DataFrame used to infer types
        use_iterative_imputer: If True, use IterativeImputer, else KNNImputer
        scale_numeric: Whether to standardize numeric features
        categorical_encoding: One of {"onehot"}
        feature_selection: Whether to include SelectFromModel step
        selection_estimator: Base estimator for selection (defaults to RandomForestClassifier)

    Returns:
        (pipeline, output_feature_names)
    """

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if use_iterative_imputer:
        num_imputer = IterativeImputer(random_state=42, sample_posterior=False, max_iter=10)
    else:
        num_imputer = KNNImputer(n_neighbors=5)

    steps_num = [("imputer", num_imputer)]
    if scale_numeric and num_cols:
        steps_num.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    num_pipe = Pipeline(steps=steps_num)

    if categorical_encoding == "onehot":
        cat_pipe = Pipeline(
            steps=[
                ("imputer", "drop"),  # let OneHotEncoder handle unknowns; we keep NaN as category
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )
    else:
        raise ValueError("Only onehot encoding is implemented in this scaffold.")

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps = [("pre", pre)]

    if feature_selection:
        est = selection_estimator or RandomForestClassifier(n_estimators=200, random_state=42)
        steps.append(("select", SelectFromModel(estimator=est, threshold="median")))

    pipe = Pipeline(steps=steps)

    # Fit to get feature names
    pipe.fit(X)
    # Feature names after OHE
    try:
        feature_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(len(num_cols) + len(cat_cols))]

    return pipe, feature_names
