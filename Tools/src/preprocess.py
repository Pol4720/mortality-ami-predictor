"""Preprocessing pipelines for numeric and categorical data with optional selection."""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor


def build_preprocess_pipelines(
    X: pd.DataFrame,
    imputer_mode: str = "iterative",  # 'iterative' | 'knn' | 'simple'
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

    if imputer_mode == "iterative":
        # Use an estimator that accepts NaNs natively to avoid BayesianRidge errors
        num_imputer = IterativeImputer(
            estimator=HistGradientBoostingRegressor(random_state=42),
            random_state=42,
            sample_posterior=False,
            max_iter=10,
            initial_strategy="median",
        )
    elif imputer_mode == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif imputer_mode == "simple":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError("imputer_mode debe ser 'iterative', 'knn' o 'simple'")

    steps_num = [("imputer", num_imputer)]
    if scale_numeric and num_cols:
        steps_num.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    num_pipe = Pipeline(steps=steps_num)

    if categorical_encoding == "onehot":
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    else:
        raise ValueError("Only onehot encoding is implemented in this scaffold.")

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps = [("pre", pre)]

    if feature_selection:
        est = selection_estimator or RandomForestClassifier(n_estimators=200, random_state=42)
        steps.append(("select", SelectFromModel(estimator=est, threshold="median")))

    pipe = Pipeline(steps=steps)

    # Do NOT fit here to avoid expensive work before CV; return approximate names
    feature_names = num_cols + cat_cols
    return pipe, feature_names
