"""Classification models for ML experiments.

This module provides factory functions for creating classification models
with predefined hyperparameter grids.
"""
from __future__ import annotations

from typing import Dict, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import inspect

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:
    LGBMClassifier = None  # type: ignore


def make_classifiers() -> Dict[str, Tuple[object, Dict]]:
    """Create dictionary of classifiers and their parameter grids.
    
    Returns:
        Dictionary mapping model_name -> (model, param_grid)
        where param_grid is for RandomizedSearchCV
    """
    models: Dict[str, Tuple[object, Dict]] = {}
    
    # K-Nearest Neighbors
    models["knn"] = (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
    )
    
    # Calibrated Logistic Regression
    logreg = LogisticRegression(
        max_iter=500,
        tol=1e-3,
        solver="lbfgs",
        class_weight="balanced"
    )
    
    # Handle sklearn API changes (estimator vs base_estimator)
    sig = inspect.signature(CalibratedClassifierCV)
    if "estimator" in sig.parameters:
        calib = CalibratedClassifierCV(estimator=logreg, method="sigmoid", cv=2)
        grid = {"estimator__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        calib = CalibratedClassifierCV(base_estimator=logreg, method="sigmoid", cv=2)
        grid = {"base_estimator__C": [0.01, 0.1, 1.0, 10.0]}
    
    models["logreg"] = (calib, grid)
    
    # Decision Tree
    models["dtree"] = (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        },
    )
    
    # XGBoost (if available)
    if XGBClassifier is not None:
        models["xgb"] = (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",
                scale_pos_weight=1.0,
                n_jobs=-1,
            ),
            {
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.5],
            },
        )

        # Balanced XGBoost for imbalanced datasets (High Recall)
        # Ratio calculation: 2819 (negative) / 274 (positive) ~= 10.28
        models["xgb_balanced"] = (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,        # Reduced to prevent overfitting
                colsample_bytree=0.8, # Reduced to prevent overfitting
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
            ),
            {
                "max_depth": [3, 4, 5],          # Lower depth to prevent overfitting
                "min_child_weight": [3, 5, 7],   # Higher weight to be more conservative
                "gamma": [0.1, 0.5, 1.0],        # Minimum loss reduction required
                # Adjusted for specific ratio ~10.3. Range covers under/over-sampling effects
                "scale_pos_weight": [8, 10, 10.3, 12, 15], 
                "reg_alpha": [0, 0.1, 1.0],      # L1 regularization
                "reg_lambda": [1, 1.5, 2.0],     # L2 regularization
            },
        )
    
    # LightGBM (if available)
    if LGBMClassifier is not None:
        models["lgbm"] = (
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                min_data_in_leaf=20,
                min_split_gain=0.0,
            ),
            {
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 6, 10],
            },
        )
    
    return models
