"""Hyperparameter tuning utilities."""
from __future__ import annotations

from typing import Dict

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def randomized_search(
    pipeline,
    param_distributions: Dict,
    X,
    y,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
):
    """Perform randomized hyperparameter search.
    
    Args:
        pipeline: Sklearn pipeline to tune
        param_distributions: Parameter distributions
        X: Training features
        y: Training labels
        n_iter: Number of iterations
        cv: Number of CV folds
        scoring: Scoring metric
        random_state: Random seed
        
    Returns:
        Fitted RandomizedSearchCV object
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    
    search.fit(X, y)
    
    return search


def grid_search(
    pipeline,
    param_grid: Dict,
    X,
    y,
    cv: int = 3,
    scoring: str = "roc_auc",
):
    """Perform grid search for hyperparameter tuning.
    
    Args:
        pipeline: Sklearn pipeline to tune
        param_grid: Parameter grid
        X: Training features
        y: Training labels
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Fitted GridSearchCV object
    """
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    
    search.fit(X, y)
    
    return search
