"""Cross-validation strategies for model evaluation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ..config import RANDOM_SEED
from ..preprocessing import build_preprocessing_pipeline, PreprocessingConfig

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None


def nested_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    quick: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Dict]:
    """Perform nested cross-validation to evaluate models.
    
    Uses outer CV for evaluation and inner CV for hyperparameter tuning.
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of {name: (model, param_grid)}
        preprocessing_config: Preprocessing configuration
        quick: If True, use reduced splits
        progress_callback: Optional callback for progress
        
    Returns:
        Dictionary mapping model_name -> {'mean_score': float, 'std_score': float, 'fold_scores': List[float]}
    """
    outer_splits = 3 if not quick else 2
    inner_splits = 3 if not quick else 2
    inner_iter = 10 if not quick else 4
    
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)
    
    results = {}
    total_models = len(models)
    
    for model_idx, (name, (model, grid)) in enumerate(models.items(), 1):
        fold_scores = []
        
        if progress_callback:
            progress_callback(f"Evaluating {name}...", (model_idx - 1) / total_models)
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Build preprocessing pipeline
            preprocess_pipeline, _ = build_preprocessing_pipeline(X_train, preprocessing_config)
            
            # Extract the preprocessor from the pipeline (to avoid nested pipelines)
            # build_preprocessing_pipeline returns Pipeline([("preprocessor", ColumnTransformer)])
            if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
                preprocess = preprocess_pipeline.steps[0][1]  # Get the ColumnTransformer
            else:
                preprocess = preprocess_pipeline
            
            # Build full pipeline with SMOTE if available
            if ImbPipeline and SMOTE:
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    ("classifier", model),
                ])
            else:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("classifier", model),
                ])
            
            # Inner CV for hyperparameter tuning
            param_grid = {f"classifier__{k}": v for k, v in grid.items()}
            
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=inner_iter,
                cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
                scoring="roc_auc",
                n_jobs=-1,
                random_state=RANDOM_SEED + fold_idx,
                refit=True,
            )
            
            search.fit(X_train, y_train)
            
            # Evaluate on validation fold
            y_pred_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
            auroc = roc_auc_score(y_val, y_pred_proba)
            fold_scores.append(auroc)
            
            if progress_callback:
                frac = (model_idx - 1 + (fold_idx + 1) / outer_splits) / total_models
                progress_callback(f"{name}: fold {fold_idx+1}/{outer_splits} AUROC={auroc:.3f}", frac)
        
        results[name] = {
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        }
    
    return results
