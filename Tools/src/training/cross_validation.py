"""Cross-validation strategies for model evaluation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

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


def rigorous_repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    n_splits: int = 5,
    n_repeats: int = 6,
    scoring: str = "roc_auc",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Dict]:
    """Perform rigorous repeated stratified k-fold cross-validation.
    
    This implements the rigorous experimental protocol with 90+ runs:
    - Repeated Stratified K-Fold (e.g., 10 splits × 10 repeats = 100 runs)
    - For each model, estimate μ (mean) and σ (std) across all runs
    - No hyperparameter tuning (use default or pre-tuned parameters)
    
    This is the TRAIN & VALIDATION phase from the experimental pipeline.
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of {name: (model, param_dict)} where param_dict contains
                fixed hyperparameters (not grid search)
        preprocessing_config: Preprocessing configuration
        n_splits: Number of folds in each repetition
        n_repeats: Number of repetitions
        scoring: Scoring metric
        progress_callback: Optional callback for progress
        
    Returns:
        Dictionary mapping model_name -> {
            'mean_score': float,  # μ
            'std_score': float,   # σ
            'all_scores': List[float],  # All 90+ scores
            'best_params': dict  # Best hyperparameters found
        }
    """
    # Create repeated stratified k-fold
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED
    )
    
    total_iterations = n_splits * n_repeats
    results = {}
    total_models = len(models)
    
    for model_idx, (name, (base_model, param_grid)) in enumerate(models.items(), 1):
        all_scores = []
        
        if progress_callback:
            progress_callback(f"🔄 Entrenando modelo '{name}' con {total_iterations} corridas (validación cruzada estratificada)...", (model_idx - 1) / total_models)
        
        for run_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Build preprocessing pipeline
            preprocess_pipeline, _ = build_preprocessing_pipeline(X_train, preprocessing_config)
            
            # Extract the preprocessor
            if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
                preprocess = preprocess_pipeline.steps[0][1]
            else:
                preprocess = preprocess_pipeline
            
            # Build full pipeline with SMOTE if available
            if ImbPipeline and SMOTE:
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    ("classifier", clone(base_model)),
                ])
            else:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("classifier", clone(base_model)),
                ])
            
            # Set hyperparameters if provided
            if param_grid:
                # Set fixed parameters (not a grid search)
                classifier_params = {f"classifier__{k}": v for k, v in param_grid.items()}
                pipeline.set_params(**classifier_params)
            
            # Train and evaluate
            try:
                pipeline.fit(X_train, y_train)
                if hasattr(pipeline, "predict_proba"):
                    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                else:
                    y_pred_proba = pipeline.decision_function(X_val)
                score = roc_auc_score(y_val, y_pred_proba)
                all_scores.append(score)
                # Show progress every run
                if progress_callback:
                    frac = (model_idx - 1 + run_idx / total_iterations) / total_models
                    msg = (
                        f"Modelo: {name}\n"
                        f"Corrida: {run_idx}/{total_iterations}\n"
                        f"AUROC actual: {score:.3f}\n"
                        f"Promedio hasta ahora: μ={np.mean(all_scores):.3f}, σ={np.std(all_scores):.3f}"
                    )
                    progress_callback(msg, frac)
            except Exception as e:
                print(f"Warning: Error in {name} run {run_idx}: {e}")
                continue
        
        # Calculate final statistics
        results[name] = {
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores, ddof=1)),  # Sample std
            "all_scores": all_scores,
            "n_runs": len(all_scores),
            "best_params": param_grid if param_grid else {},
        }
        
        if progress_callback:
            progress_callback(
                f"✅ Modelo '{name}' completado: μ={results[name]['mean_score']:.4f}, σ={results[name]['std_score']:.4f} ({len(all_scores)} corridas)",
                model_idx / total_models
            )
    
    return results

