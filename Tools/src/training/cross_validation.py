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
    checkpoint=None,  # NEW: checkpoint manager
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
        checkpoint: Optional TrainingCheckpoint for resuming
        
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
        # Check if model was already completed
        if checkpoint and checkpoint.should_skip_model(name):
            cached_results = checkpoint.get_model_results(name)
            if cached_results:
                results[name] = cached_results
                if progress_callback:
                    progress_callback(
                        f"✅ Modelo '{name}' cargado desde checkpoint (μ={cached_results['mean_score']:.4f})",
                        model_idx / total_models
                    )
                continue
        
        fold_scores = []
        completed_folds = checkpoint.get_completed_folds(name) if checkpoint else []
        
        if progress_callback:
            progress_callback(f"Evaluating {name}...", (model_idx - 1) / total_models)
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            # Skip already completed folds
            if fold_idx in completed_folds:
                # Load cached score
                if checkpoint:
                    cached_result = checkpoint.get_model_results(name)
                    if cached_result and 'fold_scores' in cached_result:
                        fold_scores.append(cached_result['fold_scores'][fold_idx])
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Build preprocessing pipeline
            preprocess_pipeline, _ = build_preprocessing_pipeline(X_train, preprocessing_config)
            
            # Extract the preprocessor from the pipeline (to avoid nested pipelines)
            if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
                preprocess = preprocess_pipeline.steps[0][1]
            else:
                preprocess = preprocess_pipeline
            
            # Build full pipeline with SMOTE if available
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    ("classifier", model),
                ])
            except ImportError:
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
            
            # Save fold checkpoint
            if checkpoint:
                checkpoint.save_fold_result(
                    model_name=name,
                    fold_idx=fold_idx,
                    score=auroc,
                    fitted_pipeline=search.best_estimator_,
                    metadata={
                        'train_indices': train_idx.tolist(),
                        'val_indices': val_idx.tolist(),
                        'best_params': search.best_params_
                    }
                )
            
            if progress_callback:
                frac = (model_idx - 1 + (fold_idx + 1) / outer_splits) / total_models
                progress_callback(f"{name}: fold {fold_idx+1}/{outer_splits} AUROC={auroc:.3f}", frac)
        
        # Calculate final results
        final_result = {
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        }
        results[name] = final_result
        
        # Mark model as completed in checkpoint
        if checkpoint:
            checkpoint.complete_model(name, final_result)
    
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
    checkpoint=None,
) -> Dict[str, Dict]:
    """Perform rigorous repeated stratified k-fold cross-validation with checkpointing."""
    
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED
    )
    
    total_iterations = n_splits * n_repeats
    results = {}
    total_models = len(models)
    
    for model_idx, (name, (base_model, param_grid)) in enumerate(models.items(), 1):
        # Check if model was already completed
        if checkpoint and checkpoint.should_skip_model(name):
            cached_results = checkpoint.get_model_results(name)
            if cached_results:
                results[name] = cached_results
                if progress_callback:
                    progress_callback(
                        f"✅ Modelo '{name}' cargado desde checkpoint | μ={cached_results['mean_score']:.4f}, σ={cached_results['std_score']:.4f}",
                        model_idx / total_models
                    )
                continue
        
        all_scores = []
        completed_folds = checkpoint.get_completed_folds(name) if checkpoint else []
        
        # ✅ CORREGIR: Cargar scores de folds ya completados
        if completed_folds and checkpoint:
            cached_data = checkpoint.state['current_model_results'].get(name, {})
            cached_scores = cached_data.get('scores', [])
            
            # Verificar que los scores existen y son válidos
            if len(cached_scores) >= len(completed_folds):
                all_scores = cached_scores[:len(completed_folds)]
                if progress_callback:
                    progress_callback(
                        f"♻️ Modelo '{name}': Recuperando {len(all_scores)} scores de {len(completed_folds)} folds completados",
                        (model_idx - 1) / total_models
                    )
            else:
                # Inconsistencia: reiniciar
                completed_folds = []
                all_scores = []
                if progress_callback:
                    progress_callback(
                        f"⚠️ Modelo '{name}': Checkpoint inconsistente, reiniciando...",
                        (model_idx - 1) / total_models
                    )
        
        if progress_callback:
            if completed_folds:
                progress_callback(
                    f"♻️ Modelo '{name}': Resumiendo desde fold {len(completed_folds)+1}/{total_iterations}",
                    (model_idx - 1) / total_models
                )
            else:
                progress_callback(
                    f"Modelo '{name}': Iniciando {total_iterations} corridas",
                    (model_idx - 1) / total_models
                )
        
        for run_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y), 1):
            # ✅ CORREGIR: Skip basado en run_idx (1-indexed)
            if run_idx in completed_folds:
                if progress_callback and run_idx % 10 == 0:  # Log cada 10 folds
                    progress_callback(
                        f"⏩ Modelo '{name}': Saltando fold {run_idx}/{total_iterations} (ya completado)",
                        (model_idx - 1 + run_idx / total_iterations) / total_models
                    )
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Show class distribution only for first NEW run
            if run_idx == (len(completed_folds) + 1) and progress_callback:
                train_dist = y_train.value_counts(normalize=True).sort_index()
                val_dist = y_val.value_counts(normalize=True).sort_index()
                dist_info = " | ".join([f"Clase {label}: Train={train_dist[label]*100:.1f}% Val={val_dist[label]*100:.1f}%" 
                                       for label in train_dist.index])
                progress_callback(f"Distribución: {dist_info}", (model_idx - 1) / total_models)
            
            # Show progress for each run
            if progress_callback:
                progress_callback(
                    f"⚙️ Modelo '{name}': Entrenando corrida {run_idx}/{total_iterations}...",
                    (model_idx - 1 + (run_idx - 0.5) / total_iterations) / total_models
                )
            
            # Build preprocessing pipeline
            preprocess_pipeline, _ = build_preprocessing_pipeline(X_train, preprocessing_config)
            
            # Extract the preprocessor
            if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
                preprocess = preprocess_pipeline.steps[0][1]
            else:
                preprocess = preprocess_pipeline
            
            # Build full pipeline with SMOTE if available
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    ("classifier", clone(base_model)),
                ])
            except ImportError:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("classifier", clone(base_model)),
                ])
            
            # Train and evaluate
            try:
                pipeline.fit(X_train, y_train)
                if hasattr(pipeline, "predict_proba"):
                    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                else:
                    y_pred_proba = pipeline.decision_function(X_val)
                score = roc_auc_score(y_val, y_pred_proba)
                all_scores.append(score)
                
                # Save fold checkpoint
                if checkpoint:
                    checkpoint.save_fold_result(
                        model_name=name,
                        fold_idx=run_idx,
                        score=score,
                        fitted_pipeline=pipeline,
                        metadata={
                            'train_indices': train_idx.tolist(),
                            'val_indices': val_idx.tolist(),
                        }
                    )
                
                # Update progress
                update_frequency = 1 if total_iterations <= 20 else 5
                if progress_callback and (run_idx % update_frequency == 0 or run_idx == total_iterations):
                    frac = (model_idx - 1 + run_idx / total_iterations) / total_models
                    current_mean = np.mean(all_scores)
                    current_std = np.std(all_scores)
                    progress_callback(
                        f"Modelo '{name}': Corrida {run_idx}/{total_iterations} | AUROC={score:.3f} | μ={current_mean:.3f}, σ={current_std:.3f}",
                        frac
                    )
            except Exception as e:
                print(f"Warning: Error in {name} run {run_idx}: {e}")
                continue
        
        # Calculate final statistics
        final_result = {
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores, ddof=1)),  # Sample std
            "all_scores": all_scores,
            "n_runs": len(all_scores),
            "best_params": param_grid if param_grid else {},
        }
        results[name] = final_result
        
        # Mark model as completed in checkpoint
        if checkpoint:
            checkpoint.complete_model(name, final_result)
        
        if progress_callback:
            progress_callback(
                f"✅ Modelo '{name}' COMPLETADO | μ={results[name]['mean_score']:.4f}, σ={results[name]['std_score']:.4f} ({len(all_scores)} corridas exitosas)",
                model_idx / total_models
            )
    
    return results

