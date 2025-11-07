"""Main training orchestration functions.

This module provides high-level functions for training ML models
with nested cross-validation and SMOTE for class imbalance.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import RANDOM_SEED
from ..models import make_classifiers
from ..preprocessing import build_preprocessing_pipeline, PreprocessingConfig
from ..data_load import save_model_with_cleanup, save_dataset_with_timestamp, cleanup_old_testsets, get_timestamp
from .cross_validation import nested_cross_validation, rigorous_repeated_cv
from .statistical_tests import compare_all_models, plot_model_comparison
from .learning_curves import generate_learning_curve, plot_learning_curve
from .checkpointing import TrainingCheckpoint, create_experiment_id  # NEW

# Try to import imbalanced-learn for SMOTE support
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None


def run_rigorous_experiment_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    n_splits: int = 10,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    test_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    use_checkpointing: bool = True,
    checkpoint_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Dict:
    """Orchestrate rigorous experiment pipeline with checkpointing."""
    
    import time
    start_time = time.time()
    results = {}
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup checkpointing
    checkpoint = None
    if use_checkpointing:
        # ✅ CORREGIDO: Usar ruta absoluta por defecto
        if checkpoint_dir is None:
            # Get absolute path to Tools/dashboard/checkpoints
            from pathlib import Path
            tools_dir = Path(__file__).parent.parent.parent  # Tools/
            checkpoint_dir = str(tools_dir / "dashboard" / "checkpoints")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate or use provided experiment ID
        if experiment_id is None:
            task_name = y.name if hasattr(y, 'name') else 'unknown'
            experiment_id = create_experiment_id(task_name, prefix="training")
        
        try:
            checkpoint = TrainingCheckpoint(
                checkpoint_dir=checkpoint_dir,
                experiment_id=experiment_id,
                models=models
            )
            
            if progress_callback:
                progress = checkpoint.get_progress()
                if progress['completed_models'] > 0:
                    progress_callback(
                        f"♻️ Checkpoint detectado: {progress['completed_models']}/{progress['total_models']} modelos completados | Experiment ID: {experiment_id}",
                        0.0
                    )
                else:
                    progress_callback(
                        f"💾 Checkpoint inicializado | Directorio: {checkpoint_dir} | ID: {experiment_id}",
                        0.0
                    )
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Error inicializando checkpoint: {e}. Continuando sin checkpointing...", 0.0)
            checkpoint = None
    
    if progress_callback:
        checkpoint_status = f"✅ Activo (ID: {experiment_id})" if checkpoint else "❌ Desactivado"
        progress_callback(f"🚀 Pipeline riguroso iniciado | Checkpointing: {checkpoint_status}", 0.0)

    # =========================================================================
    # FASE 1: TRAIN + VALIDATION (30+ corridas con k-fold estratificado)
    # =========================================================================
    if progress_callback:
        progress_callback("📊 FASE 1: Entrenamiento y Validación con validación cruzada estratificada repetida", 0.1)
    
    # 1. Rigorous repeated CV (≥30 runs to estimate μ and σ) WITH CHECKPOINTING
    cv_results = rigorous_repeated_cv(
        X, y, models,
        preprocessing_config=preprocessing_config,
        n_splits=n_splits,
        n_repeats=n_repeats,
        scoring=scoring,
        progress_callback=progress_callback,
        checkpoint=checkpoint,  # Pass checkpoint manager
    )
    results['cv_results'] = cv_results
    
    # Log validation results
    if progress_callback:
        msg = "✅ Validación Cruzada Completada:\n"
        for name, res in cv_results.items():
            msg += f"   • {name}: μ={res['mean_score']:.4f}, σ={res['std_score']:.4f} ({res['n_runs']} corridas)\n"
        progress_callback(msg, 0.3)

    # 2. Learning curves for model selection
    if progress_callback:
        progress_callback("📈 Generando curvas de aprendizaje para cada modelo...", 0.35)
    
    lc_results = {}
    for idx, (name, (model, param_grid)) in enumerate(models.items(), 1):
        from sklearn.base import clone
        model_for_lc = clone(model)
        
        if progress_callback:
            progress_callback(f"🔄 Generando curva de aprendizaje: {name} (evaluando 10 tamaños × {n_splits} folds)...", 0.35 + 0.15 * idx / len(models))
        
        lc_result = generate_learning_curve(
            model_for_lc,
            X.values,
            y.values,
            cv=n_splits,
            scoring=scoring,
            n_jobs=-1,
        )
        
        if progress_callback:
            progress_callback(f"✓ Curva generada: {name} (Train={lc_result.train_scores_mean[-1]:.3f}, Val={lc_result.val_scores_mean[-1]:.3f})", 0.35 + 0.15 * (idx + 0.5) / len(models))
        
        lc_results[name] = lc_result
        
        if progress_callback:
            progress_callback(f"💾 Guardando gráfica de curva de aprendizaje: {name}", 0.35 + 0.15 * (idx + 0.7) / len(models))
        
        fig = plot_learning_curve(lc_result, title=f"Learning Curve: {name}")
        save_path = os.path.join(output_dir, f"learning_curve_{name}.png")
        try:
            fig.write_image(save_path, width=1000, height=600)
        except Exception:
            html_path = os.path.join(output_dir, f"learning_curve_{name}.html")
            fig.write_html(html_path)
    
    results['learning_curves'] = lc_results
    
    # 3. Select best model
    best_name = max(cv_results.items(), key=lambda x: x[1]['mean_score'])[0]
    results['best_model'] = best_name
    
    if progress_callback:
        best_score = cv_results[best_name]['mean_score']
        progress_callback(f"🏆 Mejor modelo seleccionado: {best_name} (μ={best_score:.4f})", 0.5)

    # =========================================================================
    # FASE 3: COMPARACIÓN ESTADÍSTICA
    # =========================================================================
    if progress_callback:
        progress_callback("📊 FASE 3: Comparación estadística entre modelos", 0.55)
    
    model_scores = {name: res['all_scores'] for name, res in cv_results.items()}
    stat_results = compare_all_models(model_scores)
    results['statistical_comparison'] = stat_results

    if progress_callback:
        progress_callback("Generando gráficos de comparación estadística...", 0.6)
    
    for (m1, m2), stat_res in stat_results.items():
        fig = plot_model_comparison(stat_res)
        save_path = os.path.join(output_dir, f"comparison_{m1}_vs_{m2}.png")
        try:
            if hasattr(fig, 'write_image'):
                fig.write_image(save_path, width=1000, height=600)
            else:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception:
            if hasattr(fig, 'write_html'):
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
    
    from .statistical_tests import create_comparison_matrix
    fig_matrix = create_comparison_matrix(stat_results)
    matrix_path = os.path.join(output_dir, "comparison_matrix.png")
    try:
        if hasattr(fig_matrix, 'write_image'):
            fig_matrix.write_image(matrix_path, width=1200, height=800)
        else:
            fig_matrix.savefig(matrix_path, dpi=150, bbox_inches='tight')
    except Exception:
        if hasattr(fig_matrix, 'write_html'):
            fig_matrix.write_html(matrix_path.replace('.png', '.html'))
    
    # =========================================================================
    # Train and save final model
    # =========================================================================
    if progress_callback:
        progress_callback(f"Entrenando modelo final '{best_name}' con todos los datos de entrenamiento...", 0.7)
    
    best_model, best_params = models[best_name]
    from sklearn.base import clone
    best_model = clone(best_model)
    
    # Build preprocessing pipeline
    preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
    if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
        preprocess = preprocess_pipeline.steps[0][1]
    else:
        preprocess = preprocess_pipeline
    
    # Build full pipeline with SMOTE
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        final_pipeline = ImbPipeline([
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("classifier", best_model),
        ])
    except ImportError:
        from sklearn.pipeline import Pipeline
        final_pipeline = Pipeline([
            ("preprocess", preprocess),
            ("classifier", best_model),
        ])
    
    # Fit on all training data
    final_pipeline.fit(X, y)
    
    # Extract actual feature names
    try:
        if hasattr(final_pipeline, 'feature_names_in_'):
            actual_feature_names = list(final_pipeline.feature_names_in_)
        elif hasattr(final_pipeline.steps[-1][1], 'feature_names_in_'):
            actual_feature_names = list(final_pipeline.steps[-1][1].feature_names_in_)
        elif hasattr(final_pipeline.steps[0][1], 'feature_names_in_'):
            actual_feature_names = list(final_pipeline.steps[0][1].feature_names_in_)
        else:
            actual_feature_names = X.columns.tolist()
    except Exception:
        actual_feature_names = X.columns.tolist()
    
    # Extract model type
    model_type_map = {
        'logistic': 'logistic',
        'dtree': 'dtree',
        'knn': 'knn',
        'xgb': 'xgb',
        'random_forest': 'random_forest',
        'neural_network': 'neural_network',
    }
    model_type = model_type_map.get(best_name, best_name)
    
    # Save test/train sets
    from pathlib import Path
    models_dir = Path(output_dir)
    testsets_dir = models_dir / "testsets"
    
    if test_set is not None:
        X_test, y_test = test_set
        test_df = pd.concat([X_test, y_test], axis=1)
        
        test_path = save_dataset_with_timestamp(
            test_df,
            testsets_dir,
            prefix=f"testset_{model_type}",
            format="parquet"
        )
        results['test_set_path'] = str(test_path)
        
        train_df = pd.concat([X, y], axis=1)
        train_path = save_dataset_with_timestamp(
            train_df,
            testsets_dir,
            prefix=f"trainset_{model_type}",
            format="parquet"
        )
        results['train_set_path'] = str(train_path)
        
        cleanup_old_testsets(model_type, testsets_dir, keep_n_latest=1)
    else:
        test_path = None
        train_path = None
        test_df = None
        train_df = pd.concat([X, y], axis=1)
    
    # Create metadata
    if progress_callback:
        progress_callback("💾 Guardando metadatos del modelo...", 0.94)
    
    from ..models import create_metadata_from_training
    
    model_type_str = str(type(best_model).__name__)
    model_params = best_model.get_params() if hasattr(best_model, 'get_params') else {}
    
    training_duration = time.time() - start_time
    training_config = {
        'duration': training_duration,
        'cv_strategy': f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
        'n_splits': n_splits,
        'n_repeats': n_repeats,
        'total_runs': n_splits * n_repeats,
        'scoring': scoring,
        'preprocessing': preprocessing_config.__dict__ if preprocessing_config else {},
        'random_seed': RANDOM_SEED,
        'checkpointing_enabled': use_checkpointing,
        'experiment_id': experiment_id if checkpoint else None
    }
    
    metadata = create_metadata_from_training(
        model_name=best_name,
        model_type=model_type_str,
        task=y.name if hasattr(y, 'name') else 'unknown',
        model_file_path='',
        train_set_path=str(train_path) if train_path else '',
        test_set_path=str(test_path) if test_path else '',
        train_df=train_df,
        test_df=test_df if test_df is not None else train_df.head(0),
        target_column=y.name if hasattr(y, 'name') else 'target',
        hyperparameters=model_params,
        cv_results=cv_results[best_name],
        training_config=training_config,
        learning_curve_path=results.get('learning_curves_paths', {}).get(best_name),
        statistical_comparison=results.get('statistical_comparison'),
        notes=f"Best model selected from {len(models)} candidates",
        actual_feature_names=actual_feature_names
    )
    
    # Save final model
    full_training_df = pd.concat([X, y], axis=1)
    
    final_model_path = save_model_with_cleanup(
        final_pipeline,
        model_type,
        models_dir,
        keep_n_latest=1,
        metadata=metadata,
        training_data=full_training_df
    )
    results['final_model_path'] = str(final_model_path)
    results['metadata_path'] = str(final_model_path.with_suffix('.metadata.json'))
    results['training_duration'] = training_duration
    
    if progress_callback:
        progress_callback(f"✅ Modelo y metadatos guardados: {final_model_path}", 0.96)
    
    # Clean up checkpoints after successful completion
    if checkpoint and checkpoint.is_complete():
        if progress_callback:
            progress_callback("🧹 Limpiando checkpoints temporales...", 0.98)
        try:
            checkpoint.cleanup_checkpoints(keep_final=False)
            if progress_callback:
                progress_callback("✅ Checkpoints limpiados", 0.99)
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Error limpiando checkpoints: {e}", 0.99)
    
    if progress_callback:
        progress_callback("✅ Pipeline de entrenamiento completado exitosamente!", 1.0)
    
    return results


# Keep existing MODELS_DIR and other functions unchanged
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_best_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    quick: bool = False,
    task_name: str = "mortality",
    preprocessing_config: Optional[PreprocessingConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, object]:
    """Train multiple classifiers and select best based on nested CV.
    
    Performs nested cross-validation to evaluate all classifiers,
    selects the best, refits on full data, and saves.
    
    Args:
        X: Training features
        y: Training labels
        quick: If True, use reduced CV splits and iterations
        task_name: Task name for saving model
        preprocessing_config: Preprocessing configuration
        progress_callback: Optional callback(message, progress)
        
    Returns:
        Tuple of (save_path, fitted_model)
    """
    # Check if model already exists
    save_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    if os.path.exists(save_path):
        if progress_callback:
            progress_callback("Existing model found. Loading...", 1.0)
        model = joblib.load(save_path)
        return save_path, model
    
    # Get all available models
    models = make_classifiers()
    
    # Remove neural network unless explicitly enabled
    if "nn" in models and os.environ.get("ENABLE_TORCH_MODEL", "0") != "1":
        models.pop("nn", None)
    
    if progress_callback:
        progress_callback(f"Evaluating {len(models)} models...", 0.1)
    
    # Evaluate all models with nested CV
    results = nested_cross_validation(
        X=X,
        y=y,
        models=models,
        preprocessing_config=preprocessing_config,
        quick=quick,
        progress_callback=progress_callback,
    )
    
    # Find best model
    best_name = max(results.items(), key=lambda x: x[1]["mean_score"])[0]
    best_model, best_grid = models[best_name]
    
    if progress_callback:
        best_score = results[best_name]["mean_score"]
        progress_callback(f"Best model: {best_name} (AUROC: {best_score:.3f})", 0.8)
    
    # Refit best model on all data
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import RandomizedSearchCV
    
    # Build preprocessing pipeline
    preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
    
    # Extract the preprocessor from the pipeline (to avoid nested pipelines)
    if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
        preprocess = preprocess_pipeline.steps[0][1]  # Get the ColumnTransformer
    else:
        preprocess = preprocess_pipeline
    
    # Build full pipeline with SMOTE if available
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        pipeline = ImbPipeline([
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("classifier", best_model),
        ])
    except ImportError:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("classifier", best_model),
        ])
    
    # Hyperparameter search
    param_grid = {f"classifier__{k}": v for k, v in best_grid.items()}
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=15 if not quick else 6,
        cv=StratifiedKFold(n_splits=3 if not quick else 2, shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
    )
    
    search.fit(X, y)
    best_model_fitted = search.best_estimator_
    
    # Save
    joblib.dump(best_model_fitted, save_path)
    
    if progress_callback:
        progress_callback(f"Model saved to {save_path}", 1.0)
    
    return save_path, best_model_fitted


def train_selected_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: List[str],
    quick: bool = False,
    task_name: str = "mortality",
    preprocessing_config: Optional[PreprocessingConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, str]:
    """Train specific classifiers and save all.
    
    Args:
        X: Training features
        y: Training labels
        selected_models: List of model names to train
        quick: If True, use reduced CV
        task_name: Task name for saving
        preprocessing_config: Preprocessing configuration
        progress_callback: Optional callback(message, progress)
        
    Returns:
        Dictionary mapping model_name -> save_path
    """
    all_models = make_classifiers()
    models = {k: v for k, v in all_models.items() if k in selected_models}
    
    if not models:
        raise ValueError("No valid models selected")
    
    # Evaluate with nested CV
    results = nested_cross_validation(
        X=X,
        y=y,
        models=models,
        preprocessing_config=preprocessing_config,
        quick=quick,
        progress_callback=progress_callback,
    )
    
    # Refit and save each model
    save_paths = {}
    
    for idx, (name, (model, grid)) in enumerate(models.items(), 1):
        if progress_callback:
            progress_callback(f"Refitting {name}...", 0.5 + (idx / len(models)) * 0.5)
        
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import RandomizedSearchCV
        
        # Build pipeline
        preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
        
        # Extract the preprocessor from the pipeline (to avoid nested pipelines)
        if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
            preprocess = preprocess_pipeline.steps[0][1]  # Get the ColumnTransformer
        else:
            preprocess = preprocess_pipeline
        
        if ImbPipeline and SMOTE:
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
        else:
            pipeline = Pipeline([
                ("preprocess", preprocess),
                ("classifier", model),
            ])
        
        # Search and refit
        param_grid = {f"classifier__{k}": v for k, v in grid.items()}
        
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=15 if not quick else 6,
            cv=StratifiedKFold(n_splits=3 if not quick else 2, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            refit=True,
        )
        
        search.fit(X, y)
        
        # Save
        path = os.path.join(MODELS_DIR, f"model_{task_name}_{name}.joblib")
        joblib.dump(search.best_estimator_, path)
        save_paths[name] = path
    
    # Also save best overall
    best_name = max(results.items(), key=lambda x: x[1]["mean_score"])[0]
    best_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    joblib.dump(joblib.load(save_paths[best_name]), best_path)
    
    if progress_callback:
        progress_callback("Training complete!", 1.0)
    
    return save_paths
