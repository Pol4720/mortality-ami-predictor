"""Main training orchestration functions.

This module provides high-level functions for training ML models
with nested cross-validation and SMOTE for class imbalance.
"""
from __future__ import annotations

import os
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

# Try to import imbalanced-learn for SMOTE support
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
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
) -> Dict:
    """
    Orchestrate rigorous experiment pipeline following academic standards:
    
    FASE 1: TRAIN + VALIDATION
    1. Repeated stratified k-fold CV for all models (≥30 runs)
       → Estimate μ (mean) and σ (std) for each model
    2. Learning curves for model selection
    3. Select best model based on validation performance
    
    FASE 2: TEST (Final Estimate on Hold-out Set)
    4. Bootstrap resampling (with replacement) on test set
    5. Jackknife resampling (leave-one-out) on test set
       → Obtain final performance estimates with confidence intervals
    
    FASE 3: STATISTICAL COMPARISON
    6. Compare all models pairwise:
       - Shapiro-Wilk normality test
       - If normal: Paired t-test (parametric)
       - If not normal: Mann-Whitney U test (non-parametric)
       → Determine if differences are statistically significant
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of {name: (model, param_dict)}
        preprocessing_config: Preprocessing configuration
        n_splits: Number of CV folds (default: 10)
        n_repeats: Number of CV repetitions (default: 10, gives 100 runs)
        scoring: Scoring metric
        test_set: Optional (X_test, y_test) for final evaluation
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with complete experiment results
    """
    import os
    results = {}
    if output_dir is None:
        output_dir = MODELS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if progress_callback:
        progress_callback("🚀 Iniciando pipeline de experimentación riguroso...", 0.0)

    # =========================================================================
    # FASE 1: TRAIN + VALIDATION (30+ corridas con k-fold estratificado)
    # =========================================================================
    if progress_callback:
        progress_callback("📊 FASE 1: Entrenamiento y Validación con validación cruzada estratificada repetida", 0.1)
    
    # 1. Rigorous repeated CV (≥30 runs to estimate μ and σ)
    cv_results = rigorous_repeated_cv(
        X, y, models,
        preprocessing_config=preprocessing_config,
        n_splits=n_splits,
        n_repeats=n_repeats,
        scoring=scoring,
        progress_callback=progress_callback,
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
        # Use the base model (without hyperparameter tuning) for learning curves
        # Learning curves show how model performs with different training sizes,
        # not different hyperparameters
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
        # Save as PNG using plotly's write_image (requires kaleido)
        save_path = os.path.join(output_dir, f"learning_curve_{name}.png")
        try:
            fig.write_image(save_path, width=1000, height=600)
        except Exception as e:
            # If write_image fails (missing kaleido), save as HTML instead
            html_path = os.path.join(output_dir, f"learning_curve_{name}.html")
            fig.write_html(html_path)
            if progress_callback:
                progress_callback(f"⚠️ Guardado como HTML (instalar kaleido para PNG): {html_path}", 0.35 + 0.15 * idx / len(models))
    results['learning_curves'] = lc_results
    
    # 3. Select best model
    best_name = max(cv_results.items(), key=lambda x: x[1]['mean_score'])[0]
    results['best_model'] = best_name
    
    if progress_callback:
        best_score = cv_results[best_name]['mean_score']
        progress_callback(f"🏆 Mejor modelo seleccionado: {best_name} (μ={best_score:.4f})", 0.5)

    # =========================================================================
    # FASE 3: COMPARACIÓN ESTADÍSTICA (hacer antes del test final)
    # =========================================================================
    if progress_callback:
        progress_callback("📊 FASE 3: Comparación estadística entre modelos", 0.55)
    
    # Statistical comparison with Shapiro-Wilk → t-test or Mann-Whitney
    model_scores = {name: res['all_scores'] for name, res in cv_results.items()}
    stat_results = compare_all_models(model_scores)
    results['statistical_comparison'] = stat_results

    # Save pairwise comparison plots
    if progress_callback:
        progress_callback("Generando gráficos de comparación estadística...", 0.6)
    
    for (m1, m2), stat_res in stat_results.items():
        fig = plot_model_comparison(stat_res)
        save_path = os.path.join(output_dir, f"comparison_{m1}_vs_{m2}.png")
        try:
            # Check if it's a Plotly figure
            if hasattr(fig, 'write_image'):
                fig.write_image(save_path, width=1000, height=600)
            else:
                # Matplotlib figure
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            # If save fails, try HTML for Plotly
            if hasattr(fig, 'write_html'):
                html_path = os.path.join(output_dir, f"comparison_{m1}_vs_{m2}.html")
                fig.write_html(html_path)
        
        # Log statistical results
        if progress_callback:
            test_type = stat_res.test_used
            p_val = stat_res.p_value
            significant = "SÍ" if stat_res.significant else "NO"
            msg = (f"   • {m1} vs {m2}:\n"
                   f"     Test: {test_type}, p-value={p_val:.4f}\n"
                   f"     Diferencia significativa: {significant}")
            progress_callback(msg, 0.6)
    
    # Create comparison matrix
    from .statistical_tests import create_comparison_matrix
    fig_matrix = create_comparison_matrix(stat_results)
    matrix_path = os.path.join(output_dir, "comparison_matrix.png")
    try:
        if hasattr(fig_matrix, 'write_image'):
            fig_matrix.write_image(matrix_path, width=1000, height=600)
        else:
            fig_matrix.savefig(matrix_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        if hasattr(fig_matrix, 'write_html'):
            html_path = os.path.join(output_dir, "comparison_matrix.html")
            fig_matrix.write_html(html_path)
    
    # =========================================================================
    # FASE 2: TEST (Estimado final con Bootstrap y Jackknife)
    # =========================================================================
    # NOTA: La evaluación en el test set se hará en el módulo de EVALUACIÓN,
    # no aquí. Aquí solo entrenamos y guardamos el mejor modelo.
    
    # Fit best model on full training data and save
    if progress_callback:
        progress_callback(f"Entrenando modelo final '{best_name}' con todos los datos de entrenamiento...", 0.7)
    
    best_model, best_params = models[best_name]
    from sklearn.base import clone
    best_model = clone(best_model)
    
    # Note: best_params is a param_grid for search, not a single set of params
    # For the final model, we train with default params (which are already tuned in the model definition)
    # If you want to use specific params, you would need to do hyperparameter search first
    
    # Build preprocessing pipeline for the best model
    preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
    if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
        preprocess = preprocess_pipeline.steps[0][1]
    else:
        preprocess = preprocess_pipeline
    
    # Build full pipeline with SMOTE if available
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
    
    # CRITICAL: Extract actual feature names from the fitted pipeline
    # This ensures metadata reflects the EXACT features the model expects
    try:
        # Try to get feature_names_in_ from the pipeline
        if hasattr(final_pipeline, 'feature_names_in_'):
            actual_feature_names = final_pipeline.feature_names_in_.tolist()
        # Try from the last step (classifier)
        elif hasattr(final_pipeline.steps[-1][1], 'feature_names_in_'):
            actual_feature_names = final_pipeline.steps[-1][1].feature_names_in_.tolist()
        # Try from preprocessor step
        elif hasattr(final_pipeline.steps[0][1], 'feature_names_in_'):
            actual_feature_names = final_pipeline.steps[0][1].feature_names_in_.tolist()
        else:
            # Fallback to X columns
            actual_feature_names = X.columns.tolist()
    except Exception as e:
        if progress_callback:
            progress_callback(f"⚠️ Could not extract feature_names_in_, using X.columns: {e}", 0.88)
        actual_feature_names = X.columns.tolist()
    
    # Validate that feature names match
    if set(actual_feature_names) != set(X.columns.tolist()):
        if progress_callback:
            progress_callback(f"⚠️ Feature mismatch detected! Model features: {len(actual_feature_names)}, X features: {len(X.columns)}", 0.89)
    else:
        if progress_callback:
            progress_callback(f"✅ Feature validation passed: {len(actual_feature_names)} features", 0.89)
    
    # Extract model type from best_name for proper directory organization
    # Map model names to directory names
    model_type_map = {
        'logistic': 'logistic',
        'dtree': 'dtree',
        'knn': 'knn',
        'xgb': 'xgb',
        'random_forest': 'random_forest',
        'neural_network': 'neural_network',
    }
    model_type = model_type_map.get(best_name, best_name)
    
    # Save test/train sets first to get paths for metadata
    from pathlib import Path
    models_dir = Path(output_dir)
    testsets_dir = models_dir / "testsets"
    
    if test_set is not None:
        X_test, y_test = test_set
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save testset and trainset with timestamp
        test_path = save_dataset_with_timestamp(
            test_df,
            testsets_dir,
            prefix=f"testset_{model_type}",
            format="parquet"
        )
        results['test_set_path'] = str(test_path)
        
        # Also save the training set
        train_df = pd.concat([X, y], axis=1)
        train_path = save_dataset_with_timestamp(
            train_df,
            testsets_dir,
            prefix=f"trainset_{model_type}",
            format="parquet"
        )
        results['train_set_path'] = str(train_path)
        
        # Clean up old testsets/trainsets (keep only latest)
        cleanup_old_testsets(model_type, testsets_dir, keep_n_latest=1)
        
        if progress_callback:
            progress_callback(f"Test/Train sets guardados en: {testsets_dir}", 0.92)
    else:
        # If no test set, create dummy paths
        test_path = None
        train_path = None
        test_df = None
        train_df = pd.concat([X, y], axis=1)
    
    # Create and save metadata
    if progress_callback:
        progress_callback("💾 Guardando metadatos del modelo...", 0.94)
    
    from ..models import create_metadata_from_training
    import time
    
    # Get model type string
    model_type_str = str(type(best_model).__name__)
    
    # Get hyperparameters
    if hasattr(best_model, 'get_params'):
        model_params = best_model.get_params()
    else:
        model_params = {}
    
    # Training configuration
    training_config = {
        'duration': results.get('training_duration', 0.0),
        'cv_strategy': f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
        'n_splits': n_splits,
        'n_repeats': n_repeats,
        'total_runs': n_splits * n_repeats,
        'scoring': scoring,
        'preprocessing': preprocessing_config.__dict__ if preprocessing_config else {},
        'random_seed': RANDOM_SEED
    }
    
    # Create metadata WITH ACTUAL FEATURES FROM TRAINED MODEL
    metadata = create_metadata_from_training(
        model_name=best_name,
        model_type=model_type_str,
        task=y.name if hasattr(y, 'name') else 'unknown',
        model_file_path='',  # Will be set when saving
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
        actual_feature_names=actual_feature_names  # Pass actual features from trained model
    )
    
    # Save the final trained model with metadata AND training data
    # IMPORTANT: Pass the COMPLETE training dataframe (features + target)
    # This ensures we can validate features later
    full_training_df = pd.concat([X, y], axis=1)
    
    final_model_path = save_model_with_cleanup(
        final_pipeline,
        model_type,
        models_dir,
        keep_n_latest=1,
        metadata=metadata,
        training_data=full_training_df  # Save exact data used for training
    )
    results['final_model_path'] = str(final_model_path)
    results['metadata_path'] = str(final_model_path.with_suffix('.metadata.json'))
    
    if progress_callback:
        progress_callback(f"✅ Modelo y metadatos guardados: {final_model_path}", 0.96)
    
    if progress_callback:
        progress_callback("✅ Pipeline de entrenamiento completado exitosamente!", 1.0)
    
    return results


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
