"""Training module for ML model training workflows.

This module provides utilities for training, cross-validation,
hyperparameter tuning, custom models integration, and PDF report generation.
"""

from .trainer import (
    train_best_classifier,
    train_selected_classifiers,
    run_rigorous_experiment_pipeline,
)
from .cross_validation import nested_cross_validation, rigorous_repeated_cv
from .hyperparameter_tuning import randomized_search
from .statistical_tests import compare_all_models, compare_models
from .learning_curves import generate_learning_curve
from .pdf_reports import generate_training_pdf
from .custom_integration import (
    train_custom_model,
    cross_validate_custom_model,
    create_custom_model_metadata,
    save_custom_model_with_metadata,
    integrate_custom_models_in_pipeline,
    train_mixed_models_with_cv,
    is_custom_model,
)

# Backward compatibility aliases for old API
# fit_and_save_best_classifier -> train_best_classifier (same function, different name)
fit_and_save_best_classifier = train_best_classifier

# fit_and_save_selected_classifiers -> train_selected_classifiers (same function, different name)
fit_and_save_selected_classifiers = train_selected_classifiers

__all__ = [
    # Main training functions
    "train_best_classifier",
    "train_selected_classifiers",
    "run_rigorous_experiment_pipeline",
    # Cross-validation
    "nested_cross_validation",
    "rigorous_repeated_cv",
    # Hyperparameter tuning
    "randomized_search",
    # Statistical tests
    "compare_all_models",
    "compare_models",
    # Learning curves
    "generate_learning_curve",
    # PDF reports
    "generate_training_pdf",
    # Custom models integration
    "train_custom_model",
    "cross_validate_custom_model",
    "create_custom_model_metadata",
    "save_custom_model_with_metadata",
    "integrate_custom_models_in_pipeline",
    "train_mixed_models_with_cv",
    "is_custom_model",
    # Backward compatibility
    "fit_and_save_best_classifier",
    "fit_and_save_selected_classifiers",
]
