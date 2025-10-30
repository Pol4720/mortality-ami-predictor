"""Training module for ML model training workflows.

This module provides utilities for training, cross-validation,
and hyperparameter tuning.
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
    # Backward compatibility
    "fit_and_save_best_classifier",
    "fit_and_save_selected_classifiers",
]
