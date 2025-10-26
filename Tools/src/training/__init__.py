"""Training module for ML model training workflows.

This module provides utilities for training, cross-validation,
and hyperparameter tuning.
"""

from .trainer import train_best_classifier, train_selected_classifiers
from .cross_validation import nested_cross_validation
from .hyperparameter_tuning import randomized_search

# Backward compatibility aliases for old API
# fit_and_save_best_classifier -> train_best_classifier (same function, different name)
fit_and_save_best_classifier = train_best_classifier

# fit_and_save_selected_classifiers -> train_selected_classifiers (same function, different name)
fit_and_save_selected_classifiers = train_selected_classifiers

__all__ = [
    "train_best_classifier",
    "train_selected_classifiers",
    "nested_cross_validation",
    "randomized_search",
    # Backward compatibility
    "fit_and_save_best_classifier",
    "fit_and_save_selected_classifiers",
]
