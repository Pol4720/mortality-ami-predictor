"""Models module - Machine Learning model definitions.

This module provides:
- Classification models (KNN, Logistic Regression, Tree-based, XGBoost, LightGBM)
- Regression models
- Neural network models (PyTorch)
- Model registry and factory patterns
"""

from .classifiers import make_classifiers
from .regressors import make_regressors
from .neural_networks import TorchTabularClassifier
from .registry import ModelRegistry, get_model

__all__ = [
    # Factories
    "make_classifiers",
    "make_regressors",
    # Neural networks
    "TorchTabularClassifier",
    # Registry
    "ModelRegistry",
    "get_model",
]
