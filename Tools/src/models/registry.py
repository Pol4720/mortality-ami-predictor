"""Model registry for easy access to all models."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .classifiers import make_classifiers
from .regressors import make_regressors


class ModelRegistry:
    """Central registry for all available models.
    
    Provides a single interface to access classification and regression models.
    """
    
    def __init__(self):
        """Initialize registry with available models."""
        self._classifiers = make_classifiers()
        self._regressors = make_regressors()
    
    def get_classifier(self, name: str) -> Tuple[object, Dict]:
        """Get classifier and its parameter grid.
        
        Args:
            name: Model name ('knn', 'logreg', 'dtree', 'xgb', 'lgbm')
            
        Returns:
            Tuple of (model, param_grid)
            
        Raises:
            KeyError: If model name not found
        """
        if name not in self._classifiers:
            available = list(self._classifiers.keys())
            raise KeyError(
                f"Classifier '{name}' not found. "
                f"Available: {available}"
            )
        return self._classifiers[name]
    
    def get_regressor(self, name: str) -> Tuple[object, Dict]:
        """Get regressor and its parameter grid.
        
        Args:
            name: Model name
            
        Returns:
            Tuple of (model, param_grid)
            
        Raises:
            KeyError: If model name not found
        """
        if name not in self._regressors:
            available = list(self._regressors.keys())
            raise KeyError(
                f"Regressor '{name}' not found. "
                f"Available: {available}"
            )
        return self._regressors[name]
    
    def list_classifiers(self) -> list[str]:
        """List all available classifier names."""
        return list(self._classifiers.keys())
    
    def list_regressors(self) -> list[str]:
        """List all available regressor names."""
        return list(self._regressors.keys())


# Global registry instance
_registry = ModelRegistry()


def get_model(
    name: str,
    task: str = "classification",
) -> Tuple[object, Dict]:
    """Get model and parameter grid from global registry.
    
    Args:
        name: Model name
        task: Task type ('classification' or 'regression')
        
    Returns:
        Tuple of (model, param_grid)
    """
    if task == "classification":
        return _registry.get_classifier(name)
    elif task == "regression":
        return _registry.get_regressor(name)
    else:
        raise ValueError(f"Unknown task: {task}")
