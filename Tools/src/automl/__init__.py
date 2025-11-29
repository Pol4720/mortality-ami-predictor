"""AutoML Module - Automated Machine Learning Integration.

This module provides AutoML capabilities using auto-sklearn and FLAML,
integrated seamlessly with the existing ML pipeline for AMI mortality prediction.

Main components:
- AutoMLClassifier: Wrapper for auto-sklearn classifier
- AutoMLRegressor: Wrapper for auto-sklearn regressor  
- FLAMLClassifier: Cross-platform alternative using FLAML
- AutoMLConfig: Configuration management
- AutoMLSuggestions: Intelligent technique recommendations

Note:
    auto-sklearn requires Linux or WSL on Windows.
    FLAML works on all platforms and is used as fallback.
"""

from .config import AutoMLConfig, AutoMLPreset, get_automl_config
from .auto_sklearn_integration import (
    AutoMLClassifier,
    AutoMLRegressor,
    is_autosklearn_available,
)
from .flaml_integration import (
    FLAMLClassifier,
    FLAMLRegressor,
    is_flaml_available,
)
from .suggestions import (
    AutoMLSuggestions,
    DatasetAnalysis,
    TechniqueSuggestion,
    analyze_dataset,
    get_suggestions,
)
from .export import (
    export_best_model,
    export_ensemble,
    convert_to_standalone,
    create_automl_report,
    load_ensemble,
)

__all__ = [
    # Configuration
    "AutoMLConfig",
    "AutoMLPreset",
    "get_automl_config",
    # auto-sklearn
    "AutoMLClassifier",
    "AutoMLRegressor",
    "is_autosklearn_available",
    # FLAML (cross-platform)
    "FLAMLClassifier",
    "FLAMLRegressor",
    "is_flaml_available",
    # Suggestions
    "AutoMLSuggestions",
    "DatasetAnalysis",
    "TechniqueSuggestion",
    "analyze_dataset",
    "get_suggestions",
    # Export
    "export_best_model",
    "export_ensemble",
    "convert_to_standalone",
    "create_automl_report",
    "load_ensemble",
]
