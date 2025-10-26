"""Preprocessing module for ML pipelines.

This module provides sklearn-based preprocessing pipelines that integrate
with the data cleaning pipeline.
"""

from .pipelines import build_preprocessing_pipeline, PreprocessingConfig
from .integrations import load_data_with_fallback, get_latest_cleaned_dataset

# Backward compatibility alias
build_preprocess_pipelines = build_preprocessing_pipeline

__all__ = [
    "build_preprocessing_pipeline",
    "PreprocessingConfig",
    "load_data_with_fallback",
    "get_latest_cleaned_dataset",
    # Backward compatibility
    "build_preprocess_pipelines",
]
