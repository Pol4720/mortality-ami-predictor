"""Model explainability module.

This module provides SHAP analysis, permutation importance,
partial dependence plots, and custom models explainability for model interpretation.
"""

from .shap_analysis import (
    compute_shap_values,
    plot_shap_beeswarm,
    plot_shap_bar,
    plot_shap_waterfall,
    plot_shap_force,
    get_feature_importance,
    get_sample_shap_values,
)
from .permutation import compute_permutation_importance, plot_permutation_importance
from .partial_dependence import plot_partial_dependence
from .pdf_reports import generate_explainability_pdf
from .custom_integration import (
    compute_shap_for_custom_model,
    compute_permutation_importance_custom,
    get_feature_importance_universal,
    explain_prediction_custom,
    batch_explain_models,
)

__all__ = [
    "compute_shap_values",
    "plot_shap_beeswarm",
    "plot_shap_bar",
    "plot_shap_waterfall",
    "plot_shap_force",
    "get_feature_importance",
    "get_sample_shap_values",
    "compute_permutation_importance",
    "plot_permutation_importance",
    "plot_partial_dependence",
    "generate_explainability_pdf",
    # Custom models explainability
    "compute_shap_for_custom_model",
    "compute_permutation_importance_custom",
    "get_feature_importance_universal",
    "explain_prediction_custom",
    "batch_explain_models",
]
