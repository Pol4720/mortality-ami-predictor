"""Model explainability module.

This module provides SHAP analysis, permutation importance,
and partial dependence plots for model interpretation.
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
]
