"""Model explainability module.

This module provides SHAP analysis, permutation importance,
and partial dependence plots for model interpretation.
"""

from .shap_analysis import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
)
from .permutation import compute_permutation_importance, plot_permutation_importance
from .partial_dependence import plot_partial_dependence

__all__ = [
    "compute_shap_values",
    "plot_shap_summary",
    "plot_shap_waterfall",
    "compute_permutation_importance",
    "plot_permutation_importance",
    "plot_partial_dependence",
]
