"""Evaluation module for model assessment.

This module provides utilities for computing metrics, calibration curves,
decision curves, resampling methods, custom models evaluation, and generating reports.
"""

from .metrics import compute_classification_metrics, compute_regression_metrics
from .calibration import plot_calibration_curve, compute_calibration
from .decision_curves import decision_curve_analysis
from .reporters import generate_evaluation_report, evaluate_main
from .resampling import (
    bootstrap_evaluation,
    jackknife_evaluation,
    combined_resampling_evaluation,
    plot_resampling_results,
    ResamplingResult,
)
from .pdf_reports import generate_evaluation_pdf
from .custom_integration import (
    evaluate_custom_classifier,
    evaluate_custom_regressor,
    evaluate_model_universal,
    batch_evaluate_mixed_models,
    compare_model_performance,
    create_evaluation_summary,
)

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "plot_calibration_curve",
    "compute_calibration",
    "decision_curve_analysis",
    "generate_evaluation_report",
    "evaluate_main",
    "bootstrap_evaluation",
    "jackknife_evaluation",
    "combined_resampling_evaluation",
    "plot_resampling_results",
    "ResamplingResult",
    "generate_evaluation_pdf",
    # Custom models evaluation
    "evaluate_custom_classifier",
    "evaluate_custom_regressor",
    "evaluate_model_universal",
    "batch_evaluate_mixed_models",
    "compare_model_performance",
    "create_evaluation_summary",
]
