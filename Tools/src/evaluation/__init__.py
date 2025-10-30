"""Evaluation module for model assessment.

This module provides utilities for computing metrics, calibration curves,
decision curves, resampling methods, and generating reports.
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
]
