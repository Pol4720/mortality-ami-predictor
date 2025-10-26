"""Evaluation module for model assessment.

This module provides utilities for computing metrics, calibration curves,
decision curves, and generating reports.
"""

from .metrics import compute_classification_metrics, compute_regression_metrics
from .calibration import plot_calibration_curve, compute_calibration
from .decision_curves import decision_curve_analysis
from .reporters import generate_evaluation_report, evaluate_main

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "plot_calibration_curve",
    "compute_calibration",
    "decision_curve_analysis",
    "generate_evaluation_report",
    "evaluate_main",
]
