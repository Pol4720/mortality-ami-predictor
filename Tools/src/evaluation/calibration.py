"""Calibration assessment utilities."""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports", "figures"
)
os.makedirs(FIG_DIR, exist_ok=True)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        Tuple of (fraction_positives, mean_predicted_value)
    """
    return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
    n_bins: int = 10,
) -> str:
    """Plot and save calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        n_bins: Number of bins
        
    Returns:
        Path to saved figure
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, "s-", label="Model", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylabel("Fraction of positives", fontsize=12)
    plt.title(f"Calibration Curve: {name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, f"calibration_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    
    return path
