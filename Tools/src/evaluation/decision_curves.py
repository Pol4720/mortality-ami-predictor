"""Decision curve analysis for clinical utility assessment."""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports", "figures"
)
os.makedirs(FIG_DIR, exist_ok=True)


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
    n_thresholds: int = 50,
) -> str:
    """Perform decision curve analysis and plot.
    
    Decision curve analysis assesses the clinical utility of a prediction model
    by calculating the net benefit across different probability thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        n_thresholds: Number of threshold points to evaluate
        
    Returns:
        Path to saved figure
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    net_benefits = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        n = len(y_true)
        
        # Net benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold))
        pt = threshold / (1 - threshold)
        nb = (tp / n) - (fp / n) * pt
        net_benefits.append(nb)
    
    # Treat all strategy
    prevalence = y_true.mean()
    treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    
    # Treat none strategy
    treat_none_nb = np.zeros_like(thresholds)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits, label="Model", linewidth=2, color="blue")
    plt.plot(thresholds, treat_all_nb, label="Treat All", linestyle="--", color="green")
    plt.plot(thresholds, treat_none_nb, label="Treat None", linestyle=":", color="red")
    plt.xlabel("Threshold Probability", fontsize=12)
    plt.ylabel("Net Benefit", fontsize=12)
    plt.title(f"Decision Curve Analysis: {name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, f"decision_curve_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    
    return path
