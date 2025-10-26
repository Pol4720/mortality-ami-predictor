"""Partial dependence plot utilities."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports", "figures"
)
os.makedirs(FIG_DIR, exist_ok=True)


def plot_partial_dependence(
    model,
    X: np.ndarray,
    features: List[int],
    feature_names: Optional[List[str]] = None,
    name: str = "model",
    grid_resolution: int = 20,
) -> str:
    """Plot partial dependence plots.
    
    Args:
        model: Trained model
        X: Feature matrix
        features: List of feature indices to plot
        feature_names: List of feature names
        name: Model name for filename
        grid_resolution: Number of grid points for PDP
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(12, 4 * ((len(features) + 2) // 3)))
    
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features,
        feature_names=feature_names,
        grid_resolution=grid_resolution,
        ax=ax,
    )
    
    plt.suptitle(f"Partial Dependence Plots: {name}", fontsize=14, y=1.02)
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, f"partial_dependence_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return path
