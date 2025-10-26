"""SHAP (SHapley Additive exPlanations) analysis utilities."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports", "figures"
)
os.makedirs(FIG_DIR, exist_ok=True)


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    max_samples: int = 100,
):
    """Compute SHAP values for a model.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        feature_names: List of feature names
        max_samples: Maximum samples for background data
        
    Returns:
        SHAP explainer and values
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install with: pip install shap"
        )
    
    # Use subsample for efficiency
    if len(X) > max_samples:
        background_indices = np.random.choice(len(X), max_samples, replace=False)
        background = X[background_indices]
    else:
        background = X
    
    # Create explainer
    try:
        # Try TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        # Fall back to KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)
    
    return explainer, shap_values


def plot_shap_summary(
    shap_values,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    name: str = "model",
    max_display: int = 20,
) -> str:
    """Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values from compute_shap_values
        X: Feature matrix
        feature_names: List of feature names
        name: Model name for filename
        max_display: Maximum features to display
        
    Returns:
        Path to saved figure
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install with: pip install shap"
        )
    
    # Handle binary classification (extract positive class)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, f"shap_summary_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return path


def plot_shap_waterfall(
    shap_values,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    instance_idx: int = 0,
    name: str = "model",
) -> str:
    """Plot SHAP waterfall plot for a single instance.
    
    Args:
        shap_values: SHAP values from compute_shap_values
        X: Feature matrix
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        name: Model name for filename
        
    Returns:
        Path to saved figure
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install with: pip install shap"
        )
    
    # Handle binary classification
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_instance = shap_values[1][instance_idx]
    else:
        shap_values_instance = shap_values[instance_idx]
    
    plt.figure(figsize=(10, 6))
    
    # Create explanation object for waterfall plot
    if feature_names is not None:
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_instance,
                base_values=0.0,
                data=X[instance_idx],
                feature_names=feature_names,
            ),
            show=False,
        )
    else:
        # Fallback to bar plot if waterfall is not available
        indices = np.argsort(np.abs(shap_values_instance))[::-1][:20]
        plt.barh(range(len(indices)), shap_values_instance[indices])
        plt.xlabel("SHAP value")
        plt.ylabel("Feature")
        plt.title(f"SHAP Values: Instance {instance_idx}")
    
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, f"shap_waterfall_{name}_instance{instance_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return path
