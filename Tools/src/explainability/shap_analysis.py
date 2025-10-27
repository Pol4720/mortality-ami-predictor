"""SHAP (SHapley Additive exPlanations) analysis utilities."""
from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
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
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[list] = None,
    max_samples: int = 100,
) -> Tuple[shap.Explainer, shap.Explanation]:
    """Compute SHAP values for a model with proper handling of multi-output models.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        feature_names: List of feature names
        max_samples: Maximum samples for background data
        
    Returns:
        Tuple of (explainer, shap_explanation)
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install with: pip install shap"
        )
    
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is not None:
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = pd.DataFrame(X)
    else:
        X_df = X
        if feature_names is None:
            feature_names = list(X_df.columns)
    
    # Create explainer based on model type
    try:
        # Try TreeExplainer first (faster for tree-based models)
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_df)
        base_value = explainer.expected_value
    except Exception:
        # Fall back to general Explainer
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, X_df[:max_samples])
        else:
            explainer = shap.Explainer(model.predict, X_df[:max_samples])
        
        shap_values_raw = explainer(X_df)
        base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else None
    
    # Handle multi-output models (binary classification returns 2 outputs)
    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
        # For binary classification, use positive class (index 1)
        shap_values_raw = shap_values_raw[1]
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
    
    # If shap_values_raw is an Explanation object, extract and reshape
    if hasattr(shap_values_raw, 'values'):
        # It's already an Explanation object
        if shap_values_raw.values.ndim > 2:
            # Multi-output: take last dimension (positive class)
            shap_values_array = shap_values_raw.values[..., -1]
            # Handle base_values for multi-output
            if hasattr(shap_values_raw, 'base_values'):
                bv = shap_values_raw.base_values
                if isinstance(bv, np.ndarray) and bv.ndim > 1:
                    base_values_array = bv[..., -1]
                else:
                    base_values_array = bv
            else:
                base_values_array = base_value
        else:
            shap_values_array = shap_values_raw.values
            base_values_array = shap_values_raw.base_values if hasattr(shap_values_raw, 'base_values') else base_value
        
        # Create new Explanation object with correct shape
        shap_explanation = shap.Explanation(
            values=shap_values_array,
            base_values=base_values_array,
            data=shap_values_raw.data if hasattr(shap_values_raw, 'data') else X_df.values,
            feature_names=feature_names
        )
    else:
        # It's a numpy array
        shap_explanation = shap.Explanation(
            values=shap_values_raw,
            base_values=base_value,
            data=X_df.values,
            feature_names=feature_names
        )
    
    return explainer, shap_explanation


def plot_shap_beeswarm(
    shap_explanation: shap.Explanation,
    max_display: int = 20,
) -> plt.Figure:
    """Create SHAP beeswarm plot.
    
    Args:
        shap_explanation: SHAP Explanation object
        max_display: Maximum features to display
        
    Returns:
        matplotlib Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    
    return fig


def plot_shap_bar(
    shap_explanation: shap.Explanation,
    max_display: int = 20,
) -> plt.Figure:
    """Create SHAP bar plot (feature importance).
    
    Args:
        shap_explanation: SHAP Explanation object
        max_display: Maximum features to display
        
    Returns:
        matplotlib Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    
    return fig


def plot_shap_waterfall(
    shap_explanation: shap.Explanation,
    sample_idx: int = 0,
) -> plt.Figure:
    """Create SHAP waterfall plot for a single sample.
    
    Args:
        shap_explanation: SHAP Explanation object
        sample_idx: Index of sample to explain
        
    Returns:
        matplotlib Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_explanation[sample_idx], show=False)
    plt.tight_layout()
    
    return fig


def plot_shap_force(
    shap_explanation: shap.Explanation,
    sample_idx: int = 0,
) -> plt.Figure:
    """Create SHAP force plot for a single sample.
    
    Args:
        shap_explanation: SHAP Explanation object
        sample_idx: Index of sample to explain
        
    Returns:
        matplotlib Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 3))
    
    # Get the explanation for this sample
    sample_explanation = shap_explanation[sample_idx]
    
    # Get base value
    base_value = sample_explanation.base_values
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
    else:
        base_value = float(base_value) if base_value is not None else 0.0
    
    # Get SHAP values and feature values
    shap_vals = sample_explanation.values
    feature_vals = sample_explanation.data
    feature_names = sample_explanation.feature_names
    
    # Sort by absolute SHAP value
    idx_sorted = np.argsort(np.abs(shap_vals))[::-1][:20]  # Top 20 features
    
    # Create horizontal bar plot
    y_pos = np.arange(len(idx_sorted))
    colors = ['#FF0D57' if v > 0 else '#1E88E5' for v in shap_vals[idx_sorted]]
    
    ax.barh(y_pos, shap_vals[idx_sorted], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{feature_names[i]} = {feature_vals[i]:.2f}" for i in idx_sorted])
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(f'Force Plot - Sample {sample_idx}\nBase value: {base_value:.4f}', fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def get_feature_importance(
    shap_explanation: shap.Explanation,
) -> pd.DataFrame:
    """Calculate feature importance from SHAP values.
    
    Args:
        shap_explanation: SHAP Explanation object
        
    Returns:
        DataFrame with feature names and importance scores
    """
    # Get feature names
    if hasattr(shap_explanation, 'feature_names') and shap_explanation.feature_names is not None:
        feature_names = shap_explanation.feature_names
    else:
        feature_names = [f"Feature {i}" for i in range(shap_explanation.values.shape[1])]
    
    # Calculate mean absolute SHAP values
    shap_vals = shap_explanation.values
    if shap_vals.ndim > 2:
        # Multi-output: take last dimension
        shap_vals = shap_vals[..., -1]
    
    mean_shap = np.abs(shap_vals).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_shap
    }).sort_values("Mean |SHAP|", ascending=False)
    
    return importance_df


def get_sample_shap_values(
    shap_explanation: shap.Explanation,
    sample_idx: int = 0,
) -> pd.DataFrame:
    """Get SHAP values for a single sample as a DataFrame.
    
    Args:
        shap_explanation: SHAP Explanation object
        sample_idx: Index of sample
        
    Returns:
        DataFrame with feature names and SHAP values
    """
    # Get feature names
    if hasattr(shap_explanation, 'feature_names') and shap_explanation.feature_names is not None:
        feature_names = shap_explanation.feature_names
    else:
        feature_names = [f"Feature {i}" for i in range(shap_explanation.values.shape[1])]
    
    # Get SHAP values for this sample
    shap_vals = shap_explanation.values[sample_idx]
    
    # Create DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals
    }).sort_values("SHAP Value", key=abs, ascending=False)
    
    return shap_df
