"""Multivariate analysis utilities (PCA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResults:
    """Results from Principal Component Analysis."""
    
    n_components: int
    explained_variance: List[float]
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    components: np.ndarray
    feature_names: List[str]
    transformed_data: Optional[pd.DataFrame] = None


def perform_pca(
    df: pd.DataFrame,
    numeric_cols: List[str],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    scale: bool = True,
) -> PCAResults:
    """Perform Principal Component Analysis (PCA).
    
    Args:
        df: DataFrame
        numeric_cols: List of numerical column names
        n_components: Number of components. If None, uses variance_threshold
        variance_threshold: Desired cumulative variance (0-1)
        scale: Whether to standardize data before PCA
        
    Returns:
        PCAResults object
    """
    if len(numeric_cols) < 2:
        raise ValueError("At least 2 numerical variables required for PCA")
    
    # Prepare data
    df_numeric = df[numeric_cols].dropna()
    
    # Scale if requested
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
    else:
        X_scaled = df_numeric.values
    
    # Initial PCA to see all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Determine number of components
    if n_components is None:
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumsum >= variance_threshold) + 1)
    
    # Final PCA with n_components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame with components
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_columns, index=df_numeric.index)
    
    # Results
    return PCAResults(
        n_components=n_components,
        explained_variance=pca.explained_variance_.tolist(),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_).tolist(),
        components=pca.components_,
        feature_names=numeric_cols,
        transformed_data=df_pca
    )


def get_feature_importance_pca(
    pca_results: PCAResults, 
    n_components: int = 3
) -> pd.DataFrame:
    """Get feature importance in first N components.
    
    Args:
        pca_results: PCAResults object
        n_components: Number of components to consider
        
    Returns:
        DataFrame with feature importance
    """
    n_comp = min(n_components, pca_results.n_components)
    
    # Create DataFrame with loadings
    loadings_df = pd.DataFrame(
        pca_results.components[:n_comp, :],
        columns=pca_results.feature_names,
        index=[f'PC{i+1}' for i in range(n_comp)]
    ).T
    
    # Calculate total importance (sum of squared loadings)
    loadings_df['importance'] = np.sqrt((loadings_df ** 2).sum(axis=1))
    loadings_df = loadings_df.sort_values('importance', ascending=False)
    
    return loadings_df
