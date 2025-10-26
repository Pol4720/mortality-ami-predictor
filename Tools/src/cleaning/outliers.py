"""Outlier detection and treatment strategies.

This module provides methods for detecting and treating outliers
in numerical data using statistical methods.
"""
from __future__ import annotations

from enum import Enum
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats


class OutlierMethod(str, Enum):
    """Methods for outlier detection."""
    
    IQR = "iqr"  # Interquartile Range
    ZSCORE = "zscore"  # Z-score
    NONE = "none"


class OutlierTreatment(str, Enum):
    """Strategies for treating detected outliers."""
    
    CAP = "cap"  # Cap at threshold (Winsorization)
    REMOVE = "remove"  # Remove outliers (mark as NaN)
    NONE = "none"  # Detect but don't treat


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5,
) -> Tuple[pd.Series, float, float]:
    """Detect outliers using Interquartile Range method.
    
    Args:
        series: Input numeric series
        multiplier: IQR multiplier (typically 1.5 or 3.0)
        
    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return pd.Series(False, index=series.index), 0.0, 0.0
    
    Q1 = clean_data.quantile(0.25)
    Q3 = clean_data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    
    return outlier_mask, float(lower_bound), float(upper_bound)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using Z-score method.
    
    Args:
        series: Input numeric series
        threshold: Z-score threshold (typically 3.0)
        
    Returns:
        Boolean series indicating outliers
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return pd.Series(False, index=series.index)
    
    z_scores = np.abs(stats.zscore(clean_data, nan_policy='omit'))
    
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[clean_data.index] = z_scores > threshold
    
    return outlier_mask


def treat_outliers(
    series: pd.Series,
    outlier_mask: pd.Series,
    treatment: OutlierTreatment,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> pd.Series:
    """Treat detected outliers using specified strategy.
    
    Args:
        series: Input series
        outlier_mask: Boolean series indicating outliers
        treatment: Treatment strategy
        lower_bound: Lower bound for capping
        upper_bound: Upper bound for capping
        
    Returns:
        Series with treated outliers
    """
    if treatment == OutlierTreatment.NONE:
        return series
    
    result = series.copy()
    
    if treatment == OutlierTreatment.CAP:
        if lower_bound is not None:
            result.loc[outlier_mask & (result < lower_bound)] = lower_bound
        if upper_bound is not None:
            result.loc[outlier_mask & (result > upper_bound)] = upper_bound
    
    elif treatment == OutlierTreatment.REMOVE:
        result.loc[outlier_mask] = np.nan
    
    return result


def handle_column_outliers(
    series: pd.Series,
    method: OutlierMethod = OutlierMethod.IQR,
    treatment: OutlierTreatment = OutlierTreatment.CAP,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> Tuple[pd.Series, int]:
    """Detect and treat outliers in a numeric column.
    
    Args:
        series: Input numeric series
        method: Detection method
        treatment: Treatment strategy
        iqr_multiplier: Multiplier for IQR method
        zscore_threshold: Threshold for Z-score method
        
    Returns:
        Tuple of (treated_series, outlier_count)
    """
    if method == OutlierMethod.NONE:
        return series, 0
    
    # Detect outliers
    if method == OutlierMethod.IQR:
        outlier_mask, lower, upper = detect_outliers_iqr(series, iqr_multiplier)
        result = treat_outliers(series, outlier_mask, treatment, lower, upper)
    
    elif method == OutlierMethod.ZSCORE:
        outlier_mask = detect_outliers_zscore(series, zscore_threshold)
        # For z-score, use percentiles as bounds
        lower = series.quantile(0.01) if treatment == OutlierTreatment.CAP else None
        upper = series.quantile(0.99) if treatment == OutlierTreatment.CAP else None
        result = treat_outliers(series, outlier_mask, treatment, lower, upper)
    
    else:
        return series, 0
    
    outlier_count = int(outlier_mask.sum())
    
    return result, outlier_count
