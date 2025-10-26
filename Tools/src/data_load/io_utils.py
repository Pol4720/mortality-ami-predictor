"""I/O utilities for data operations.

This module provides utilities for file format detection,
dataset saving, and other I/O operations.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def detect_file_format(path: str) -> str:
    """Detect file format from extension.
    
    Args:
        path: File path
        
    Returns:
        Format string: 'csv', 'parquet', 'feather', 'xlsx', 'xls', or 'unknown'
    """
    ext = os.path.splitext(path)[1].lower()
    
    format_map = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".feather": "feather",
        ".xlsx": "xlsx",
        ".xls": "xls",
    }
    
    return format_map.get(ext, "unknown")


def save_dataset(
    df: pd.DataFrame,
    path: str,
    format: Optional[str] = None,
    **kwargs
) -> None:
    """Save DataFrame to file with format auto-detection.
    
    Args:
        df: DataFrame to save
        path: Output file path
        format: Format to use ('csv', 'parquet', 'feather', 'xlsx').
               If None, infers from file extension
        **kwargs: Additional arguments passed to pandas save method
    """
    if format is None:
        format = detect_file_format(path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if format == "csv":
        df.to_csv(path, index=False, **kwargs)
    elif format == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif format == "feather":
        df.to_feather(path, **kwargs)
    elif format == "xlsx":
        df.to_excel(path, index=False, engine="openpyxl", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
