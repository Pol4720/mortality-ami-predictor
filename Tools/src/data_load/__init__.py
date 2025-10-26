"""Data loading and I/O module.

This module provides utilities for:
- Loading datasets from multiple formats (CSV, Excel, Parquet, Feather)
- Train/test splitting with various strategies
- Dataset metadata and information
- Data auditing and summarization
"""

from .loaders import (
    load_dataset,
    get_dataset_info,
    DatasetInfo,
    summarize_dataframe,
    select_feature_target,
    data_audit,
)
from .splitters import train_test_split, create_temporal_split, create_stratified_split
from .io_utils import detect_file_format, save_dataset, read_csv_with_encoding

__all__ = [
    # Loaders
    "load_dataset",
    "get_dataset_info",
    "DatasetInfo",
    "summarize_dataframe",
    "select_feature_target",
    "data_audit",
    # Splitters
    "train_test_split",
    "create_temporal_split",
    "create_stratified_split",
    # IO Utils
    "detect_file_format",
    "save_dataset",
    "read_csv_with_encoding",
]
