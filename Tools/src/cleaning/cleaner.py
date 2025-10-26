"""Main data cleaning orchestration class.

This module provides the DataCleaner class that orchestrates all cleaning
operations and tracks metadata throughout the process.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Tuple

import pandas as pd

from .metadata import VariableMetadata
from .imputation import (
    ImputationStrategy,
    impute_numeric_column,
    impute_categorical_column,
    impute_knn,
)
from .outliers import (
    OutlierMethod,
    OutlierTreatment,
    handle_column_outliers,
)
from .encoding import (
    EncodingStrategy,
    encode_categorical_column,
)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning pipeline.
    
    Attributes:
        numeric_imputation: Strategy for numeric imputation
        categorical_imputation: Strategy for categorical imputation
        knn_neighbors: Number of neighbors for KNN imputation
        constant_fill_numeric: Fill value for constant numeric imputation
        constant_fill_categorical: Fill value for constant categorical imputation
        outlier_method: Method for outlier detection
        iqr_multiplier: Multiplier for IQR method
        zscore_threshold: Threshold for Z-score method
        outlier_treatment: How to treat detected outliers
        categorical_encoding: Encoding strategy for categoricals
        ordinal_categories: Category orders for ordinal encoding
        discretization_strategy: Strategy for discretizing numeric variables
        discretization_bins: Number of bins for discretization
        custom_bins: Custom bin edges per variable
        custom_labels: Custom labels for bins per variable
        drop_duplicates: Whether to drop duplicate rows
        drop_fully_missing: Whether to drop fully missing columns
        drop_constant: Whether to drop constant columns
        constant_threshold: Threshold for considering column constant
    """
    # Imputation
    numeric_imputation: str = "median"
    categorical_imputation: str = "mode"
    knn_neighbors: int = 5
    constant_fill_numeric: float = 0.0
    constant_fill_categorical: str = "missing"
    
    # Outliers
    outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    outlier_treatment: str = "cap"
    
    # Encoding
    categorical_encoding: str = "label"
    ordinal_categories: Dict[str, List[str]] = field(default_factory=dict)
    
    # Discretization
    discretization_strategy: str = "none"
    discretization_bins: int = 5
    custom_bins: Dict[str, List[float]] = field(default_factory=dict)
    custom_labels: Dict[str, List[str]] = field(default_factory=dict)
    
    # General
    drop_duplicates: bool = True
    drop_fully_missing: bool = True
    drop_constant: bool = True
    constant_threshold: float = 0.95
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return asdict(self)


class DataCleaner:
    """Main data cleaning pipeline with metadata tracking.
    
    This class orchestrates all cleaning operations and maintains
    detailed metadata about transformations applied to each variable.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """Initialize cleaner with configuration.
        
        Args:
            config: Cleaning configuration. If None, uses defaults
        """
        self.config = config or CleaningConfig()
        self.metadata: Dict[str, VariableMetadata] = {}
        self.encoders: Dict[str, Any] = {}
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply complete cleaning pipeline to DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Target column to preserve (not cleaned)
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Separate target if exists
        target = None
        if target_column and target_column in df_clean.columns:
            target = df_clean[target_column].copy()
            df_clean = df_clean.drop(columns=[target_column])
        
        # 1. Identify column types
        numeric_cols, categorical_cols = self._identify_column_types(df_clean)
        
        # 2. Initialize metadata
        self._initialize_metadata(df_clean, numeric_cols, categorical_cols)
        
        # 3. Drop problematic columns
        df_clean = self._drop_problematic_columns(df_clean, numeric_cols, categorical_cols)
        
        # 4. Remove duplicates
        if self.config.drop_duplicates:
            df_clean = self._remove_duplicates(df_clean)
        
        # 5. Handle outliers (before imputation)
        df_clean = self._handle_outliers(df_clean, numeric_cols)
        
        # 6. Impute missing values
        df_clean = self._impute_missing(df_clean, numeric_cols, categorical_cols)
        
        # 7. Encode categoricals
        df_clean = self._encode_categorical(df_clean, categorical_cols)
        
        # 8. Validate dtypes
        df_clean = self._validate_dtypes(df_clean)
        
        # 9. Update final metadata
        self._update_final_metadata(df_clean)
        
        # Restore target
        if target is not None:
            df_clean[target_column] = target
        
        return df_clean
    
    def _identify_column_types(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical columns."""
        import numpy as np
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Detect numeric columns that are actually categorical
        for col in numeric_cols[:]:
            if df[col].nunique() <= 10 and df[col].notna().sum() > 0:
                categorical_cols.append(col)
                numeric_cols.remove(col)
        
        return numeric_cols, categorical_cols
    
    def _initialize_metadata(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        """Initialize metadata for each variable."""
        for col in numeric_cols:
            series = df[col]
            self.metadata[col] = VariableMetadata(
                name=col,
                original_type=str(series.dtype),
                is_numerical=True,
                is_categorical=False,
                original_min=float(series.min()) if series.notna().any() else None,
                original_max=float(series.max()) if series.notna().any() else None,
                missing_count_original=int(series.isna().sum()),
                missing_percent_original=float(series.isna().mean() * 100),
            )
        
        for col in categorical_cols:
            series = df[col]
            unique_vals = series.dropna().unique().tolist()
            self.metadata[col] = VariableMetadata(
                name=col,
                original_type=str(series.dtype),
                is_numerical=False,
                is_categorical=True,
                unique_values=[str(v) for v in unique_vals[:50]],
                missing_count_original=int(series.isna().sum()),
                missing_percent_original=float(series.isna().mean() * 100),
            )
    
    def _drop_problematic_columns(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Drop fully missing or constant columns."""
        df_clean = df.copy()
        cols_to_drop = []
        
        for col in df_clean.columns:
            # Fully missing
            if self.config.drop_fully_missing and df_clean[col].isna().all():
                cols_to_drop.append(col)
                if col in self.metadata:
                    self.metadata[col].quality_flags.append("fully_missing")
                continue
            
            # Nearly constant
            if self.config.drop_constant:
                value_counts = df_clean[col].value_counts(dropna=True)
                if len(value_counts) > 0:
                    most_common_ratio = value_counts.iloc[0] / df_clean[col].notna().sum()
                    if most_common_ratio >= self.config.constant_threshold:
                        cols_to_drop.append(col)
                        if col in self.metadata:
                            self.metadata[col].quality_flags.append("nearly_constant")
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            for col in cols_to_drop:
                if col in numeric_cols:
                    numeric_cols.remove(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        n_after = len(df_clean)
        
        if n_after < n_before:
            print(f"⚠️  Removed {n_before - n_after} duplicate rows")
        
        return df_clean
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> pd.DataFrame:
        """Detect and treat outliers in numeric columns."""
        method = OutlierMethod(self.config.outlier_method)
        treatment = OutlierTreatment(self.config.outlier_treatment)
        
        if method == OutlierMethod.NONE:
            return df
        
        df_clean = df.copy()
        
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue
            
            result, n_outliers = handle_column_outliers(
                df_clean[col],
                method=method,
                treatment=treatment,
                iqr_multiplier=self.config.iqr_multiplier,
                zscore_threshold=self.config.zscore_threshold,
            )
            
            df_clean[col] = result
            
            if col in self.metadata:
                self.metadata[col].outliers_detected = n_outliers
                self.metadata[col].outliers_treated = n_outliers
                if n_outliers > 0:
                    self.metadata[col].quality_flags.append(
                        f"outliers_{treatment.value}_{n_outliers}"
                    )
        
        return df_clean
    
    def _impute_missing(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Impute missing values."""
        df_clean = df.copy()
        
        # Handle KNN imputation separately (needs all columns at once)
        if self.config.numeric_imputation == "knn":
            df_clean = impute_knn(df_clean, numeric_cols, self.config.knn_neighbors)
            for col in numeric_cols:
                if col in self.metadata:
                    self.metadata[col].imputation_method = "knn"
        else:
            # Impute numeric columns individually
            for col in numeric_cols:
                if col not in df_clean.columns or df_clean[col].isna().sum() == 0:
                    continue
                
                strategy = ImputationStrategy(self.config.numeric_imputation)
                df_clean[col] = impute_numeric_column(
                    df_clean[col],
                    strategy=strategy,
                    fill_value=self.config.constant_fill_numeric,
                    knn_neighbors=self.config.knn_neighbors,
                )
                
                if col in self.metadata:
                    self.metadata[col].imputation_method = strategy.value
        
        # Impute categorical columns
        for col in categorical_cols:
            if col not in df_clean.columns or df_clean[col].isna().sum() == 0:
                continue
            
            strategy = ImputationStrategy(self.config.categorical_imputation)
            df_clean[col] = impute_categorical_column(
                df_clean[col],
                strategy=strategy,
                fill_value=self.config.constant_fill_categorical,
            )
            
            if col in self.metadata:
                self.metadata[col].imputation_method = strategy.value
        
        return df_clean
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        strategy = EncodingStrategy(self.config.categorical_encoding)
        
        if strategy == EncodingStrategy.NONE:
            return df
        
        df_clean = df.copy()
        
        for col in categorical_cols:
            if col not in df_clean.columns:
                continue
            
            # Convert to string
            df_clean[col] = df_clean[col].astype(str)
            
            ordinal_cats = self.config.ordinal_categories.get(col, None)
            
            df_clean, encoding_info = encode_categorical_column(
                df_clean,
                col,
                strategy,
                ordinal_categories=ordinal_cats,
            )
            
            if col in self.metadata:
                self.metadata[col].encoding_type = encoding_info['type']
                self.metadata[col].encoding_mapping = encoding_info.get('mapping', {})
        
        return df_clean
    
    def _validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct data types."""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except Exception:
                    pass
        
        return df_clean
    
    def _update_final_metadata(self, df: pd.DataFrame) -> None:
        """Update metadata with final information after cleaning."""
        import pandas as pd
        
        for col in df.columns:
            if col not in self.metadata:
                continue
            
            meta = self.metadata[col]
            meta.cleaned_type = str(df[col].dtype)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                meta.cleaned_min = float(df[col].min()) if df[col].notna().any() else None
                meta.cleaned_max = float(df[col].max()) if df[col].notna().any() else None
            
            # Quality flags
            missing_final = df[col].isna().sum()
            if missing_final > 0:
                meta.quality_flags.append(f"missing_values_remain_{missing_final}")
            else:
                meta.quality_flags.append("no_missing_values")
    
    def save_metadata(self, filepath: Path) -> None:
        """Save metadata to JSON file.
        
        Args:
            filepath: Output path for JSON file
        """
        metadata_dict = {
            name: meta.to_dict() 
            for name, meta in self.metadata.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Generate cleaning summary report.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "variables_cleaned": len(self.metadata),
            "variables_with_outliers": sum(
                1 for m in self.metadata.values() if m.outliers_detected > 0
            ),
            "variables_imputed": sum(
                1 for m in self.metadata.values() if m.imputation_method is not None
            ),
            "variables_encoded": sum(
                1 for m in self.metadata.values() if m.encoding_type is not None
            ),
            "quality_issues": {
                name: meta.quality_flags
                for name, meta in self.metadata.items()
                if meta.quality_flags
            }
        }


def quick_clean(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, DataCleaner]:
    """Quick convenience function for data cleaning.
    
    Args:
        df: DataFrame to clean
        target_column: Target column to preserve
        **kwargs: Arguments for CleaningConfig
        
    Returns:
        Tuple of (cleaned_df, cleaner_with_metadata)
    """
    config = CleaningConfig(**kwargs)
    cleaner = DataCleaner(config)
    df_clean = cleaner.fit_transform(df, target_column=target_column)
    
    return df_clean, cleaner
