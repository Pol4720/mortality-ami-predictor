"""Feature utilities like list management and safe selection."""
from __future__ import annotations

from typing import List
import pandas as pd


EXCLUDE_COLS = {"patient_id", "mrn", "admission_id"}


def safe_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    """Return a list of feature columns excluding targets and identifiers if present."""
    cols = [c for c in df.columns if c not in set(target_cols) | EXCLUDE_COLS]
    return cols
