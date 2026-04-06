import pandas as pd
import numpy as np

"""
Classifier for determining column types in a DataFrame.
Rules are checked in priority order — first match wins.
"""

# ── Column type constants ──────────────────────────────────────────────────
NUMERIC_CONTINUOUS = "NUMERIC_CONTINUOUS"
NUMERIC_DISCRETE   = "NUMERIC_DISCRETE"
CATEGORICAL_LOW    = "CATEGORICAL_LOW"
CATEGORICAL_HIGH   = "CATEGORICAL_HIGH"
DATETIME           = "DATETIME"
BOOLEAN            = "BOOLEAN"
CONSTANT           = "CONSTANT"
NEAR_CONSTANT      = "NEAR_CONSTANT"
ID_LIKE            = "ID_LIKE"


def _is_string_dtype(series: pd.Series) -> bool:
    """Check if a series holds string/object data across pandas 2 and 3."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _looks_like_datetime(series: pd.Series) -> bool:
    """Try parsing first 50 non-null values as dates. Returns True if it works."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.iloc[:50]
    try:
        parsed = pd.to_datetime(sample, format="mixed", errors="raise")
        # extra check — make sure it's not just integers being parsed as years
        if pd.api.types.is_integer_dtype(series):
            return False
        return True
    except Exception:
        return False


def classify_column(series: pd.Series) -> str:
    """
    Classify a single column into one of the type constants.
    Rules are checked in priority order — first match wins.
    """
    non_null    = series.dropna()
    n_total     = len(series)
    n_unique    = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0

    # ── Priority 1: constant ──────────────────────────────────────────────
    if n_unique <= 1:
        return CONSTANT

    # ── Priority 2: near-constant ─────────────────────────────────────────
    top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
    if top_freq >= 0.98:
        return NEAR_CONSTANT

    # ── Priority 3: datetime ──────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME

    # string columns — try datetime parse BEFORE checking ID_LIKE
    if _is_string_dtype(series) and _looks_like_datetime(series):
        return DATETIME

    # ── Priority 4: ID-like ───────────────────────────────────────────────
    if unique_ratio > 0.95 and (pd.api.types.is_integer_dtype(series) or _is_string_dtype(series)):
        return ID_LIKE

    # ── Priority 5: boolean ───────────────────────────────────────────────
    if series.dtype == bool:
        return BOOLEAN

    if pd.api.types.is_integer_dtype(series) and set(non_null.unique()).issubset({0, 1}):
        return BOOLEAN

    # ── Priority 6 & 7: numeric ───────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        if n_unique < 20:
            return NUMERIC_DISCRETE
        return NUMERIC_CONTINUOUS

    # ── Priority 8 & 9: categorical ───────────────────────────────────────
    if n_unique <= 15:
        return CATEGORICAL_LOW

    return CATEGORICAL_HIGH


def classify_dataframe(df: pd.DataFrame) -> dict:
    """
    Classify every column in the dataframe.
    Returns {col_name: type_string}
    """
    return {col: classify_column(df[col]) for col in df.columns}