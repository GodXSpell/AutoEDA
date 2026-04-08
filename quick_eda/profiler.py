import pandas as pd
import numpy as np
from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE, CATEGORICAL_LOW,
    CATEGORICAL_HIGH, DATETIME, BOOLEAN, CONSTANT,
    NEAR_CONSTANT, ID_LIKE
)
"""
    The work of a profiler is to take a DataFrame and produce a profile for each column.
    By profile, we mean a dictionary of summary statistics and insights relevant to that column's type.
    The profiler will use the column type classification from classifier.py to determine which statistics to compute.
    For example:
    - For NUMERIC_CONTINUOUS, we might compute mean, median, std, min, max, skewness, kurtosis, and percentiles.
    - For CATEGORICAL_LOW, we might compute value counts, top categories, and their frequencies.
    - For DATETIME, we might compute min, max, range, and common time units (e.g., most common  month or day of week).
    - For BOOLEAN, we might compute the proportion of True vs False.
    - For CONSTANT, we might just report the single unique value and its frequency.
    The profiler should be designed to be efficient and handle large datasets gracefully.
"""
def profile_numeric(series: pd.Series) -> dict:
    """
    Compute numeric column profiles
    """
    non_null = series.dropna()
    n_total  = len(series)

    q1  = non_null.quantile(0.25)        # Q1 and Q3 are used to calculate the IQR, which helps identify outliers in the data
    q3  = non_null.quantile(0.75)        # IQR-based outlier detection is more robust than using mean and std, especially for skewed distributions
    iqr = q3 - q1                        # IQR is used to identify outliers using the 1.5 * IQR rule

    outlier_mask  = (non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)
    outlier_count = int(outlier_mask.sum())

    return {
        "count":        int(non_null.count()),
        "missing":      int(series.isnull().sum()),
        "missing_pct":  round(series.isnull().sum() / n_total * 100, 1),
        "mean":         round(float(non_null.mean()), 4),
        "median":       round(float(non_null.median()), 4),
        "std":          round(float(non_null.std()), 4),
        "min":          round(float(non_null.min()), 4),
        "max":          round(float(non_null.max()), 4),
        "skew":         round(float(non_null.skew()), 4),
        "kurtosis":     round(float(non_null.kurtosis()), 4),
        "outlier_count": outlier_count,
        "outlier_pct":  round(outlier_count / len(non_null) * 100, 1) if len(non_null) > 0 else 0,
        "zeros_pct":    round((non_null == 0).sum() / len(non_null) * 100, 1) if len(non_null) > 0 else 0,
        "unique_count": int(series.nunique()),
    }

def profile_categorical(series: pd.Series) -> dict:
    """
    Compute stats for a categorical or boolean column.
    """
    n_total   = len(series)
    
    value_counts_normalize = series.value_counts(normalize=True, dropna=True)
    top_values = (
        value_counts_normalize
        .head(5)
        .mul(100)
        .round(1)
        .to_dict()
    )
    
    if not value_counts_normalize.empty:
        imbalance_ratio = round(float(value_counts_normalize.iloc[0] / value_counts_normalize.iloc[-1]), 1)
        rare_category_pct = round(float(value_counts_normalize[value_counts_normalize < 0.01].sum() * 100), 1)
        
        # Compute Shannon entropy
        entropy = -np.sum(value_counts_normalize * np.log2(value_counts_normalize + 1e-9))
        entropy = round(float(entropy), 2)
    else:
        imbalance_ratio = 0.0
        rare_category_pct = 0.0
        entropy = 0.0

    return {
        "count":             int(series.count()),
        "missing":           int(series.isnull().sum()),
        "missing_pct":       round(series.isnull().sum() / n_total * 100, 1) if n_total > 0 else 0.0,
        "unique_count":      int(series.nunique()),
        "top_values":        top_values,   # {value: pct_frequency}
        "imbalance_ratio":   imbalance_ratio,
        "rare_category_pct": rare_category_pct,
        "entropy":           entropy,
    }


def profile_datetime(series: pd.Series) -> dict:
    """
    Compute stats for a datetime column.
    """
    n_total  = len(series)
    non_null = series.dropna()

    # always parse to datetime regardless of stored dtype
    try:
        parsed = pd.to_datetime(non_null, format="mixed", errors="raise")
    except Exception:
        try:
            parsed = pd.to_datetime(non_null, format="mixed", errors="coerce")
            parsed = parsed.dropna()
        except Exception:
            return {"error": "Could not parse as datetime"}

    if len(parsed) == 0:
        return {"error": "No valid dates found"}

    return {
        "count":        int(parsed.count()),
        "missing":      int(series.isnull().sum()),
        "missing_pct":  round(series.isnull().sum() / n_total * 100, 1),
        "min_date":     str(parsed.min()),
        "max_date":     str(parsed.max()),
        "range_days":   (parsed.max() - parsed.min()).days,
        "is_monotonic": bool(parsed.is_monotonic_increasing),
    }


def profile_dataframe(df: pd.DataFrame, col_types: dict) -> dict:
    """
    Profile every column based on its classified type.
    Returns {col_name: profile_dict}
    """
    profiles = {}

    for col, col_type in col_types.items():

        # skip useless columns — just note the type, no deep profiling
        if col_type in (CONSTANT, NEAR_CONSTANT, ID_LIKE):
            profiles[col] = {
                "skipped": True,
                "reason":  col_type,
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 1),
                "unique_count": int(df[col].nunique()),
            }
            continue

        if col_type in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE):
            profiles[col] = profile_numeric(df[col])

        elif col_type in (CATEGORICAL_LOW, CATEGORICAL_HIGH, BOOLEAN):
            profiles[col] = profile_categorical(df[col])

        elif col_type == DATETIME:
            profiles[col] = profile_datetime(df[col])

        else:
            # fallback — shouldn't happen but handle gracefully
            profiles[col] = {"skipped": True, "reason": "unknown_type"}

    return profiles


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Dataset-level summary stats.
    These go in the top banner of the report.
    """
    n_rows, n_cols = df.shape
    total_cells    = n_rows * n_cols
    total_missing  = df.isnull().sum().sum()

    return {
        "rows":            n_rows,
        "cols":            n_cols,
        "total_missing_pct": round(total_missing / total_cells * 100, 1),
        "duplicate_rows":  int(df.duplicated().sum()),
        "memory_mb":       round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "numeric_cols":    int((df.select_dtypes(include="number")).shape[1]),
        "categorical_cols": int((df.select_dtypes(include=["str", "string", "object", "category"])).shape[1]),
        "datetime_cols":   int((df.select_dtypes(include="datetime")).shape[1]),
    }