import pandas as pd
import numpy as np
import scipy.stats as stats
from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE,
    CATEGORICAL_LOW, CATEGORICAL_HIGH, BOOLEAN,
    CONSTANT, NEAR_CONSTANT, ID_LIKE
)

"""
    This module contains functions for profiling columns of a DataFrame based on their classified types.
    It computes summary statistics for numeric and datetime columns, and identifies relationships like correlations and duplicate columns
"""
def compute_correlations(df: pd.DataFrame, col_types: dict, threshold: float = 0.85) -> list:
    """
    Find pairs of numeric columns with high correlation.
    Returns list of (col_a, col_b, correlation_value) sorted by abs value desc.
    """
    # only run on numeric columns that are worth analyzing
    numeric_cols = [
        col for col, t in col_types.items()
        if t in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE)
    ]

    if len(numeric_cols) < 2:
        return []

    corr_matrix = df[numeric_cols].corr()

    pairs = []
    cols  = corr_matrix.columns.tolist()

    # only look at lower triangle to avoid duplicate pairs
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 4)))

    # sort by absolute correlation descending
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def _cramers_v(cat_series1, cat_series2):
    confusion_matrix = pd.crosstab(cat_series1, cat_series2)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0.0

def _get_correlation_strength(corr_val):
    abs_val = abs(corr_val)
    if abs_val < 0.01:
        return "near zero"
    elif abs_val < 0.3:
        return "weak"
    elif abs_val < 0.7:
        return "moderate"
    else:
        return "strong"

def correlate_with_target(df: pd.DataFrame, target: str, col_types: dict, target_type: str = "regression") -> list:
    """
    Rank all numeric and categorical columns by their correlation to the target column.
    Returns list of (col, correlation_value, direction, strength).
    """
    results = []

    for col, t in col_types.items():
        if col == target:
            continue
            
        try:
            val = None
            direction = "N/A"
            
            valid_mask = df[col].notna() & df[target].notna()
            series_col = df.loc[valid_mask, col]
            series_tgt = df.loc[valid_mask, target]
            
            if len(series_col) == 0:
                continue

            num_cats_target = series_tgt.nunique() if target_type == "classification" else None
            num_cats_col = series_col.nunique()

            if target_type == "regression": 
                if t in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE):
                    val = stats.pearsonr(series_col, series_tgt)[0]
                    direction = "positive" if val >= 0 else "negative"
                elif t in (CATEGORICAL_LOW, CATEGORICAL_HIGH):
                    # For categorical feature to numeric target, Cramers doesn't apply.
                    # Actually spec says: 
                    #   - For categorical target: use point-biserial correlation for numeric columns
                    #   - For categorical columns vs target: use Cramér's V statistic
                    # Thus if target=regression, we just do feature vs target using pearson for nums
                    continue

            elif target_type == "classification":
                if t in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE):
                    if num_cats_target == 2:
                        binary_tgt = pd.factorize(series_tgt)[0]
                        val = stats.pointbiserialr(binary_tgt, series_col)[0]
                        direction = "positive" if val >= 0 else "negative"
                    else:
                        continue 
                elif t in (CATEGORICAL_LOW, CATEGORICAL_HIGH):
                    val = _cramers_v(series_col, series_tgt)
                    direction = "N/A"

            if val is not None and not pd.isna(val):
                strength = _get_correlation_strength(val)
                # Keep absolute value but record direction separately
                results.append((col, round(float(abs(val)), 4), direction, strength))
                
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def detect_duplicate_columns(df: pd.DataFrame) -> list:
    """
    Find columns that have identical values to another column.
    Returns list of (col_a, col_b) pairs.
    """
    duplicates = []
    cols       = df.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]

            # compare values — handle nulls properly
            try:
                if df[col_a].equals(df[col_b]):
                    duplicates.append((col_a, col_b))
            except Exception:
                continue

    return duplicates