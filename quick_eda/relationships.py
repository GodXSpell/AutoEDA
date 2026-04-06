import pandas as pd
import numpy as np
from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE,
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


def correlate_with_target(df: pd.DataFrame, target: str, col_types: dict) -> list:
    """
    Rank all numeric columns by their correlation to the target column.
    Returns list of (col, correlation_value) sorted by abs value desc.
    Only runs if user passed target= parameter.
    """
    numeric_cols = [
        col for col, t in col_types.items()
        if t in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE)
        and col != target
    ]

    if not numeric_cols:
        return []

    results = []
    for col in numeric_cols:
        val = df[col].corr(df[target])
        if pd.isna(val):
            continue
        results.append((col, round(float(val), 4)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
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