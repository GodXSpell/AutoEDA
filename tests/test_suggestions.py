import pytest
import pandas as pd
import numpy as np
from quick_eda.classifier import (
    NUMERIC_CONTINUOUS, CATEGORICAL_LOW,
    CONSTANT, NEAR_CONSTANT, ID_LIKE, CATEGORICAL_HIGH
)
from quick_eda.suggestions import (
    suggest_for_column, suggest_for_dataframe,
    get_global_suggestions
)


# ── suggest_for_column ─────────────────────────────────────────────────────

def test_constant_gets_drop():
    result = suggest_for_column("col", CONSTANT, {}, [])
    assert any("Drop" in s for s in result)

def test_constant_returns_early():
    # constant columns should only get one suggestion
    result = suggest_for_column("col", CONSTANT, {"missing_pct": 50, "skew": 3}, [])
    assert len(result) == 1

def test_near_constant_gets_drop():
    result = suggest_for_column("col", NEAR_CONSTANT, {"missing_pct": 0}, [])
    assert any("dropping" in s.lower() for s in result)

def test_id_like_gets_drop():
    result = suggest_for_column("col", ID_LIKE, {}, [])
    assert any("identifier" in s.lower() for s in result)

def test_high_missing_suggest_drop():
    profile = {"missing_pct": 60}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert any("dropping" in s.lower() for s in result)

def test_moderate_missing_suggest_impute():
    profile = {"missing_pct": 30, "skew": 0, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert any("impute" in s.lower() for s in result)

def test_low_missing_suggest_impute_median():
    profile = {"missing_pct": 10, "skew": 0, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert any("median" in s.lower() for s in result)

def test_high_skew_suggest_transform():
    profile = {"missing_pct": 0, "skew": 2.5, "min": 0.5, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert any("log transform" in s.lower() for s in result)

def test_high_skew_negative_values():
    profile = {"missing_pct": 0, "skew": 2.5, "min": -1.0, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert any("power transform" in s.lower() for s in result)
    # The actual string returned is: "Consider power transform (Box-Cox) — column has negative values, log transform not applicable"
    # Since "log transform" is IN the power transform message, asserting not any("log transform" in s.lower()) will fail.
    # So we don't assert that.
    assert not any(s.startswith("Apply log transform") for s in result)

def test_drop_suppresses_other_suggestions():
    # near constant with high missing and skew etc should only get drop
    profile = {"missing_pct": 60, "skew": 5.0, "min": 0, "outlier_pct": 20, "unique_count": 1}
    result_nr = suggest_for_column("col", NEAR_CONSTANT, profile, [])
    assert len(result_nr) == 1
    assert any("dropping" in s.lower() for s in result_nr)

    result_id = suggest_for_column("col", ID_LIKE, profile, [])
    assert len(result_id) == 1
    assert any("identifier" in s.lower() for s in result_id)

    # high missing > 50 also suppresses others now
    result_miss = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert len(result_miss) == 1
    assert any("dropping" in s.lower() for s in result_miss)

def test_high_outliers_suggest_investigate():
    profile = {"missing_pct": 0, "skew": 0, "outlier_pct": 12.0, "zeros_pct": 0, "kurtosis": 0}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert any("outlier" in s.lower() for s in result)

def test_high_correlation_suggestion():
    profile = {"missing_pct": 0, "skew": 0, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0}
    corr_flags = [("col", "other_col", 0.92)]
    result = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, corr_flags)
    assert any("correlation" in s.lower() for s in result)

def test_clean_column_no_suggestions():
    profile = {"missing_pct": 0, "skew": 0.1, "outlier_pct": 1.0, "zeros_pct": 0, "kurtosis": 1}
    result  = suggest_for_column("col", NUMERIC_CONTINUOUS, profile, [])
    assert result == []

def test_high_cardinality_suggestion():
    profile = {"missing_pct": 0, "unique_count": 200}
    result  = suggest_for_column("col", CATEGORICAL_HIGH, profile, [])
    assert any("cardinality" in s.lower() for s in result)


# ── suggest_for_dataframe ──────────────────────────────────────────────────

def test_suggest_for_dataframe_only_flagged_cols():
    profiles = {
        "clean_col": {"missing_pct": 0, "skew": 0, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0},
        "bad_col":   {"missing_pct": 60},
    }
    col_types = {
        "clean_col": NUMERIC_CONTINUOUS,
        "bad_col":   NUMERIC_CONTINUOUS,
    }
    result = suggest_for_dataframe(profiles, col_types, [])
    assert "bad_col"   in result
    assert "clean_col" not in result

def test_suggest_for_dataframe_returns_dict():
    profiles  = {"col": {"missing_pct": 0, "skew": 0, "outlier_pct": 0, "zeros_pct": 0, "kurtosis": 0}}
    col_types = {"col": NUMERIC_CONTINUOUS}
    result    = suggest_for_dataframe(profiles, col_types, [])
    assert isinstance(result, dict)


# ── get_global_suggestions ─────────────────────────────────────────────────

def test_global_duplicate_rows():
    stats  = {"duplicate_rows": 5, "total_missing_pct": 0, "rows": 500}
    result = get_global_suggestions(stats)
    assert any("duplicate" in s.lower() for s in result)

def test_global_high_missing():
    stats  = {"duplicate_rows": 0, "total_missing_pct": 35, "rows": 500}
    result = get_global_suggestions(stats)
    assert any("missingness" in s.lower() for s in result)

def test_global_small_dataset():
    stats  = {"duplicate_rows": 0, "total_missing_pct": 0, "rows": 50}
    result = get_global_suggestions(stats)
    assert any("small" in s.lower() for s in result)

def test_global_clean_dataset():
    stats  = {"duplicate_rows": 0, "total_missing_pct": 5, "rows": 500}
    result = get_global_suggestions(stats)
    assert result == []