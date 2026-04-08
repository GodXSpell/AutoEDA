import pytest
import pandas as pd
import numpy as np
from quick_eda.classifier import classify_dataframe
from quick_eda.profiler import (
    profile_numeric, profile_categorical,
    profile_datetime, profile_dataframe,
    get_dataset_stats
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age":      np.random.normal(35, 10, 200).clip(18, 80).astype(int),
        "income":   np.random.exponential(50000, 200),
        "status":   np.random.choice(["active", "inactive"], 200),
        "score":    np.random.normal(70, 15, 200),
        "signup":   pd.date_range("2020-01-01", periods=200, freq="D"),
    })


# ── profile_numeric ────────────────────────────────────────────────────────

def test_profile_numeric_keys():
    s      = pd.Series(np.random.normal(0, 1, 100))
    result = profile_numeric(s)
    expected_keys = [
        "count", "missing", "missing_pct", "mean", "median",
        "std", "min", "max", "skew", "kurtosis",
        "outlier_count", "outlier_pct", "zeros_pct", "unique_count"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

def test_profile_numeric_missing_count():
    s      = pd.Series([1.0, 2.0, None, 4.0, None])
    result = profile_numeric(s)
    assert result["missing"]     == 2
    assert result["missing_pct"] == 40.0

def test_profile_numeric_outlier_detection():
    # inject obvious outliers
    s      = pd.Series([10] * 95 + [10000, 20000, 30000, 40000, 50000])
    result = profile_numeric(s)
    assert result["outlier_count"] > 0

def test_profile_numeric_no_nulls():
    s      = pd.Series([1, 2, 3, 4, 5])
    result = profile_numeric(s)
    assert result["missing"]     == 0
    assert result["missing_pct"] == 0.0


# ── profile_categorical ────────────────────────────────────────────────────

def test_profile_categorical_keys():
    s      = pd.Series(["a", "b", "a", "c", "b", "a"])
    result = profile_categorical(s)
    assert "count"        in result
    assert "missing"      in result
    assert "missing_pct"  in result
    assert "unique_count" in result
    assert "top_values"   in result
    assert "imbalance_ratio" in result
    assert "rare_category_pct" in result
    assert "entropy"      in result

def test_profile_categorical_top_values():
    s      = pd.Series(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    result = profile_categorical(s)
    assert "a" in result["top_values"]
    assert result["top_values"]["a"] == 60.0

def test_profile_categorical_missing():
    s      = pd.Series(["a", "b", None, "a", None])
    result = profile_categorical(s)
    assert result["missing"]     == 2
    assert result["missing_pct"] == 40.0


# ── profile_datetime ───────────────────────────────────────────────────────

def test_profile_datetime_keys():
    s      = pd.Series(pd.date_range("2024-01-01", periods=50))
    result = profile_datetime(s)
    assert "min_date"     in result
    assert "max_date"     in result
    assert "range_days"   in result
    assert "is_monotonic" in result

def test_profile_datetime_range():
    s      = pd.Series(pd.date_range("2024-01-01", periods=365))
    result = profile_datetime(s)
    assert result["range_days"] == 364
    assert result["is_monotonic"] is True


# ── profile_dataframe ──────────────────────────────────────────────────────

def test_profile_dataframe_returns_all_cols(sample_df):
    col_types = classify_dataframe(sample_df)
    profiles  = profile_dataframe(sample_df, col_types)
    assert set(profiles.keys()) == set(sample_df.columns)

def test_profile_dataframe_no_error(sample_df):
    col_types = classify_dataframe(sample_df)
    profiles  = profile_dataframe(sample_df, col_types)
    assert isinstance(profiles, dict)


# ── get_dataset_stats ──────────────────────────────────────────────────────

def test_get_dataset_stats_keys(sample_df):
    stats = get_dataset_stats(sample_df)
    expected = [
        "rows", "cols", "total_missing_pct",
        "duplicate_rows", "memory_mb",
        "numeric_cols", "categorical_cols", "datetime_cols"
    ]
    for key in expected:
        assert key in stats, f"Missing key: {key}"

def test_get_dataset_stats_shape(sample_df):
    stats = get_dataset_stats(sample_df)
    assert stats["rows"] == 200
    assert stats["cols"] == 5

def test_get_dataset_stats_duplicates():
    df    = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    stats = get_dataset_stats(df)
    assert stats["duplicate_rows"] == 1