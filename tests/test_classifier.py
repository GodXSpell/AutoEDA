import pytest
import pandas as pd
import numpy as np
from quick_eda.classifier import (
    classify_column, classify_dataframe,
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE,
    CATEGORICAL_LOW, CATEGORICAL_HIGH,
    DATETIME, BOOLEAN, CONSTANT,
    NEAR_CONSTANT, ID_LIKE
)


# ── classify_column ────────────────────────────────────────────────────────

def test_constant_single_value():
    s = pd.Series([1, 1, 1, 1, 1])
    assert classify_column(s) == CONSTANT

def test_constant_all_null():
    s = pd.Series([None, None, None])
    assert classify_column(s) == CONSTANT

def test_near_constant():
    s = pd.Series(["active"] * 98 + ["inactive"] * 2)
    assert classify_column(s) == NEAR_CONSTANT

def test_id_like_integer():
    s = pd.Series(range(1000))
    assert classify_column(s) == ID_LIKE

def test_id_like_string():
    s = pd.Series([f"user_{i}" for i in range(1000)])
    assert classify_column(s) == ID_LIKE

def test_datetime_dtype():
    s = pd.Series(pd.date_range("2024-01-01", periods=100))
    assert classify_column(s) == DATETIME

def test_datetime_string():
    s = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"] * 10)
    assert classify_column(s) == DATETIME

def test_boolean_dtype():
    s = pd.Series([True, False, True, False])
    assert classify_column(s) == BOOLEAN

def test_boolean_zero_one():
    s = pd.Series([0, 1, 0, 1, 0, 1] * 10)
    assert classify_column(s) == BOOLEAN

def test_numeric_continuous():
    s = pd.Series(np.random.normal(50, 10, 500))
    assert classify_column(s) == NUMERIC_CONTINUOUS

def test_numeric_discrete():
    s = pd.Series([1, 2, 3, 4, 5] * 100)
    assert classify_column(s) == NUMERIC_DISCRETE

def test_categorical_low():
    s = pd.Series(["cat", "dog", "fish"] * 100)
    assert classify_column(s) == CATEGORICAL_LOW

def test_categorical_high():
    s = pd.Series([f"city_{i}" for i in range(50)] * 5)
    assert classify_column(s) == CATEGORICAL_HIGH


# ── classify_dataframe ─────────────────────────────────────────────────────

def test_classify_dataframe_returns_dict():
    df = pd.DataFrame({
        "id":     range(100),
        "age":    np.random.randint(18, 80, 100),
        "status": ["active"] * 100,
    })
    result = classify_dataframe(df)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(df.columns)

def test_classify_dataframe_correct_types():
    df = pd.DataFrame({
        "id":       range(1000),
        "income":   np.random.normal(50000, 10000, 1000),
        "category": ["A", "B", "C"] * 333 + ["A"],
    })
    result = classify_dataframe(df)
    assert result["id"]       == ID_LIKE
    assert result["income"]   == NUMERIC_CONTINUOUS
    assert result["category"] == CATEGORICAL_LOW

def test_classify_dataframe_no_error_on_mixed_df():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "c": [True, False, True],
        "d": pd.date_range("2024-01-01", periods=3),
    })
    result = classify_dataframe(df)
    assert len(result) == 4