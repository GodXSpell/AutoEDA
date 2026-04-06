import pytest
import pandas as pd
import numpy as np
from quick_eda import quick_eda


@pytest.fixture
def clean_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age":    np.random.normal(35, 10, 200).clip(18, 80).astype(int),
        "income": np.random.normal(50000, 10000, 200),
        "score":  np.random.normal(70, 15, 200),
        "gender": np.random.choice(["M", "F"], 200),
    })


@pytest.fixture
def messy_df():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "user_id":  range(n),
        "age":      np.random.normal(35, 10, n).astype(int),
        "income":   np.random.exponential(50000, n),
        "status":   ["active"] * 295 + ["inactive"] * 5,
        "constant": [1] * n,
        "category": np.random.choice([f"cat_{i}" for i in range(60)], n),
    })
    # inject missing
    df.loc[np.random.choice(n, 40, replace=False), "income"] = np.nan
    # inject outliers
    df.loc[0, "income"] = 5_000_000
    df.loc[1, "income"] = 4_000_000
    # inject duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


# ── input validation ───────────────────────────────────────────────────────

def test_raises_on_non_dataframe():
    with pytest.raises(TypeError):
        quick_eda([1, 2, 3])

def test_raises_on_empty_dataframe():
    with pytest.raises(ValueError):
        quick_eda(pd.DataFrame())

def test_raises_on_single_column():
    with pytest.raises(ValueError):
        quick_eda(pd.DataFrame({"a": [1, 2, 3]}))

def test_raises_on_invalid_mode(clean_df):
    with pytest.raises(ValueError):
        quick_eda(clean_df, mode="invalid")

def test_raises_on_missing_target(clean_df):
    with pytest.raises(ValueError):
        quick_eda(clean_df, target="nonexistent_col")


# ── return_report ──────────────────────────────────────────────────────────

def test_return_report_is_dict(clean_df):
    report = quick_eda(clean_df, return_report=True, plots=False)
    assert isinstance(report, dict)

def test_return_report_keys(clean_df):
    report = quick_eda(clean_df, return_report=True, plots=False)
    expected_keys = [
        "shape", "col_types", "profiles", "dataset_stats",
        "correlations", "suggestions", "global_suggestions"
    ]
    for key in expected_keys:
        assert key in report, f"Missing key: {key}"

def test_return_none_by_default(clean_df):
    result = quick_eda(clean_df, plots=False)
    assert result is None


# ── modes ──────────────────────────────────────────────────────────────────

def test_tldr_mode_no_error(clean_df):
    quick_eda(clean_df, mode="tldr")

def test_full_mode_no_error(clean_df):
    quick_eda(clean_df, mode="full", plots=False)


# ── messy data ─────────────────────────────────────────────────────────────

def test_messy_df_produces_suggestions(messy_df):
    report = quick_eda(messy_df, return_report=True, plots=False)
    assert len(report["suggestions"]) > 0

def test_messy_df_catches_id_column(messy_df):
    report = quick_eda(messy_df, return_report=True, plots=False)
    assert "user_id" in report["suggestions"]

def test_messy_df_catches_constant(messy_df):
    report = quick_eda(messy_df, return_report=True, plots=False)
    assert "constant" in report["suggestions"]

def test_messy_df_catches_duplicates(messy_df):
    report = quick_eda(messy_df, return_report=True, plots=False)
    assert report["dataset_stats"]["duplicate_rows"] > 0


# ── sampling ───────────────────────────────────────────────────────────────

def test_sampling_triggers_on_large_df(capsys):
    large_df = pd.DataFrame({
        "a": np.random.normal(0, 1, 100_000),
        "b": np.random.normal(0, 1, 100_000),
    })
    quick_eda(large_df, plots=False)
    captured = capsys.readouterr()
    assert "sample" in captured.out.lower()

def test_sampling_false_uses_full_df():
    large_df = pd.DataFrame({
        "a": np.random.normal(0, 1, 60_000),
        "b": np.random.normal(0, 1, 60_000),
    })
    report = quick_eda(large_df, sample=False, return_report=True, plots=False)
    assert report["shape"][0] == 60_000


# ── target parameter ───────────────────────────────────────────────────────

def test_target_param_no_error(clean_df):
    report = quick_eda(clean_df, target="income", return_report=True, plots=False)
    assert "target_correlations" in report

def test_target_correlations_ranked(clean_df):
    report = quick_eda(clean_df, target="income", return_report=True, plots=False)
    tc     = report["target_correlations"]
    if len(tc) > 1:
        assert abs(tc[0][1]) >= abs(tc[1][1])