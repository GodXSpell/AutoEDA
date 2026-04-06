# quick_eda — Project TODO

> Build order matters. Work top to bottom. Don't jump ahead.
> Each task should be completable in one focused session.

---

## PHASE 0 — Project Setup

- [x] Create repo on GitHub named `quick-eda`
- [x] Set up folder structure exactly as:
  ```
  quick_eda/
  ├── quick_eda/
  │   ├── __init__.py
  │   ├── core.py
  │   ├── classifier.py
  │   ├── profiler.py
  │   ├── relationships.py
  │   ├── suggestions.py
  │   ├── renderer.py
  │   └── plots.py
  ├── tests/
  │   ├── test_classifier.py
  │   ├── test_profiler.py
  │   ├── test_suggestions.py
  │   └── test_core.py
  ├── examples/
  │   └── demo.ipynb
  ├── setup.py
  ├── requirements.txt
  ├── README.md
  ├── TODO.md
  └── .gitignore
  ```
- [x] Create `setup.py` with name, version `0.1.0`, author, dependencies
- [x] Create `requirements.txt`:
  - `pandas>=1.3`
  - `numpy>=1.21`
  - `matplotlib>=3.4`
  - `seaborn>=0.11`
  - `IPython`
- [x] Create `.gitignore` (Python standard + `.ipynb_checkpoints`, `dist/`, `*.egg-info`)
- [x] Set up virtual environment locally
- [x] First commit: skeleton files only, nothing functional yet

---

## PHASE 1 — Column Classifier (`classifier.py`)

> This is the most important module. Everything downstream depends on getting column types right.

- [ ] Define the type enum or constants:
  ```
  NUMERIC_CONTINUOUS
  NUMERIC_DISCRETE       # int with low unique count (< 20 unique values)
  CATEGORICAL_LOW        # object/category with <= 15 unique values
  CATEGORICAL_HIGH       # object/category with > 15 unique values
  DATETIME
  BOOLEAN
  CONSTANT               # only 1 unique value
  NEAR_CONSTANT          # top value >= 98% of rows
  ID_LIKE                # unique ratio > 95%, int or object
  ```

- [ ] Write `classify_column(series) -> str` function:
  - [ ] Check constant first (fastest exit)
  - [ ] Check near-constant second
  - [ ] Check ID-like (unique ratio heuristic)
  - [ ] Check pandas dtype for datetime
  - [ ] Check boolean
  - [ ] Check numeric — then decide continuous vs discrete based on unique count
  - [ ] Fall through to categorical low vs high based on unique count threshold

- [ ] Write `classify_dataframe(df) -> dict` — returns `{col_name: type}` for every column

- [ ] Edge cases to handle explicitly:
  - [ ] All-null column → classify as CONSTANT, flag for drop
  - [ ] Numeric column that looks like an ID (e.g. `user_id` stored as int) → ID_LIKE wins over NUMERIC
  - [ ] Boolean stored as 0/1 integers → detect and classify as BOOLEAN not NUMERIC
  - [ ] Mixed type columns (object with some numbers) → classify as CATEGORICAL, add warning

- [ ] Write unit tests in `tests/test_classifier.py`:
  - [ ] Test each type with a synthetic series
  - [ ] Test edge cases above
  - [ ] Test that all columns in a real-ish df get classified without error

---

## PHASE 2 — Profiler (`profiler.py`)

> Per-column stats. Pure computation, no display.

- [ ] Write `profile_numeric(series) -> dict`:
  - [ ] `count` — non-null count
  - [ ] `missing` — null count
  - [ ] `missing_pct` — null % rounded to 1 decimal
  - [ ] `mean`, `median`, `std`, `min`, `max`
  - [ ] `skew` — pandas `.skew()`
  - [ ] `kurtosis` — pandas `.kurtosis()`
  - [ ] `outlier_count` — IQR method (< Q1 - 1.5×IQR or > Q3 + 1.5×IQR)
  - [ ] `outlier_pct` — outliers as % of non-null rows
  - [ ] `zeros_pct` — % of values that are exactly 0 (useful signal)
  - [ ] `unique_count`

- [ ] Write `profile_categorical(series) -> dict`:
  - [ ] `count`, `missing`, `missing_pct`
  - [ ] `unique_count`
  - [ ] `top_values` — top 5 values and their % frequency as list of tuples
  - [ ] `entropy` — optional but useful signal for uniformity

- [ ] Write `profile_datetime(series) -> dict`:
  - [ ] `min_date`, `max_date`
  - [ ] `range_days`
  - [ ] `missing`, `missing_pct`
  - [ ] `is_monotonic` — always increasing/decreasing?

- [ ] Write `profile_dataframe(df, col_types: dict) -> dict`:
  - [ ] Routes each column to the right profiler based on classified type
  - [ ] Returns `{col_name: profile_dict}` for all columns
  - [ ] Skips CONSTANT / NEAR_CONSTANT / ID_LIKE columns from deep profiling (just flag them)

- [ ] Dataset-level stats (attach to top of report dict):
  - [ ] `shape` — (rows, cols)
  - [ ] `total_missing_pct`
  - [ ] `duplicate_rows` — count
  - [ ] `memory_mb` — `df.memory_usage(deep=True).sum() / 1e6`
  - [ ] `numeric_col_count`, `categorical_col_count`, `datetime_col_count`

- [ ] Write unit tests in `tests/test_profiler.py`

---

## PHASE 3 — Relationships (`relationships.py`)

> Cross-column analysis. Only runs on columns worth analyzing.

- [ ] Write `compute_correlations(df, col_types) -> dict`:
  - [ ] Filter to only NUMERIC_CONTINUOUS and NUMERIC_DISCRETE columns
  - [ ] Compute full Pearson correlation matrix
  - [ ] Extract pairs where `abs(corr) >= 0.85` (make threshold configurable)
  - [ ] Return as list of `(col_a, col_b, correlation_value)` tuples, sorted by abs value desc
  - [ ] If fewer than 2 numeric columns → return empty list gracefully

- [ ] Write `correlate_with_target(df, target_col, col_types) -> list`:
  - [ ] Only runs if user passed `target=` parameter
  - [ ] Returns all numeric columns ranked by abs correlation to target
  - [ ] Include direction (positive/negative)

- [ ] Write `detect_duplicate_columns(df) -> list`:
  - [ ] Find columns with identical values (not just same name)
  - [ ] Return list of duplicate pairs

- [ ] Edge cases:
  - [ ] Correlation on columns with zero variance → catch and skip
  - [ ] All-null column in correlation → skip
  - [ ] Single numeric column → skip correlation entirely, no error

---

## PHASE 4 — Suggestion Engine (`suggestions.py`)

> The most user-facing logic. Every rule maps to one plain-English line.

- [ ] Define rule priority order (earlier rules = higher priority, column gets first matching suggestion):

  ```
  Priority 1 — DROP candidates
    - CONSTANT column
    - NEAR_CONSTANT column (>= 98%)
    - ID_LIKE column
    - ALL_NULL column
    - Duplicate of another column

  Priority 2 — HIGH CONCERN
    - Missing > 50%
    - Part of high-correlation pair (>= 0.85)
    - Outlier % > 10%

  Priority 3 — MODERATE CONCERN
    - Missing 20–50%
    - Skew abs > 1.0
    - High cardinality categorical (> 50 unique)

  Priority 4 — LOW CONCERN / INFO
    - Missing < 20%
    - Skew 0.5–1.0
    - Kurtosis > 3 (heavy tails)
    - Zeros > 30% (suspicious for a non-binary column)
  ```

- [ ] Write `suggest_for_column(col_name, col_type, profile, correlation_flags) -> list[str]`:
  - [ ] Returns list of suggestion strings for that column
  - [ ] Each string is one plain-English sentence
  - [ ] Examples:
    - `"Drop — constant column, carries no information"`
    - `"Drop — appears to be a unique row identifier"`
    - `"Impute with median — 12.3% missing"`
    - `"Apply log transform — skew of 2.4 indicates right-skewed distribution"`
    - `"Investigate outliers — 8.2% of values fall outside IQR bounds"`
    - `"High correlation with 'age' (0.93) — consider dropping one before modeling"`
    - `"High cardinality (142 unique values) — consider grouping rare categories"`

- [ ] Write `suggest_for_dataframe(profiles, col_types, correlations) -> dict`:
  - [ ] Returns `{col_name: [suggestions]}` for every flagged column
  - [ ] Columns with no flags → not included in dict (keep output short)

- [ ] Write `get_global_suggestions(dataset_stats) -> list[str]`:
  - [ ] Dataset-level suggestions (not per-column):
    - [ ] Duplicate rows present → "Remove N duplicate rows"
    - [ ] Overall missing > 30% → "Dataset has high overall missingness — review collection pipeline"
    - [ ] Very few rows (< 100) → "Small dataset — results may not generalize"

- [ ] Write unit tests in `tests/test_suggestions.py`:
  - [ ] Test each rule fires on the right input
  - [ ] Test that clean columns produce zero suggestions
  - [ ] Test priority ordering (constant column shouldn't also get impute suggestion)

---

## PHASE 5 — Renderer (`renderer.py`)

> All display logic lives here. Nothing else should call `print()` or `display()`.

- [ ] Write `render_banner(dataset_stats)`:
  - [ ] Rows, columns, missing %, duplicates, memory
  - [ ] Use IPython HTML for stat tiles
  - [ ] Keep it to 2-3 lines of visual height max

- [ ] Write `render_warnings(col_suggestions, col_types)`:
  - [ ] Only show columns that have suggestions
  - [ ] Group by severity: DROP first, then HIGH, then MODERATE, then INFO
  - [ ] Format: `  income   — 10.2% missing, skew 2.3`
  - [ ] If no warnings → print `"✓ No issues detected"`

- [ ] Write `render_suggestions(col_suggestions, global_suggestions)`:
  - [ ] Numbered list, one line per suggestion
  - [ ] Format: `  1. Drop user_id — unique identifier, no predictive value`

- [ ] Write `render_full_stats(profiles, col_types)`:
  - [ ] Only called in `mode="full"`
  - [ ] Per-column table: mean, median, std, skew, missing %, outliers
  - [ ] Use pandas DataFrame display for numeric summary
  - [ ] Keep it clean — not a raw `df.describe()` dump

- [ ] Write `render_divider(title)`:
  - [ ] Reusable section separator: `━━━━━━ WARNINGS ━━━━━━`

- [ ] Write `render_all(report_dict, mode)`:
  - [ ] Master render function that calls everything in order
  - [ ] `mode="tldr"` → banner + warnings + suggestions only
  - [ ] `mode="full"` (default) → banner + warnings + suggestions + full stats + plots
  - [ ] `mode="full"` with `plots=False` → same as full but skip plots

---

## PHASE 6 — Plots (`plots.py`)

> Minimal. 3 plot types only. No clutter.

- [ ] Set a global style function `_set_style()` called once:
  - [ ] No top/right spines
  - [ ] Light grid
  - [ ] Clean font
  - [ ] Consistent color palette (one primary color, one accent for warnings)

- [ ] Write `plot_distributions(df, col_types)`:
  - [ ] Only NUMERIC_CONTINUOUS and NUMERIC_DISCRETE columns
  - [ ] Histograms in a grid, max 3 per row
  - [ ] Annotate skew value on each plot if abs(skew) > 0.5
  - [ ] Red annotation if skew > 1, gray if between 0.5 and 1
  - [ ] Single `plt.show()` at the end, not per subplot

- [ ] Write `plot_outliers(df, col_types)`:
  - [ ] Boxplots for all numeric columns
  - [ ] Annotate outlier count on each box
  - [ ] Same grid layout as distributions

- [ ] Write `plot_correlation(df, col_types, high_corr_pairs)`:
  - [ ] If numeric columns <= 15: show full heatmap, lower triangle only
  - [ ] If numeric columns > 15: skip heatmap, show top 10 correlated pairs as a simple bar chart instead
  - [ ] Annotate correlation values on heatmap cells

- [ ] Write `plot_all(df, col_types, profiles, correlations)`:
  - [ ] Calls the three above in order
  - [ ] Skips any plot where there's not enough data to render it meaningfully

---

## PHASE 7 — Core Orchestrator (`core.py`)

> The conductor. Calls everything else in order.

- [ ] Write `quick_eda(df, mode="full", target=None, plots=True, sample=True, sample_size=50_000, return_report=False)`:

  - [ ] **Input validation**:
    - [ ] Check `df` is a pandas DataFrame
    - [ ] Check `df` is not empty
    - [ ] Check at least 2 columns exist
    - [ ] If `target` passed, check it exists in `df`
    - [ ] Raise `ValueError` with friendly message for each failure

  - [ ] **Sampling**:
    - [ ] If `sample=True` and `len(df) > sample_size`:
      - [ ] Sample randomly with `random_state=42`
      - [ ] Print: `"⚡ Large dataset — running on 50,000 row sample. Pass sample=False to use full data."`
    - [ ] If `sample=False` and df is huge → just run (user's choice)

  - [ ] **Run pipeline in order**:
    1. `classify_dataframe(df)` → col_types
    2. `profile_dataframe(df, col_types)` → profiles
    3. Dataset-level stats from profiles
    4. `compute_correlations(df, col_types)` → correlations
    5. If target: `correlate_with_target(df, target, col_types)` → target_correlations
    6. `suggest_for_dataframe(profiles, col_types, correlations)` → col_suggestions
    7. `get_global_suggestions(dataset_stats)` → global_suggestions

  - [ ] **Build report dict** (always, regardless of mode):
    ```python
    report = {
      "shape": ...,
      "missing_pct": ...,
      "duplicates": ...,
      "memory_mb": ...,
      "col_types": col_types,
      "profiles": profiles,
      "correlations": correlations,
      "suggestions": col_suggestions,
      "global_suggestions": global_suggestions,
    }
    ```

  - [ ] **Render**:
    - [ ] Call `render_all(report, mode=mode)`
    - [ ] If `plots=True` and `mode != "tldr"`: call `plot_all(...)`

  - [ ] **Return**:
    - [ ] If `return_report=True` → return report dict
    - [ ] Else → return `None` (side-effect only, like pandas `df.info()`)

- [ ] Write `__init__.py`:
  - [ ] Only export `quick_eda`
  - [ ] Set `__version__ = "0.1.0"`

---

## PHASE 8 — Testing & Polish

- [ ] Write `tests/test_core.py`:
  - [ ] Test with a clean DataFrame — no warnings expected
  - [ ] Test with a messy DataFrame — all suggestion types should fire
  - [ ] Test `return_report=True` — check dict keys exist
  - [ ] Test `mode="tldr"` — should not error
  - [ ] Test with `target=` column — should not error
  - [ ] Test with empty DataFrame — should raise friendly error
  - [ ] Test with non-DataFrame input — should raise friendly error
  - [ ] Test with single column — should raise friendly error
  - [ ] Test large df (>50k rows) — sampling message should print

- [ ] Create `examples/demo.ipynb`:
  - [ ] Cell 1: install + import
  - [ ] Cell 2: load a public dataset (Titanic or similar)
  - [ ] Cell 3: `quick_eda(df)` — show full output
  - [ ] Cell 4: `quick_eda(df, mode="tldr")`
  - [ ] Cell 5: `report = quick_eda(df, return_report=True)` then `report["suggestions"]`
  - [ ] Cell 6: `quick_eda(df, target="Survived")`

---

## PHASE 9 — Publishing to PyPI

- [ ] Add `long_description` in `setup.py` pointing to README.md
- [ ] Add `classifiers` in `setup.py`:
  - `Programming Language :: Python :: 3`
  - `License :: OSI Approved :: MIT License`
  - `Topic :: Scientific/Engineering :: Information Analysis`
- [ ] Create `pyproject.toml` (modern builds):
  ```toml
  [build-system]
  requires = ["setuptools", "wheel"]
  build-backend = "setuptools.backends.legacy:build"
  ```
- [ ] Register on PyPI (create account if needed)
- [ ] Build: `python -m build`
- [ ] Test on TestPyPI first: `twine upload --repository testpypi dist/*`
- [ ] Install from TestPyPI and verify: `pip install -i https://test.pypi.org/simple/ quick-eda`
- [ ] If clean: publish to real PyPI: `twine upload dist/*`
- [ ] Verify: `pip install quick-eda` works from scratch

---

## PHASE 10 — v0.2 Backlog (don't touch until v0.1 is on PyPI)

- [ ] `target=` parameter full implementation (reorient output around target column)
- [ ] Time series detection and line plot support for datetime-indexed DataFrames
- [ ] HTML export: `quick_eda(df, export="report.html")`
- [ ] `mode="full"` per-column stat table (like `df.describe()` but cleaner)
- [ ] Automatic encoding suggestions for high-cardinality categoricals
- [ ] CI/CD quality gate helper: `quick_eda.assert_quality(df, max_missing=0.2)`
- [ ] Categorical plot: top-N bar chart for low-cardinality columns
- [ ] Better sampling: stratified by target if target is passed

---

## Current Status

- [x] Phase 0 — Setup
- [ ] Phase 1 — Classifier
- [ ] Phase 2 — Profiler
- [ ] Phase 3 — Relationships
- [ ] Phase 4 — Suggestion Engine
- [ ] Phase 5 — Renderer
- [ ] Phase 6 — Plots
- [ ] Phase 7 — Core
- [ ] Phase 8 — Testing
- [ ] Phase 9 — PyPI
