# quick_eda — Full Roadmap TODO

> This document tracks everything from v0.1.1 to v1.0.0.
> Work top to bottom. Do not skip versions. Each version builds on the last.
> Update checkboxes as you complete tasks. Commit TODO.md with every version tag.

---

## Current Status

```
v0.1.0   Core pipeline working, tested on real datasets, pushed to GitHub
v0.1.1   Bug fixes 
v0.2.0   Categorical intelligence
v0.3.0   Target mode (full implementation)
v0.4.0   Time series support
v0.5.0   CLI support
v0.6.0   AI narrative summary
v0.7.0   Data quality scoring
v0.8.0   HTML export + shareable reports
v0.9.0   Performance + polish
v1.0.0   Interactive mode + compare + full docs
```

---

## v0.2.0 — Categorical Intelligence
> Currently blind to categorical data. No plots, limited insights.
> This version makes categorical columns first-class citizens.

### classifier.py
- [x] Review CATEGORICAL_LOW threshold — currently 15 unique values
  - Consider raising to 20 for wider datasets
  - Add config option so user can override: `quick_eda(df, cat_threshold=20)`

### profiler.py
- [x] Enhance `profile_categorical()`:
  - [x] Add `imbalance_ratio` — ratio of most frequent to least frequent value
  - [x] Add `rare_category_pct` — % of rows that belong to categories appearing < 1% of the time
  - [x] Add `entropy` — Shannon entropy as a uniformity measure (0 = one value, high = very spread)

### suggestions.py
- [x] Add imbalanced category rule:
  - If top value > 80% but < 98% (below NEAR_CONSTANT threshold) → `"Imbalanced — '{top_val}' dominates {pct}% of rows, may cause model bias"`
- [x] Improve high cardinality suggestions — be more specific:
  - > 50 unique AND rare_category_pct > 20% → `"Many rare categories ({pct}% of rows) — consider grouping into 'Other'"`
  - > 50 unique AND imbalance is low → `"High cardinality — consider frequency encoding or target encoding"`
- [x] Add boolean column suggestion:
  - If True/False split is extremely imbalanced (> 95% one value) → treat like NEAR_CONSTANT

### plots.py
- [x] Add `plot_categoricals(df, col_types)`:
  - For every CATEGORICAL_LOW column: horizontal bar chart of top-N value counts
  - Sort bars by frequency descending
  - Annotate each bar with its percentage
  - Max 3 columns per row, same grid layout as distributions
  - Cap at top 10 values — if more exist, show "other" as a final bar
  - Color: use accent color for the top bar, muted for the rest
- [x] Add `plot_boolean(df, col_types)`:
  - For every BOOLEAN column: simple two-bar chart (True vs False counts)
  - Annotate with percentage split
- [x] Update `plot_all()` to call both new functions after existing plots

### renderer.py
- [x] Add categorical summary section to `_warnings()`:
  - Show imbalanced categoricals under MODERATE concern
- [x] Add categorical section to `_full_stats()`:
  - Table showing: column, unique count, top value, top value %, rare category %, entropy
  - Separate from numeric stats table, not mixed in

### core.py
- [x] Pass categorical profiles to renderer correctly
- [x] Ensure new plots are gated behind `plots=True` and `mode != "tldr"`

### tests
- [x] `test_profiler.py` — test `imbalance_ratio`, `rare_category_pct`, `entropy` keys exist
- [x] `test_suggestions.py` — test imbalanced category rule fires correctly
- [x] `test_suggestions.py` — test improved high cardinality suggestions
- [x] `test_core.py` — test categorical plots don't error on a df with categoricals

### release
- [x] Bump version to `0.2.0`
- [x] Update README — add categorical detection to "What it detects" table
- [ ] Commit: `FEAT: v0.2.0 — categorical intelligence, plots, and suggestions`
- [ ] Tag: `git tag v0.2.0 && git push origin main --tags`

---

## v0.3.0 — Target Mode (Full Implementation)
> `target=` parameter exists but barely does anything.
> This version makes it genuinely useful for ML workflows.

### relationships.py
- [ ] Enhance `correlate_with_target()`:
  - [ ] Include direction — positive or negative correlation
  - [ ] Include correlation strength label — "strong", "moderate", "weak"
  - [ ] For categorical target: use point-biserial correlation for numeric columns
  - [ ] For categorical columns vs target: use Cramér's V statistic

### suggestions.py
- [ ] Add target-aware suggestions:
  - Columns with abs(correlation to target) < 0.01 → `"Near-zero correlation with target '{target}' — likely low predictive value"`
  - Columns with abs(correlation to target) > 0.7 → `"Strong correlation with target — important feature, handle carefully"`
- [ ] Add class balance check when target is categorical:
  - If any class < 5% of rows → `"Class imbalance detected — '{class}' has only {pct}% of samples, consider oversampling"`

### renderer.py
- [ ] Add target correlation section to output (only when `target=` is passed):
  ```
  ─── FEATURE RELEVANCE (target: price) ───

    strong positive   bedrooms        0.82
    strong positive   bathrooms       0.71
    moderate positive sqft_living     0.61
    weak              yr_built        0.09
    near zero         zipcode         0.01  ← low predictive value
  ```
- [ ] Move this section between WARNINGS and SUGGESTIONS so it's prominent

### plots.py
- [ ] Add `plot_target_correlations(correlations, target)`:
  - Horizontal bar chart of all features ranked by abs(correlation to target)
  - Color by direction — blue for positive, coral for negative
  - Only shown when `target=` is passed
- [ ] Add `plot_target_distributions(df, target, col_types)`:
  - For numeric target: color histograms by target quartile
  - For categorical target (classification): overlay distributions per class
  - Only top 4 most correlated features shown

### core.py
- [ ] Pass `target` info through to renderer and plots correctly
- [ ] Detect regression vs classification:
  - If target has <= 10 unique values → classification mode
  - If target is continuous → regression mode
  - Store in report dict as `report["target_type"]`

### tests
- [ ] Test target correlation ranking is correct
- [ ] Test classification vs regression detection
- [ ] Test class imbalance suggestion fires
- [ ] Test near-zero correlation suggestion fires
- [ ] Test target section only appears when target is passed

### release
- [ ] Bump version to `0.3.0`
- [ ] Update README — document `target=` parameter fully with examples
- [ ] Commit: `FEAT: v0.3.0 — full target mode, feature relevance ranking`
- [ ] Tag: `git tag v0.3.0 && git push origin main --tags`

---

## v0.4.0 — Time Series Support
> Datetime columns currently just get profiled for range and monotonicity.
> This version adds actual time series intelligence.

### classifier.py
- [ ] Add `TIME_SERIES` type — distinct from DATETIME
  - A column is TIME_SERIES if: it's datetime AND the dataframe is indexed by it OR it's the dominant datetime column
- [ ] Add `_is_dominant_datetime(df)` helper:
  - Returns the name of the datetime column if there's exactly one, else None

### profiler.py
- [ ] Add `profile_timeseries(series, value_cols)` function:
  - [ ] `frequency` — detected frequency (daily, weekly, monthly, irregular)
  - [ ] `gaps` — count of missing time periods
  - [ ] `gap_dates` — list of first 5 gap dates for display
  - [ ] `trend` — "upward", "downward", "flat" based on linear regression slope
  - [ ] `seasonality_hint` — if autocorrelation at lag 7, 12, or 52 is > 0.5, flag possible seasonality

### plots.py
- [ ] Add `plot_timeseries(df, time_col, col_types)`:
  - Line plot for each numeric column against the time axis
  - Highlight gaps as vertical shaded regions
  - Show trend line overlay
  - Only triggered when a TIME_SERIES column is detected

### suggestions.py
- [ ] Add time series specific suggestions:
  - Gaps detected → `"Time gaps found in '{col}' — {n} missing periods, consider interpolation"`
  - Trend detected → `"'{col}' shows a {direction} trend — consider differencing before modeling"`
  - Seasonality hint → `"Possible seasonality detected in '{col}' — check autocorrelation at lag {lag}"`

### renderer.py
- [ ] Add time series section to output when detected:
  ```
  ─── TIME SERIES ───
    time column     timestamp
    frequency       daily
    range           2020-01-01 → 2024-12-31  (1826 days)
    gaps            3 missing periods
    trend           upward
    seasonality     possible (lag 7)
  ```

### core.py
- [ ] Auto-detect if dataframe has time series structure before running pipeline
- [ ] If time series detected → print `"Time series structure detected — switching to temporal analysis"`

### tests
- [ ] Test frequency detection on daily/weekly/monthly series
- [ ] Test gap detection
- [ ] Test trend detection
- [ ] Test time series suggestions fire correctly

### release
- [ ] Bump version to `0.4.0`
- [ ] Update README — add time series section
- [ ] Commit: `FEAT: v0.4.0 — time series detection and analysis`
- [ ] Tag: `git tag v0.4.0 && git push origin main --tags`

---

## v0.5.0 — CLI Support
> Currently only works in Jupyter. This version makes it work everywhere.

### setup.py
- [ ] Add entry point:
  ```python
  entry_points={
      "console_scripts": [
          "quick-eda=quick_eda.cli:main",
      ]
  }
  ```

### quick_eda/cli.py (new file)
- [ ] Create `cli.py` with `main()` function
- [ ] Argument parsing using `argparse`:
  - [ ] `file` — positional, path to CSV/Excel file (required)
  - [ ] `--mode` — `full` or `tldr` (default: `full`)
  - [ ] `--target` — column name for target mode
  - [ ] `--no-plots` — disable plots
  - [ ] `--output` — save HTML report to file
  - [ ] `--sample-size` — override default 50k sample size
  - [ ] `--no-sample` — disable sampling
  - [ ] `--sep` — separator for CSV (default: auto-detect)
  - [ ] `--version` — print version and exit
- [ ] Auto file format detection:
  - `.csv` → `pd.read_csv()`
  - `.tsv` → `pd.read_csv(sep="\t")`
  - `.xlsx`, `.xls` → `pd.read_excel()`
  - `.parquet` → `pd.read_parquet()`
  - `.json` → `pd.read_json()`
  - Unknown extension → try CSV, fail with friendly error
- [ ] Friendly error messages:
  - File not found → `"File not found: {path}. Check the path and try again."`
  - Wrong format → `"Could not read {path}. Supported formats: CSV, Excel, Parquet, JSON"`
  - Target not found → `"Column '{target}' not found. Available columns: {list}"`

### Usage after install:
```bash
quick-eda data.csv
quick-eda data.csv --mode tldr
quick-eda data.csv --target price
quick-eda data.csv --no-plots
quick-eda data.csv --output report.html
quick-eda data.xlsx --target churn
```

### renderer.py
- [ ] Detect if running in Jupyter or terminal
  - Use `IPython.get_ipython()` — if None, we're in a terminal
  - In terminal mode: print output as-is (already works since we use `print()`)
  - In Jupyter mode: current behavior
- [ ] Plots in CLI mode:
  - If `--output` is set → save plots as PNG alongside the HTML
  - If no output → `plt.show()` (opens a window, works in most terminals)

### tests
- [ ] Test CLI with a CSV file — no error
- [ ] Test CLI `--mode tldr`
- [ ] Test CLI `--target col`
- [ ] Test CLI with non-existent file — friendly error
- [ ] Test CLI with wrong format — friendly error
- [ ] Test auto format detection for CSV, Excel, Parquet

### release
- [ ] Bump version to `0.5.0`
- [ ] Update README — add CLI section with usage examples
- [ ] Commit: `FEAT: v0.5.0 — CLI support, multi-format file loading`
- [ ] Tag: `git tag v0.5.0 && git push origin main --tags`
- [ ] **Publish to PyPI** — this is the first version worth publishing
  - [ ] `python -m build`
  - [ ] Test on TestPyPI: `twine upload --repository testpypi dist/*`
  - [ ] Verify: `pip install -i https://test.pypi.org/simple/ quick-eda`
  - [ ] Publish: `twine upload dist/*`
  - [ ] Verify: `pip install quick-eda` works from scratch

---

## v0.6.0 — AI Narrative Summary
> The first WOW feature. Plain English dataset summary using an LLM.
> No raw data is ever sent — only the report dict (stats, types, suggestions).

### quick_eda/ai_summary.py (new file)
- [ ] Create `generate_summary(report: dict, api_key: str, provider: str) -> str`
- [ ] Provider support:
  - [ ] `"anthropic"` — uses Claude via `anthropic` SDK
  - [ ] `"openai"` — uses GPT via `openai` SDK
  - [ ] Auto-detect from env vars: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- [ ] Build the prompt from report dict — never send raw data:
  ```
  Dataset: {rows} rows, {cols} columns
  Types: {numeric} numeric, {categorical} categorical, {datetime} datetime
  Missing: {missing_pct}% overall
  Duplicates: {duplicates}
  Key issues: {top 5 suggestions}
  High correlations: {correlation pairs}
  Skewed columns: {skewed cols with skew values}
  ...
  ```
- [ ] Prompt instruction: "Write 3-5 sentences in plain English describing this dataset, its main issues, and the most important first step to take. Be specific. Do not use bullet points."
- [ ] Cache the summary in the report dict so it's not re-generated on every call
- [ ] Token limit: max 300 tokens output — keep it short
- [ ] Graceful fallback: if API call fails → skip summary silently, don't crash

### core.py
- [ ] Add `ai_summary=False` parameter to `quick_eda()`
- [ ] Add `api_key=None` parameter — override env var if passed directly
- [ ] Add `provider="auto"` parameter — auto-detect or specify
- [ ] If `ai_summary=True`:
  - Check for API key — if missing, print warning and skip
  - Call `generate_summary()` after building report dict
  - Store result in `report["ai_summary"]`

### renderer.py
- [ ] Add `_render_ai_summary(summary: str)` function:
  - Prints at the very top, before the banner
  - Format:
    ```
    ─── DATASET SUMMARY ───

      This appears to be an Airbnb listings dataset for New York City with ~49k
      properties. The price column is extremely right-skewed — a log transform is
      strongly recommended. Two columns (id, name) are identifiers and should be
      dropped. About 20% of reviews data is missing, likely because newer listings
      haven't been reviewed yet.

    ────────────────────────────────────────────
    ```
- [ ] Update `render_all()` to call `_render_ai_summary()` first if summary exists in report

### requirements.txt
- [ ] Add `anthropic>=0.20` as optional dependency
- [ ] Add `openai>=1.0` as optional dependency
- [ ] Mark both as optional in setup.py extras:
  ```python
  extras_require={
      "ai": ["anthropic>=0.20", "openai>=1.0"],
  }
  ```
- [ ] Install with: `pip install quick-eda[ai]`

### tests
- [ ] Test `generate_summary()` with a mocked API call — no real API needed in tests
- [ ] Test `ai_summary=False` (default) — no API call made
- [ ] Test missing API key — warning printed, no crash
- [ ] Test failed API call — graceful fallback, no crash
- [ ] Test summary stored in report dict when `return_report=True`

### release
- [ ] Bump version to `0.6.0`
- [ ] Update README — add AI summary section with setup instructions
- [ ] Commit: `FEAT: v0.6.0 — AI narrative summary (Anthropic + OpenAI)`
- [ ] Tag: `git tag v0.6.0 && git push origin main --tags`
- [ ] Update PyPI

---

## v0.7.0 — Data Quality Scoring
> Give the dataset a score so users have one number to optimize.
> Makes quality tangible and gamifies the cleaning process.

### quick_eda/scorer.py (new file)
- [ ] Create `compute_quality_score(dataset_stats, profiles, col_types, suggestions) -> dict`
- [ ] Scoring formula (starts at 100, penalties deducted):

  ```
  Missing values:
    0%          → 0 penalty
    1-5%        → -5
    5-10%       → -10
    10-20%      → -20
    > 20%       → -30

  Duplicate rows:
    0           → 0 penalty
    > 0         → -10

  Useless columns (per column):
    CONSTANT    → -5 each (max -15)
    NEAR_CONST  → -3 each (max -10)
    ID_LIKE     → -3 each (max -10)

  Data quality issues (per column):
    skew > 1    → -2 each (max -10)
    outliers > 10% → -2 each (max -10)
    missing col > 20% → -3 each (max -15)

  Bonus points:
    0 duplicates        → +5
    0 missing           → +5
    no useless columns  → +5
  ```

- [ ] Return dict:
  ```python
  {
    "score": 74,
    "grade": "B",       # A=90+, B=75+, C=60+, D=45+, F=below 45
    "penalties": [...], # list of (reason, points_lost)
    "bonuses":   [...], # list of (reason, points_gained)
  }
  ```

### renderer.py
- [ ] Add score to banner:
  ```
    quick_eda
  ────────────────────────────────────────────────────
    rows          3,204       quality score   74/100 B
    columns          14
  ```
- [ ] Color code the score:
  - 90+  → green
  - 75+  → yellow
  - 60+  → orange
  - below → red
  - (use unicode characters since we're plain text: no actual colors)
  - Instead use: `[A]`, `[B]`, `[C]`, `[D]`, `[F]` grade labels

### quick_eda/assert_quality.py (new file)
- [ ] Create `assert_quality(df, min_score=70, max_missing=None, max_duplicates=None)`:
  - Runs the full pipeline silently (no display)
  - Raises `DataQualityError` if any threshold is violated
  - `DataQualityError` message lists exactly which checks failed
  - Perfect for CI/CD:
    ```python
    # in your data pipeline
    from quick_eda import assert_quality
    assert_quality(df, min_score=80, max_missing=0.1)
    # raises if score < 80 or missing > 10%
    ```
- [ ] Create custom `DataQualityError(Exception)` class in `exceptions.py`

### tests
- [ ] Test scoring on a clean df — should be near 100
- [ ] Test scoring on a messy df — should be low
- [ ] Test each penalty fires correctly
- [ ] Test bonus points fire correctly
- [ ] Test grade labels are correct
- [ ] Test `assert_quality` passes on clean df
- [ ] Test `assert_quality` raises on messy df
- [ ] Test `assert_quality` error message is informative

### release
- [ ] Bump version to `0.7.0`
- [ ] Update README — add quality score section and `assert_quality` docs
- [ ] Commit: `FEAT: v0.7.0 — data quality scoring and assert_quality`
- [ ] Tag: `git tag v0.7.0 && git push origin main --tags`
- [ ] Update PyPI

---

## v0.8.0 — HTML Export + Shareable Reports
> For non-technical users who need to share findings with stakeholders.
> One file, self-contained, opens in any browser.

### quick_eda/exporter.py (new file)
- [ ] Create `export_html(report, plots, output_path)` function
- [ ] HTML structure:
  - [ ] Header with dataset name (from filename if available) and generation timestamp
  - [ ] AI summary section at top (if present in report)
  - [ ] Quality score badge
  - [ ] Summary banner (rows, cols, missing, duplicates, memory)
  - [ ] Warnings table
  - [ ] Suggestions numbered list
  - [ ] Column stats table
  - [ ] All plots embedded as base64 PNG — no external dependencies
  - [ ] Footer with quick_eda version
- [ ] Self-contained — single `.html` file, no CSS/JS files needed
- [ ] Mobile-friendly layout — readable on phone
- [ ] Dark/light theme toggle button
- [ ] File size target: under 2MB for typical datasets

### core.py
- [ ] Add `export=None` parameter to `quick_eda()`
- [ ] If `export="report.html"` → call `export_html()` after rendering
- [ ] Print confirmation: `"✓ Report saved to report.html"`

### cli.py
- [ ] `--output report.html` → passes to `export=` parameter

### tests
- [ ] Test HTML file is created at specified path
- [ ] Test HTML file is valid (contains expected sections)
- [ ] Test plots are embedded (file size > baseline)
- [ ] Test with `ai_summary=True` — summary appears in HTML

### release
- [ ] Bump version to `0.8.0`
- [ ] Update README — add HTML export section with screenshot
- [ ] Commit: `FEAT: v0.8.0 — HTML export and shareable reports`
- [ ] Tag: `git tag v0.8.0 && git push origin main --tags`
- [ ] Update PyPI

---

## v0.9.0 — Performance + Polish
> Before v1.0.0 everything must be fast, solid, and production-ready.
> No new features. Only hardening, speed, and cleanup.

### Performance
- [ ] Parallel column profiling using `concurrent.futures.ThreadPoolExecutor`:
  - Profile all columns in parallel instead of sequentially
  - Significant speedup on wide datasets (50+ columns)
  - Add `n_jobs=-1` parameter to use all available cores
  - Benchmark: measure time on a 100-column dataset before and after
- [ ] Progress bar for large datasets using `tqdm`:
  - `Profiling columns ━━━━━━━━━━ 100% | 50/50 [0.8s]`
  - Only shown when dataset > 10k rows
  - Add `progress=True` parameter (default True, set False to disable)
- [ ] Memory-efficient mode for very large files:
  - Add `chunksize` parameter for CSV reading
  - When `chunksize` is set, profile in chunks and aggregate

### Edge case hardening
- [ ] All-null DataFrame → friendly error
- [ ] Single-row DataFrame → friendly warning, still runs
- [ ] Single-column DataFrame → already raises, make message clearer
- [ ] Unicode column names → test and fix if broken
- [ ] Column names with spaces → test and fix if broken
- [ ] Columns named `index` or `level_0` → test and fix if broken
- [ ] DataFrame with all-identical rows → test duplicate detection
- [ ] DataFrame with mixed types in a column → handle gracefully
- [ ] Very wide DataFrame (500+ columns) → test performance, add warning if > 200 cols

### Code quality
- [ ] Remove all `# type: ignore` comments and fix typing properly
- [ ] Add type hints to all public functions
- [ ] Docstrings on every public function — numpy style
- [ ] Remove dead code — the old datetime sniff block that was left in classifier
- [ ] Consistent naming — audit all variable names for clarity
- [ ] No bare `except:` clauses — always catch specific exceptions

### Testing
- [ ] Achieve 90%+ test coverage — run `pytest --cov=quick_eda`
- [ ] Add edge case tests for all items above
- [ ] Add performance test — assert `quick_eda(df)` completes in < 5s on 50k rows
- [ ] Add regression tests — save expected output and assert it doesn't change
- [ ] Test on Python 3.9, 3.10, 3.11, 3.12, 3.14 — add GitHub Actions matrix

### GitHub Actions CI
- [ ] Create `.github/workflows/tests.yml`:
  ```yaml
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          python-version: ["3.9", "3.10", "3.11", "3.12"]
      steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v4
        - run: pip install -e ".[dev]"
        - run: pytest tests/ --cov=quick_eda
  ```
- [ ] Add passing badge to README: `![Tests](https://github.com/GodXSpell/AutoEDA/actions/workflows/tests.yml/badge.svg)`

### CHANGELOG.md
- [ ] Create `CHANGELOG.md` and document every version from v0.1.0 to v0.9.0
- [ ] Format:
  ```markdown
  ## [0.9.0] - 2025-XX-XX
  ### Added
  - Parallel column profiling
  - Progress bar for large datasets
  ### Fixed
  - Edge case: all-null DataFrame
  ```

### release
- [ ] Bump version to `0.9.0`
- [ ] Update README — add CI badge, performance benchmarks
- [ ] Commit: `FEAT: v0.9.0 — performance, polish, CI, 90% test coverage`
- [ ] Tag: `git tag v0.9.0 && git push origin main --tags`
- [ ] Update PyPI

---

## v1.0.0 — The Complete Package
> This is the release. Everything is solid, tested, documented.
> The two additions that make it a genuine 1.0.

### Interactive mode (WOW feature #2)
- [ ] Add `interactive=True` parameter to `quick_eda()`
- [ ] When `interactive=True`, use Plotly instead of matplotlib:
  - [ ] Histograms → Plotly histogram with hover showing exact count and range
  - [ ] Boxplots → Plotly box with hover showing all IQR stats
  - [ ] Correlation heatmap → Plotly heatmap, hover shows exact correlation value
  - [ ] Categorical bars → Plotly bar with hover showing exact count and %
- [ ] Clicking a histogram bar → shows the rows in that bin (as a small table)
- [ ] Clicking a heatmap cell → shows scatter plot of that pair inline
- [ ] Add `plotly>=5.0` as optional dependency:
  ```
  pip install quick-eda[interactive]
  ```
- [ ] Falls back to matplotlib gracefully if plotly not installed

### `quick_eda.compare()` (WOW feature #3)
- [ ] Create `compare(df_before, df_after, title_before="before", title_after="after")`
- [ ] Shows side-by-side comparison of two versions of a dataset:
  - Missing % before vs after — did cleaning help?
  - Duplicate count before vs after
  - Quality score before vs after
  - Distribution changes for numeric columns — overlay histograms
  - Which columns were added or removed
  - Which suggestions from `df_before` are now resolved in `df_after`
- [ ] Output:
  ```
  ─── COMPARISON ───

    quality score    62/100 → 89/100  (+27)
    missing %         11.4% → 2.1%    (improved)
    duplicates           23 → 0        (resolved)
    columns              14 → 12       (2 dropped)

  RESOLVED
    ✓ user_id dropped
    ✓ duplicate rows removed
    ✓ income imputed

  REMAINING
    ✗ income still skewed (skew 2.1)
  ```

### Documentation site
- [ ] Set up GitHub Pages or ReadTheDocs
- [ ] Pages:
  - [ ] Getting started — install + first run in 2 minutes
  - [ ] API reference — every parameter documented with types and examples
  - [ ] Tutorials:
    - [ ] Basic EDA walkthrough (Titanic dataset)
    - [ ] ML prep workflow (House Prices dataset)
    - [ ] CI/CD integration (assert_quality in GitHub Actions)
    - [ ] AI summary setup
  - [ ] Changelog
- [ ] Link docs from README badge

### Final PyPI release
- [ ] Audit `setup.py` — all metadata correct
- [ ] Ensure `pip install quick-eda` installs cleanly on Python 3.9-3.14
- [ ] Ensure `pip install quick-eda[ai]` installs AI dependencies
- [ ] Ensure `pip install quick-eda[interactive]` installs Plotly
- [ ] Update PyPI description and screenshots

### README final update
- [ ] Update "What you get" section with actual screenshot from notebook
- [ ] Update author section with real name
- [ ] Add documentation badge
- [ ] Add CI badge
- [ ] Add PyPI download count badge
- [ ] Update comparison table — add interactive mode row

### release
- [ ] Bump version to `1.0.0`
- [ ] Full CHANGELOG.md review
- [ ] Commit: `RELEASE: v1.0.0 — interactive mode, compare(), full documentation`
- [ ] Tag: `git tag v1.0.0 && git push origin main --tags`
- [ ] Create GitHub Release with full release notes
- [ ] Announce on LinkedIn / Twitter / Reddit r/learnpython r/datascience

---

## Version Summary

| Version | Theme | Key deliverable |
|---------|-------|----------------|
| v0.1.0 | ✅ Core | Pipeline working, tested on real datasets |
| v0.1.1 | 🔧 Fixes | Renderer bugs, suggestion guards, dep warnings |
| v0.2.0 | 📊 Categorical | Bar charts, imbalance detection, entropy |
| v0.3.0 | 🎯 Target | Feature relevance, class balance, ranked correlations |
| v0.4.0 | 📈 Time series | Line plots, gaps, trend, seasonality |
| v0.5.0 | 💻 CLI | `quick-eda data.csv`, multi-format, PyPI publish |
| v0.6.0 | 🤖 AI summary | Plain English narrative, Anthropic + OpenAI |
| v0.7.0 | 🏆 Scoring | Quality score, grades, `assert_quality()` |
| v0.8.0 | 📄 Export | Self-contained HTML report |
| v0.9.0 | ⚡ Polish | Parallel profiling, CI, 90% coverage |
| v1.0.0 | 🚀 Launch | Interactive Plotly, `compare()`, docs site |