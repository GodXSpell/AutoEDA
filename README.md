# quick_eda

> TL;DR for your dataset — quick insights, minimal noise, actionable results.

[![PyPI version](https://badge.fury.io/py/quick-eda.svg)](https://badge.fury.io/py/quick-eda)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Most EDA tools give you a 50-page HTML report nobody reads.  
`quick_eda` gives you what you actually need — in seconds, in your notebook, right where you're working.

---

## What you get

```python
import quick_eda

quick_eda(df)
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  quick_eda  •  10,843 rows  •  12 cols  •  3.2 MB
  4.1% missing  •  23 duplicates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠  WARNINGS
   user_id     — looks like a unique identifier
   status      — near-constant (96.2% = "active")
   income      — 11.4% missing, skew 2.8
   age_group   — high cardinality (87 unique values)

💡 SUGGESTIONS
   1. Drop user_id      — unique identifier, no predictive value
   2. Drop status       — near-constant, low variance
   3. Impute income     — median recommended (11.4% missing)
   4. Log-transform income — skew 2.8 exceeds threshold
   5. Review income outliers — 6.1% of values beyond IQR bounds
   6. Remove 23 duplicate rows
```

Then your 3 plots: distributions, boxplots, correlation heatmap.  
That's it. Nothing else.

---

## Install

```bash
pip install quick-eda
```

**Requirements:** Python 3.8+, pandas, numpy, matplotlib, seaborn, IPython

---

## Usage

### One-liner (default)
```python
import quick_eda

quick_eda(df)
```

### TL;DR mode — text only, no plots, instant
```python
quick_eda(df, mode="tldr")
```

### Skip plots but keep full stats
```python
quick_eda(df, plots=False)
```

### Focus on a target column (v0.3.0+)
```python
quick_eda(df, target="price")
```
When `target=` is provided, `quick_eda` unlocks model-aware features:
- **Automatic Target Type Detection:** Detects if the target is continuous (regression) or categorical (classification).
- **Target Correlations:** Ranks all features by their correlation to the target using appropriate statistical tests (Pearson, Point-biserial, Cramér's V).
- **Target-Aware Suggestions:** Flags features with near-zero or extremely high correlation to the target. For classification targets, checks for class imbalance.
- **Target Visualizations:** Plots feature correlations with the target and overlays distributions of top features by target quartiles or classes.

### Get a machine-readable report back
```python
report = quick_eda(df, return_report=True)

# Use it downstream
report["suggestions"]          # all suggestions as a dict
report["profiles"]["income"]   # full stats for a specific column
report["correlations"]         # high-correlation pairs
```

### Large datasets
```python
# Default: auto-samples to 50k rows if df is larger
quick_eda(df)

# Run on full data (may be slower)
quick_eda(df, sample=False)
```

---

## All parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | required | Your dataset |
| `mode` | `str` | `"full"` | `"full"` · `"tldr"` |
| `target` | `str` | `None` | Column to treat as prediction target |
| `plots` | `bool` | `True` | Show plots (only in `"full"` mode) |
| `sample` | `bool` | `True` | Auto-sample large datasets |
| `sample_size` | `int` | `50_000` | Row limit when sampling |
| `return_report` | `bool` | `False` | Return report as dict |

---

## What it detects

### Column-level
| Issue | Detection method | Suggestion |
|---|---|---|
| Constant column | 1 unique value | Drop |
| Near-constant | Top value ≥ 98% of rows | Consider dropping |
| ID-like column | Unique ratio > 95% | Drop before modeling |
| High missingness | > 50% null | Consider dropping |
| Moderate missingness | 20–50% null | Impute with care |
| Low missingness | < 20% null | Impute (median/mode) |
| High skew | abs(skew) > 1.0 | Log transform or Box-Cox |
| Outliers | IQR method, > 10% of rows | Investigate |
| High cardinality | Categorical with > 50 unique values | Group rare categories |
| Rare categories | Categories appearing < 5% | Group into 'Other' |
| Categorical imbalance | Top category > 80% of rows | May cause model bias |
| High correlation | abs(r) ≥ 0.85 with another column | Consider dropping one |
| Duplicate column | Identical values to another column | Drop |

### Dataset-level
| Issue | Suggestion |
|---|---|
| Duplicate rows | Remove N duplicate rows |
| Overall missingness > 30% | Review data collection pipeline |
| Very small dataset (< 100 rows) | Results may not generalize |

---

## The report dict

When `return_report=True`, you get back:

```python
{
  "shape": (10843, 12),
  "missing_pct": 4.1,
  "duplicates": 23,
  "memory_mb": 3.2,
  "col_types": {
    "user_id": "ID_LIKE",
    "income": "NUMERIC_CONTINUOUS",
    "status": "NEAR_CONSTANT",
    ...
  },
  "profiles": {
    "income": {
      "missing_pct": 11.4,
      "mean": 52340.2,
      "median": 41000.0,
      "skew": 2.8,
      "outlier_count": 662,
      "outlier_pct": 6.1,
      ...
    },
    ...
  },
  "correlations": [
    ("age", "birth_year", 0.99),
    ("income", "salary", 0.91),
  ],
  "suggestions": {
    "user_id": ["Drop — unique identifier, no predictive value"],
    "income":  ["Impute with median — 11.4% missing", "Log-transform — skew 2.8"],
    ...
  },
  "global_suggestions": [
    "Remove 23 duplicate rows"
  ]
}
```

---

## Use in CI/CD pipelines

`return_report=True` makes `quick_eda` useful as a data quality gate. Example GitHub Actions step:

```python
# data_quality_check.py
import pandas as pd
import quick_eda
import sys

df = pd.read_csv("data/latest.csv")
report = quick_eda(df, mode="tldr", return_report=True)

if report["missing_pct"] > 20:
    print(f"FAIL: {report['missing_pct']}% missing values exceed threshold")
    sys.exit(1)

drop_suggestions = [
    col for col, suggestions in report["suggestions"].items()
    if any("Drop" in s for s in suggestions)
]
if drop_suggestions:
    print(f"WARNING: Columns flagged for review: {drop_suggestions}")

print("Data quality check passed.")
```

---

## Comparison

| | `quick_eda` | `ydata-profiling` | `sweetviz` | Manual |
|---|---|---|---|---|
| Time to run | < 2 seconds | 30–120 seconds | 10–30 seconds | 10–15 minutes |
| Output length | Short | 400+ scroll HTML | Long HTML | Varies |
| Actionable suggestions | ✅ | ❌ | ❌ | Only if you know what to look for |
| Works in notebook | ✅ | Partial | ❌ | ✅ |
| CI/CD ready | ✅ | ❌ | ❌ | ❌ |
| No config needed | ✅ | ❌ | Partial | — |

---

## Plots (full mode)

**Distributions** — histogram per numeric column. Skew annotated in red if abs(skew) > 1.

**Outliers** — boxplot per numeric column. Outlier count from IQR shown per box.

**Correlation heatmap** — lower triangle only, annotated. If > 15 numeric columns, shows top correlated pairs as a ranked list instead (heatmaps with 20+ columns are unreadable).

---

## Project structure

```
quick_eda/
├── quick_eda/
│   ├── __init__.py        # exposes quick_eda()
│   ├── core.py            # entry point, orchestrates pipeline
│   ├── classifier.py      # column type detection
│   ├── profiler.py        # per-column statistics
│   ├── relationships.py   # correlations, target analysis
│   ├── suggestions.py     # rule-based suggestion engine
│   ├── renderer.py        # all display/print logic
│   └── plots.py           # matplotlib/seaborn plots
├── tests/
├── examples/
│   └── demo.ipynb
├── setup.py
└── README.md
```

---

## Roadmap

**v0.1**
- [x] Core pipeline
- [x] Column classifier
- [x] Suggestion engine
- [x] 3 core plots
- [x] `return_report=True`
- [x] `mode="tldr"`
- [x] Auto-sampling

**v0.2**
- [x] Categorical plots (top-N bar charts)
- [x] Boolean plots
- [x] Duplicate columns

**v0.3** — Target Mode (current release)
- [x] `target=` full implementation
- [x] Class balance check
- [x] Target-aware suggestions
- [x] Target visualizations (feature correlations & distributions)

**v0.4** — planned
- [ ] Time series support (datetime-indexed DataFrames)
- [ ] HTML export: `quick_eda(df, export="report.html")`
- [ ] `quick_eda.assert_quality(df, max_missing=0.2)` for strict CI gates

---

## Contributing

Contributions welcome. Please open an issue before starting any major work so we can align on approach.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with tests
4. Run tests: `python -m pytest tests/`
5. Open a PR with a clear description

---

## License

MIT — do whatever you want with it.

---
## Author
Built by [Tarunpreet Singh](https://github.com/GodXSpell)
