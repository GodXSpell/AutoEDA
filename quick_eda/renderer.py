import pandas as pd
from IPython.display import display, HTML
from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE, CATEGORICAL_LOW,
    CATEGORICAL_HIGH, DATETIME, BOOLEAN, CONSTANT,
    NEAR_CONSTANT, ID_LIKE
)

DROP_TYPES = (CONSTANT, NEAR_CONSTANT, ID_LIKE)


def _severity(col_type: str, suggestions: list) -> str:
    if col_type in DROP_TYPES:
        return "drop"
    first = suggestions[0].lower() if suggestions else ""
    if any(w in first for w in ("consider dropping", "impute with care", "high correlation")):
        return "high"
    return "moderate"


def _divider(title: str = "", width: int = 52):
    if title:
        pad  = "─" * 3
        line = f"{pad} {title} {pad}"
    else:
        line = "─" * width
    print(line)


def _banner(dataset_stats: dict):
    rows    = dataset_stats["rows"]
    cols    = dataset_stats["cols"]
    missing = dataset_stats["total_missing_pct"]
    dupes   = dataset_stats["duplicate_rows"]
    memory  = dataset_stats["memory_mb"]
    num_c   = dataset_stats["numeric_cols"]
    cat_c   = dataset_stats["categorical_cols"]
    dt_c    = dataset_stats["datetime_cols"]

    print()
    print("  quick_eda")
    _divider(width=52)
    print(f"  rows        {rows:>10,}")
    print(f"  columns     {cols:>10}")
    print(f"  missing     {missing:>9.1f}%"  + ("  !" if missing > 10 else ""))
    print(f"  duplicates  {dupes:>10,}"       + ("  !" if dupes > 0 else ""))
    print(f"  memory      {memory:>9.2f} MB")
    print(f"  numeric     {num_c:>10}")
    print(f"  categorical {cat_c:>10}")
    print(f"  datetime    {dt_c:>10}")
    _divider(width=52)
    print()


def _warnings(col_suggestions: dict, col_types: dict):
    # print("  WARNINGS")
    print()

    if not col_suggestions:
        print("  ✓ No issues detected")
        print()
        return

    # group by severity
    groups = {"drop": [], "high": [], "moderate": []}
    for col, suggestions in col_suggestions.items():
        col_type = col_types.get(col, "")
        sev      = _severity(col_type, suggestions)
        groups[sev].append((col, suggestions))

    labels = {
        "drop":     "  DROP",
        "high":     "  HIGH CONCERN",
        "moderate": "  MODERATE",
    }

    for sev, group_cols in groups.items():
        if not group_cols:
            continue
        print(labels[sev])
        for col, suggestions in group_cols:
            # first suggestion as the one-liner summary
            summary = suggestions[0]
            print(f"    {col:<30}  {summary}")
        print()


def _suggestions(col_suggestions: dict, global_suggestions: list):
    # print("  SUGGESTIONS")
    print()

    all_suggestions = []

    for s in global_suggestions:
        all_suggestions.append(("dataset", s))

    for col, suggestions in col_suggestions.items():
        for s in suggestions:
            all_suggestions.append((col, s))

    if not all_suggestions:
        print("  ✓ No suggestions — dataset looks clean")
        print()
        return

    for i, (col, suggestion) in enumerate(all_suggestions, 1):
        if col == "dataset":
            print(f"  {i:>2}.  {suggestion}")
        else:
            print(f"  {i:>2}.  [{col}]  {suggestion}")

    print()


def _full_stats(profiles: dict, col_types: dict):
    numeric_types = (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE)
    cat_types = (CATEGORICAL_LOW, CATEGORICAL_HIGH, BOOLEAN)
    num_rows = []
    cat_rows = []

    for col, profile in profiles.items():
        if profile.get("skipped"):
            continue
        c_type = col_types.get(col)
        if c_type in numeric_types:
            num_rows.append({
                "column":     col,
                "count":      profile.get("count"),
                "missing %":  profile.get("missing_pct"),
                "mean":       profile.get("mean"),
                "median":     profile.get("median"),
                "std":        profile.get("std"),
                "min":        profile.get("min"),
                "max":        profile.get("max"),
                "skew":       profile.get("skew"),
                "outliers %": profile.get("outlier_pct"),
            })
        elif c_type in cat_types:
            top_values = profile.get("top_values", {})
            top_val = next(iter(top_values.keys())) if top_values else None
            top_pct = top_values.get(top_val, 0) if top_values else 0
            cat_rows.append({
                "column":       col,
                "unique count": profile.get("unique_count"),
                "top value":    top_val,
                "top value %":  top_pct,
                "rare category %": profile.get("rare_category_pct", 0),
                "entropy":      profile.get("entropy", 0),
            })

    if num_rows:
        print()
        print("  NUMERIC STATS")
        print()
        stats_df = pd.DataFrame(num_rows).set_index("column")
        print(stats_df.to_string())
        print()

    if cat_rows:
        print()
        print("  CATEGORICAL STATS")
        print()
        cat_df = pd.DataFrame(cat_rows).set_index("column")
        print(cat_df.to_string())
        print()

    if not num_rows and not cat_rows:
        print("  No columns to display.")
        print()
        return


def _target_correlations(target: str, target_correlations: list):
    _divider(f"FEATURE RELEVANCE (target: {target})")
    print()
    if not target_correlations:
        print("    No correlations found.")
        print()
        return

    for col, val, direction, strength in target_correlations:
        if direction == "N/A":
            label = strength
        else:
            label = f"{strength} {direction}"
            
        note = "  ← low predictive value" if strength == "near zero" else ""
        print(f"    {label:<17} {col:<15} {val:.2f}{note}")
    print()

def render_all(report: dict, mode: str = "full", target: str = None):
    """
    Master render function.
    mode='tldr'  → banner + warnings + suggestions only
    mode='full'  → everything including stats table
    """
    _banner(report["dataset_stats"])
    _divider("WARNINGS")
    _warnings(report["suggestions"], report["col_types"])
    
    if target and report.get("target_correlations"):
        _target_correlations(target, report["target_correlations"])
        
    _divider("SUGGESTIONS")
    _suggestions(report["suggestions"], report["global_suggestions"])

    if mode == "full":
        _divider("COLUMN STATS")
        print()
        _full_stats(report["profiles"], report["col_types"])