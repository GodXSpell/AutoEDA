from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE, CATEGORICAL_LOW,
    CATEGORICAL_HIGH, DATETIME, BOOLEAN, CONSTANT,
    NEAR_CONSTANT, ID_LIKE
)

"""
    This module generates plain-English suggestions for each column based on its profile and relationships.
    Suggestions are categorized by priority: DROP candidates first, then HIGH concern issues, then MODERATE concerns, and finally INFO-level notes.
    The main function is suggest_for_dataframe(), which takes the column profiles, types, and correlations, and returns a dictionary of suggestions for each column that has at least one suggestion.
"""
def suggest_for_column(
    col_name:          str,
    col_type:          str,
    profile:           dict,
    correlation_flags: list   # list of (col_a, col_b, corr_val) pairs involving this column
) -> list:
    """
    Generate plain-English suggestions for a single column.
    Returns a list of suggestion strings.
    Priority order: DROP first, then HIGH concern, then MODERATE, then INFO.
    """
    suggestions = []

    # ── Priority 1: DROP candidates ───────────────────────────────────────

    if col_type == CONSTANT:
        suggestions.append("Drop — constant column, carries no information")
        return suggestions  # no point checking anything else

    if col_type == NEAR_CONSTANT:
        top_val  = max(profile.get("top_values", {None: 0}), key=profile.get("top_values", {}).get, default="?") if "top_values" in profile else "?"
        top_pct  = profile.get("unique_count", "?")
        missing  = profile.get("missing_pct", 0)
        suggestions.append(f"Consider dropping — near-constant, one value dominates {100 - missing:.0f}% of rows")
        return suggestions

    if col_type == BOOLEAN:
        top_values = profile.get("top_values", {})
        if top_values:
            top_val = max(top_values, key=top_values.get)
            top_pct = top_values[top_val]
            if top_pct > 95:
                suggestions.append(f"Consider dropping — near-constant, one value dominates {top_pct:.0f}% of rows")
                return suggestions

    if col_type == ID_LIKE:
        suggestions.append("Drop before modeling — appears to be a unique row identifier")
        return suggestions

    # ── Priority 2: HIGH concern ──────────────────────────────────────────

    missing_pct = profile.get("missing_pct", 0)

    if missing_pct > 50:
        suggestions.append(f"Consider dropping — {missing_pct}% of values are missing")
        return suggestions

    elif missing_pct > 20:
        suggestions.append(f"Impute with care — {missing_pct}% missing, high enough to introduce bias")

    elif missing_pct > 0:
        if col_type in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE):
            suggestions.append(f"Impute with median — {missing_pct}% missing")
        elif col_type in (CATEGORICAL_LOW, CATEGORICAL_HIGH):
            suggestions.append(f"Impute with mode — {missing_pct}% missing")

    # high correlation with another column
    for col_a, col_b, corr_val in correlation_flags:
        other = col_b if col_a == col_name else col_a
        suggestions.append(
            f"High correlation with '{other}' ({corr_val:.2f}) — consider dropping one before modeling"
        )

    # ── Priority 3: MODERATE concern ─────────────────────────────────────

    if col_type in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE):

        skew = profile.get("skew", 0)
        if abs(skew) > 1.0:
            direction = "right" if skew > 0 else "left"
            if profile.get("min", 0) >= 0:
                suggestions.append(
                    f"Apply log transform — {direction}-skewed distribution (skew {skew:+.2f})"
                )
            else:
                suggestions.append(
                    f"Consider power transform (Box-Cox) — column has negative values, log transform not applicable"
                )
        elif abs(skew) > 0.5:
            suggestions.append(f"Mildly skewed (skew {skew:+.2f}) — monitor, may not need transformation")

        outlier_pct = profile.get("outlier_pct", 0)
        if outlier_pct > 10:
            suggestions.append(
                f"Investigate outliers — {outlier_pct}% of values fall outside IQR bounds"
            )
        elif outlier_pct > 5:
            suggestions.append(
                f"Some outliers present — {outlier_pct}% beyond IQR bounds, review before modeling"
            )

        zeros_pct = profile.get("zeros_pct", 0)
        if zeros_pct > 30 and col_type == NUMERIC_CONTINUOUS:
            suggestions.append(
                f"High zero rate ({zeros_pct}%) — check if zeros are valid or represent missing data"
            )

    if col_type == CATEGORICAL_HIGH:
        unique_count = profile.get("unique_count", 0)
        rare_pct = profile.get("rare_category_pct", 0)
        imbalance = profile.get("imbalance_ratio", 0)

        if unique_count > 50 and rare_pct > 20:
            suggestions.append(
                f"Many rare categories ({rare_pct}% of rows) — consider grouping into 'Other'"
            )
        elif unique_count > 50 and imbalance < 10:
            suggestions.append(
                "High cardinality — consider frequency encoding or target encoding"
            )
        else:
            suggestions.append(
                f"High cardinality ({unique_count} unique values) — consider grouping rare categories or target encoding"
            )

    if col_type in (CATEGORICAL_LOW, CATEGORICAL_HIGH):
        top_values = profile.get("top_values", {})
        if top_values:
            top_val = max(top_values, key=top_values.get)
            top_pct = top_values[top_val]
            if 80 < top_pct < 98:
                suggestions.append(f"Imbalanced — '{top_val}' dominates {top_pct:.0f}% of rows, may cause model bias")

    # ── Priority 4: INFO ──────────────────────────────────────────────────

    if col_type in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE):
        kurtosis = profile.get("kurtosis", 0)
        if kurtosis > 3:
            suggestions.append(
                f"Heavy-tailed distribution (kurtosis {kurtosis:.2f}) — extreme values may affect model"
            )

    return suggestions


def suggest_for_dataframe(
    profiles:     dict,
    col_types:    dict,
    correlations: list
) -> dict:
    """
    Run suggestions for every column in the dataframe.
    Returns {col_name: [suggestions]} — only includes columns that have at least one suggestion.
    """
    results = {}

    for col, profile in profiles.items():
        col_type = col_types.get(col, "")

        # find correlation pairs that involve this column
        col_correlations = [
            (a, b, v) for a, b, v in correlations
            if a == col or b == col
        ]

        suggestions = suggest_for_column(col, col_type, profile, col_correlations)

        if suggestions:
            results[col] = suggestions

    return results


def get_global_suggestions(dataset_stats: dict) -> list:
    """
    Dataset-level suggestions — not tied to any specific column.
    """
    suggestions = []

    duplicates = dataset_stats.get("duplicate_rows", 0)
    if duplicates > 0:
        suggestions.append(f"Remove {duplicates} duplicate rows")

    missing_pct = dataset_stats.get("total_missing_pct", 0)
    if missing_pct > 30:
        suggestions.append(
            f"Dataset has high overall missingness ({missing_pct}%) — review data collection pipeline"
        )

    rows = dataset_stats.get("rows", 0)
    if rows < 100:
        suggestions.append(
            f"Small dataset ({rows} rows) — results may not generalize, treat insights with caution"
        )

    return suggestions