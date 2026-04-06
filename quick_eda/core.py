import pandas as pd
from .classifier   import classify_dataframe
from .profiler     import profile_dataframe, get_dataset_stats
from .relationships import compute_correlations, correlate_with_target, detect_duplicate_columns
from .suggestions  import suggest_for_dataframe, get_global_suggestions
from .renderer     import render_all
from .plots        import plot_all


def quick_eda(
    df,
    mode         = "full",
    target       = None,
    plots        = True,
    sample       = True,
    sample_size  = 50_000,
    return_report= False,
):
    """
    Run a minimal EDA report inline in Jupyter.

    Parameters
    ----------
    df            : pandas DataFrame
    mode          : "full" (default) or "tldr" (no plots, no stats table)
    target        : column name to treat as prediction target (optional)
    plots         : show plots — only applies in mode="full" (default True)
    sample        : auto-sample large datasets (default True)
    sample_size   : row limit when sampling (default 50,000)
    return_report : return the report as a dict (default False)

    Examples
    --------
    >>> quick_eda(df)
    >>> quick_eda(df, mode="tldr")
    >>> quick_eda(df, target="price")
    >>> report = quick_eda(df, return_report=True)
    """

    # ── input validation ───────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "quick_eda() expects a pandas DataFrame. "
            f"Got {type(df).__name__} instead."
        )

    if df.empty:
        raise ValueError("DataFrame is empty — nothing to analyze.")

    if df.shape[1] < 2:
        raise ValueError(
            "DataFrame has only 1 column. "
            "quick_eda needs at least 2 columns to be useful."
        )

    if mode not in ("full", "tldr"):
        raise ValueError(
            f"Invalid mode '{mode}'. Choose 'full' or 'tldr'."
        )

    if target is not None and target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # ── sampling ───────────────────────────────────────────────────────────
    working_df = df.copy()

    if sample and len(df) > sample_size:
        working_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(
            f"⚡ Large dataset detected ({len(df):,} rows) — "
            f"running on {sample_size:,} row sample. "
            f"Pass sample=False to use full data."
        )

    # ── pipeline ───────────────────────────────────────────────────────────

    # step 1 — classify columns
    col_types = classify_dataframe(working_df)

    # step 2 — profile columns + dataset stats
    profiles      = profile_dataframe(working_df, col_types)
    dataset_stats = get_dataset_stats(working_df)

    # step 3 — relationships
    correlations       = compute_correlations(working_df, col_types)
    duplicate_cols     = detect_duplicate_columns(working_df)
    target_correlations = []
    if target is not None:
        target_correlations = correlate_with_target(working_df, target, col_types)

    # step 4 — suggestions
    col_suggestions    = suggest_for_dataframe(profiles, col_types, correlations)
    global_suggestions = get_global_suggestions(dataset_stats)

    # flag duplicate columns in suggestions
    for col_a, col_b in duplicate_cols:
        msg = f"Identical values to '{col_b}' — drop one before modeling"
        if col_a in col_suggestions:
            col_suggestions[col_a].insert(0, msg)
        else:
            col_suggestions[col_a] = [msg]

    # ── build report dict ──────────────────────────────────────────────────
    report = {
        "shape":               (working_df.shape[0], working_df.shape[1]),
        "col_types":           col_types,
        "profiles":            profiles,
        "dataset_stats":       dataset_stats,
        "correlations":        correlations,
        "duplicate_cols":      duplicate_cols,
        "target_correlations": target_correlations,
        "suggestions":         col_suggestions,
        "global_suggestions":  global_suggestions,
    }

    # ── render ─────────────────────────────────────────────────────────────
    render_all(report, mode=mode)

    if plots and mode != "tldr":
        plot_all(working_df, col_types, correlations)

    # ── return ─────────────────────────────────────────────────────────────
    if return_report:
        return report

    return None