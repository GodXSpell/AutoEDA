import pandas as pd
from IPython.display import display, HTML
from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE, CATEGORICAL_LOW,
    CATEGORICAL_HIGH, DATETIME, BOOLEAN, CONSTANT,
    NEAR_CONSTANT, ID_LIKE
)

"""
    This is the Renderer module responsible for displaying the EDA report in a clean and user-friendly format.
    It uses HTML and CSS styling to create a visually appealing report that highlights key insights and suggestions for the user.
    The main function is render_all(), which takes the full report dictionary and renders the banner, warnings, suggestions, and optionally the full stats table based on the selected mode (tldr or full).
"""

# ── severity grouping for warnings display ─────────────────────────────────
DROP_TYPES    = (CONSTANT, NEAR_CONSTANT, ID_LIKE)
HIGH_CONCERN  = ("missing_high", "correlation")
MOD_CONCERN   = ("skew", "outliers", "cardinality")


def _severity(col_type: str, suggestions: list) -> str:
    """Assign a severity level to a column based on its type and suggestions."""
    if col_type in DROP_TYPES:
        return "drop"
    first = suggestions[0].lower() if suggestions else ""
    if "consider dropping" in first or "missing" in first and "50%" in first:
        return "high"
    if any(w in first for w in ("impute", "correlation", "high correlation")):
        return "high"
    return "moderate"


def render_divider(title: str = ""):
    """Print a section divider with optional title."""
    if title:
        pad   = "━" * 6
        line  = f"{pad} {title} {pad}"
    else:
        line  = "━" * 42
    display(HTML(f"<pre style='font-family:monospace;color:var(--jp-content-font-color1,#333);margin:6px 0'>{line}</pre>"))


def render_banner(dataset_stats: dict):
    """Render the top summary banner — rows, cols, missing, duplicates, memory."""
    rows     = dataset_stats["rows"]
    cols     = dataset_stats["cols"]
    missing  = dataset_stats["total_missing_pct"]
    dupes    = dataset_stats["duplicate_rows"]
    memory   = dataset_stats["memory_mb"]
    num_c    = dataset_stats["numeric_cols"]
    cat_c    = dataset_stats["categorical_cols"]
    dt_c     = dataset_stats["datetime_cols"]

    missing_color = "#e03131" if missing > 10 else "#2f9e44" if missing == 0 else "#e67700"
    dupes_color   = "#e03131" if dupes > 0   else "#2f9e44"

    tile = lambda label, value, color="#3b5bdb": (
        f"<div style='background:#f0f4ff;border-radius:8px;padding:10px 20px;"
        f"text-align:center;min-width:80px'>"
        f"<div style='font-size:20px;font-weight:600;color:{color}'>{value}</div>"
        f"<div style='font-size:11px;color:#888;margin-top:2px'>{label}</div>"
        f"</div>"
    )

    html = f"""
    <div style='font-family:sans-serif;margin:8px 0 4px'>
      <div style='font-size:15px;font-weight:600;margin-bottom:8px'>
        &#128202; quick_eda
      </div>
      <div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px'>
        {tile("rows",        f"{rows:,}")}
        {tile("columns",     cols)}
        {tile("missing",     f"{missing}%",  missing_color)}
        {tile("duplicates",  dupes,          dupes_color)}
        {tile("memory",      f"{memory} MB")}
        {tile("numeric",     num_c,          "#7048e8")}
        {tile("categorical", cat_c,          "#7048e8")}
        {tile("datetime",    dt_c,           "#7048e8")}
      </div>
    </div>
    """
    display(HTML(html))


def render_warnings(col_suggestions: dict, col_types: dict):
    """
    Show only columns that have something wrong.
    Grouped by severity: DROP → HIGH → MODERATE.
    """
    if not col_suggestions:
        display(HTML("<p style='font-family:sans-serif;color:#2f9e44'>✓ No issues detected</p>"))
        return

    # group by severity
    groups = {"drop": [], "high": [], "moderate": []}
    for col, suggestions in col_suggestions.items():
        col_type = col_types.get(col, "")
        sev      = _severity(col_type, suggestions)
        groups[sev].append((col, suggestions))

    severity_config = {
        "drop":     ("🗑  DROP CANDIDATES",  "#e03131", "#fff5f5"),
        "high":     ("⚠  HIGH CONCERN",      "#e67700", "#fff8e1"),
        "moderate": ("ℹ  MODERATE CONCERN",  "#1971c2", "#e7f5ff"),
    }

    rows_html = ""
    for sev, (label, color, bg) in severity_config.items():
        cols_in_group = groups[sev]
        if not cols_in_group:
            continue

        rows_html += f"""
        <tr>
          <td colspan='2' style='padding:6px 10px;font-size:11px;font-weight:600;
              color:{color};background:{bg};border-radius:4px'>{label}</td>
        </tr>
        """
        for col, suggestions in cols_in_group:
            # join all suggestions for this column into one line for the warnings view
            summary = " · ".join(suggestions)
            rows_html += f"""
            <tr style='border-bottom:1px solid #f0f0f0'>
              <td style='padding:5px 14px;font-family:monospace;font-size:13px;
                  font-weight:600;white-space:nowrap;color:#333'>{col}</td>
              <td style='padding:5px 10px;font-size:12px;color:#555'>{summary}</td>
            </tr>
            """

    html = f"""
    <div style='font-family:sans-serif;margin:10px 0'>
      <table style='border-collapse:collapse;width:100%;max-width:800px'>
        {rows_html}
      </table>
    </div>
    """
    display(HTML(html))


def render_suggestions(col_suggestions: dict, global_suggestions: list):
    """
    Numbered list of all actionable suggestions.
    Global suggestions (dataset-level) come first.
    """
    all_suggestions = []

    # global first
    for s in global_suggestions:
        all_suggestions.append(("dataset", s))

    # then per-column
    for col, suggestions in col_suggestions.items():
        for s in suggestions:
            all_suggestions.append((col, s))

    if not all_suggestions:
        display(HTML("<p style='font-family:sans-serif;color:#2f9e44'>✓ No suggestions — dataset looks clean</p>"))
        return

    items_html = ""
    for i, (col, suggestion) in enumerate(all_suggestions, 1):
        col_label = (
            f"<span style='font-family:monospace;font-size:12px;"
            f"background:#f0f4ff;padding:1px 6px;border-radius:4px;"
            f"color:#3b5bdb;margin-right:6px'>{col}</span>"
            if col != "dataset" else ""
        )
        items_html += f"""
        <div style='display:flex;gap:10px;padding:5px 0;
            border-bottom:1px solid #f5f5f5;align-items:baseline'>
          <span style='font-size:11px;color:#aaa;min-width:20px;
              text-align:right;flex-shrink:0'>{i}.</span>
          <span style='font-size:13px;color:#333'>{col_label}{suggestion}</span>
        </div>
        """

    html = f"""
    <div style='font-family:sans-serif;margin:10px 0;max-width:800px'>
      {items_html}
    </div>
    """
    display(HTML(html))


def render_full_stats(profiles: dict, col_types: dict):
    """
    Per-column stat table — only shown in mode='full'.
    Numeric columns only. Clean table, not a raw df.describe() dump.
    """
    numeric_types = (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE)
    rows = []

    for col, profile in profiles.items():
        if col_types.get(col) not in numeric_types:
            continue
        if profile.get("skipped"):
            continue
        rows.append({
            "column":       col,
            "count":        profile.get("count"),
            "missing %":    profile.get("missing_pct"),
            "mean":         profile.get("mean"),
            "median":       profile.get("median"),
            "std":          profile.get("std"),
            "min":          profile.get("min"),
            "max":          profile.get("max"),
            "skew":         profile.get("skew"),
            "outliers %":   profile.get("outlier_pct"),
        })

    if not rows:
        display(HTML("<p style='font-family:sans-serif;color:#888'>No numeric columns to display.</p>"))
        return

    stats_df = pd.DataFrame(rows).set_index("column")
    display(HTML("<div style='font-family:sans-serif;font-size:12px;font-weight:600;"
                 "margin:8px 0 4px'>Numeric column stats</div>"))
    display(stats_df.style
        .format(precision=2, na_rep="—")
        .set_properties(**{"font-size": "12px"})
        .background_gradient(subset=["missing %"], cmap="Reds", vmin=0, vmax=50)
        .background_gradient(subset=["skew"],       cmap="RdYlGn_r", vmin=-3, vmax=3)
    )


def render_all(report: dict, mode: str = "full"):
    """
    Master render function. Called by core.py.
    Renders everything in the right order based on mode.

    mode='tldr'  → banner + warnings + suggestions only
    mode='full'  → everything including full stats table
    """
    render_banner(report["dataset_stats"])
    render_divider("WARNINGS")
    render_warnings(report["suggestions"], report["col_types"])
    render_divider("SUGGESTIONS")
    render_suggestions(report["suggestions"], report["global_suggestions"])

    if mode == "full":
        render_divider("COLUMN STATS")
        render_full_stats(report["profiles"], report["col_types"])