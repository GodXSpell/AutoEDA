import math
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .classifier import (
    NUMERIC_CONTINUOUS, NUMERIC_DISCRETE,
    CONSTANT, NEAR_CONSTANT, ID_LIKE
)

warnings.filterwarnings("ignore")

# ── palette ────────────────────────────────────────────────────────────────
PRIMARY = "#4361ee"
ACCENT  = "#e03131"
MUTED   = "#868e96"

"""
    This module contains functions for plotting various aspects of the dataset based on the classified column types and relationships.
    It includes functions for plotting distributions, outliers, and correlations.
    The main function is plot_all(), which is called by core.py to generate all relevant plots for the EDA report. Each plot function checks if there are enough numeric columns to plot and skips if not.
"""
def _set_style():
    """Global style applied to all plots."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom":True,
        "axes.grid":         True,
        "grid.color":        "#f0f0f0",
        "grid.linewidth":    0.6,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.titlepad":     8,
    })


def _get_numeric_cols(df: pd.DataFrame, col_types: dict) -> list:
    """Return numeric columns that are worth plotting."""
    return [
        col for col, t in col_types.items()
        if t in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE)
        and col in df.columns
    ]


def _make_grid(n: int, max_cols: int = 3):
    """Calculate rows and cols for a subplot grid."""
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    return rows, cols


def plot_distributions(df: pd.DataFrame, col_types: dict):
    """
    Histogram per numeric column.
    Skew annotated — gray if mild, red if high.
    """
    num_cols = _get_numeric_cols(df, col_types)
    if not num_cols:
        return

    _set_style()
    rows, cols = _make_grid(len(num_cols))
    fig, axes  = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    # always work with a flat list
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, col in zip(axes, num_cols):
        data = df[col].dropna()

        ax.hist(data, bins=30, color=PRIMARY, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=9)

        # skew annotation
        skew = float(data.skew())
        if abs(skew) > 1.0:
            ax.annotate(f"skew {skew:+.2f}", xy=(0.97, 0.93),
                        xycoords="axes fraction", ha="right",
                        fontsize=8, color=ACCENT,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#fff5f5", edgecolor="none"))
        elif abs(skew) > 0.5:
            ax.annotate(f"skew {skew:+.2f}", xy=(0.97, 0.93),
                        xycoords="axes fraction", ha="right",
                        fontsize=8, color=MUTED)

    # hide unused axes
    for ax in axes[len(num_cols):]:
        ax.set_visible(False)

    fig.suptitle("Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_outliers(df: pd.DataFrame, col_types: dict):
    """
    Boxplot per numeric column.
    Outlier count annotated on each box.
    """
    num_cols = _get_numeric_cols(df, col_types)
    if not num_cols:
        return

    _set_style()
    rows, cols = _make_grid(len(num_cols))
    fig, axes  = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, col in zip(axes, num_cols):
        data = df[col].dropna()

        ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color=ACCENT,   linewidth=2),
            boxprops=dict(facecolor=PRIMARY,  alpha=0.25),
            whiskerprops=dict(linewidth=0.8,  color=MUTED),
            capprops=dict(linewidth=0.8,      color=MUTED),
            flierprops=dict(marker="o",       markersize=3,
                            markerfacecolor=ACCENT,
                            alpha=0.4,        linestyle="none"),
        )
        ax.set_title(col)
        ax.set_xticks([])

        # count outliers via IQR and annotate
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr    = q3 - q1
        n_out  = int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())
        pct    = round(n_out / len(data) * 100, 1) if len(data) > 0 else 0

        if n_out > 0:
            color = ACCENT if pct > 5 else MUTED
            ax.annotate(f"{n_out} outliers ({pct}%)",
                        xy=(0.97, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8, color=color)

    for ax in axes[len(num_cols):]:
        ax.set_visible(False)

    fig.suptitle("Outlier Detection (IQR)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_correlation(df: pd.DataFrame, col_types: dict, high_corr_pairs: list):
    """
    If <= 15 numeric columns: full heatmap, lower triangle only.
    If > 15 numeric columns: horizontal bar chart of top correlated pairs instead.
    """
    num_cols = _get_numeric_cols(df, col_types)
    if len(num_cols) < 2:
        return

    _set_style()

    if len(num_cols) <= 15:
        # ── heatmap ───────────────────────────────────────────────────────
        corr   = df[num_cols].corr()
        size   = max(5, min(len(num_cols) * 0.9, 12))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        import numpy as np
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr, mask=mask, ax=ax,
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.4, linecolor="#f0f0f0",
            cbar_kws={"shrink": 0.7},
            square=True,
        )
        ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    else:
        # ── top pairs bar chart ───────────────────────────────────────────
        if not high_corr_pairs:
            return

        top_pairs = high_corr_pairs[:10]
        labels    = [f"{a} × {b}" for a, b, _ in top_pairs]
        values    = [abs(v) for _, _, v in top_pairs]
        colors    = [ACCENT if v >= 0.95 else PRIMARY for v in values]

        fig, ax = plt.subplots(figsize=(8, len(top_pairs) * 0.55 + 1))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="right",
                    fontsize=8, color="white", fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_xlabel("Absolute Correlation")
        ax.set_title("Top Correlated Pairs", fontsize=13, fontweight="bold")
        ax.axvline(0.85, color=ACCENT, linewidth=0.8,
                   linestyle="--", alpha=0.5, label="0.85 threshold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_target_correlations(target_correlations: list, target: str):
    """
    Horizontal bar chart of all features ranked by abs(correlation to target).
    Color by direction — blue for positive, coral for negative.
    """
    if not target_correlations:
        return

    _set_style()
    
    # target_correlations format: [(col_name, abs_val, direction, strength), ...]
    # Sort ascending for plotting bars bottom-up
    sorted_corrs = sorted(target_correlations, key=lambda x: x[1], reverse=False)
    
    cols = [x[0] for x in sorted_corrs]
    # For visualization, it's nice to plot signed magnitude if direction is known and not N/A
    # But prompt says "ranked by abs" and just "blue for positive, coral for negative"
    vals = [x[1] for x in sorted_corrs]
    
    colors = []
    for x in sorted_corrs:
        if x[2] == "negative":
            colors.append("coral")
        else:
            colors.append("#4361ee")

    fig, ax = plt.subplots(figsize=(8, max(4, len(cols) * 0.4)))
    
    bars = ax.barh(cols, vals, color=colors, alpha=0.8)
    ax.set_title(f"Feature Correlations with Target: '{target}'", fontsize=13, fontweight="bold")
    ax.set_xlabel("Absolute Correlation")
    
    # Annotate bars
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, 
                f"{bar.get_width():.2f}", va="center", ha="left", fontsize=9, color="black")
                
    plt.tight_layout()
    plt.show()


def plot_target_distributions(df: pd.DataFrame, target: str, col_types: dict, target_correlations: list):
    """
    For numeric target: color histograms by target quartile
    For categorical target (classification): overlay distributions per class
    Only top 4 most correlated features shown
    """
    if not target_correlations or target not in df.columns:
        return

    target_type = "classification" if df[target].nunique() <= 10 else "regression"
    
    # Get top 4 features by descending correlation magnitude
    top_4_cols = [x[0] for x in sorted(target_correlations, key=lambda x: x[1], reverse=True)[:4]]
    valid_cols = [c for c in top_4_cols if c in df.columns]
    
    if not valid_cols:
        return

    _set_style()
    rows, cols = _make_grid(len(valid_cols), max_cols=2)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, col in zip(axes, valid_cols):
        is_numeric = col_types.get(col) in (NUMERIC_CONTINUOUS, NUMERIC_DISCRETE)
        
        plot_df = df[[col, target]].dropna().copy()
        
        if target_type == "regression":
            # Continuous target: quantile split
            try:
                plot_df["target_q"] = pd.qcut(plot_df[target], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            except ValueError:
                # If too few unique values for qcut
                plot_df["target_q"] = plot_df[target]
                
            if is_numeric:
                sns.histplot(data=plot_df, x=col, hue="target_q", multiple="stack", palette="viridis", ax=ax, linewidth=0)
            else:
                sns.boxplot(data=plot_df, x="target_q", y=col, ax=ax, palette="viridis")
            ax.set_title(f"{col} by '{target}' Quartiles", fontsize=11, fontweight="bold")
            
        else:
            # Categorical target: plot distributions per target class
            if is_numeric:
                # overlay KDE or histograms
                sns.kdeplot(data=plot_df, x=col, hue=target, common_norm=False, fill=True, alpha=0.4, ax=ax, palette="Set1")
            else:
                sns.countplot(data=plot_df, y=col, hue=target, ax=ax, palette="Set1")
            ax.set_title(f"{col} by '{target}' Class", fontsize=11, fontweight="bold")

    for ax in axes[len(valid_cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Top 4 Corelated Feature Distributions by Target", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

def plot_all(df: pd.DataFrame, col_types: dict, correlations: list, target_correlations: list = None, target: str = None):
    """
    Master plot function called by core.py.
    Runs all plots in order.
    Skips any plot where there's not enough data.
    """
    plot_distributions(df, col_types)
    plot_outliers(df, col_types)
    plot_correlation(df, col_types, correlations)
    plot_categoricals(df, col_types)
    plot_boolean(df, col_types)
    
    if target and target_correlations:
        plot_target_correlations(target_correlations, target)
        plot_target_distributions(df, target, col_types, target_correlations)

def plot_categoricals(df: pd.DataFrame, col_types: dict):
    """
    Horizontal bar chart of top-N value counts for CATEGORICAL_LOW.
    """
    from .classifier import CATEGORICAL_LOW
    cat_cols = [col for col, t in col_types.items() if t == CATEGORICAL_LOW and col in df.columns]
    if not cat_cols:
        return

    _set_style()
    rows, cols = _make_grid(len(cat_cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, col in zip(axes, cat_cols):
        data = df[col].value_counts(dropna=False).head(10)
        
        # Calculate percentages
        total = len(df[col])
        pcts = (data / total * 100).round(1)

        # Plot bars
        bars = ax.barh(data.index.astype(str)[::-1], data.values[::-1], color=MUTED)
        if len(bars) > 0:
            bars[-1].set_color(ACCENT)  # highlight top bar

        for bar, pct in zip(bars, pcts.values[::-1]):
            ax.annotate(f"{pct}%", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)

        ax.set_title(col)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False)

    for ax in axes[len(cat_cols):]:
        ax.set_visible(False)

    fig.suptitle("Categorical Counts (Top 10)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

def plot_boolean(df: pd.DataFrame, col_types: dict):
    """
    Simple two-bar chart for BOOLEAN columns.
    """
    from .classifier import BOOLEAN
    bool_cols = [col for col, t in col_types.items() if t == BOOLEAN and col in df.columns]
    if not bool_cols:
        return

    _set_style()
    rows, cols = _make_grid(len(bool_cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, col in zip(axes, bool_cols):
        data = df[col].value_counts(dropna=False)
        
        total = len(df[col])
        pcts = (data / total * 100).round(1)
        
        bars = ax.bar(data.index.astype(str), data.values, color=MUTED)
        if len(bars) > 0:
            bars[0].set_color(ACCENT)

        for bar, pct in zip(bars, pcts.values):
            ax.annotate(f"{pct}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax.set_title(col)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', bottom=False)

    for ax in axes[len(bool_cols):]:
        ax.set_visible(False)

    fig.suptitle("Boolean Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
