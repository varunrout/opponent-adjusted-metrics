"""Matplotlib-only plotting helpers for analysis reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from opponent_adjusted.analysis.shared.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _prepare_path(path: Path | str) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    return resolved


def save_histogram(
    series: pd.Series,
    path: Path | str,
    title: str,
    xlabel: str,
) -> Path:
    """Save a histogram for a numeric series."""

    resolved = _prepare_path(path)
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if values.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=30, color="#2f6f9f", edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Shots")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(resolved, dpi=150)
    plt.close(fig)
    return resolved


def save_ranked_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    path: Path | str,
    title: str,
    top_n: int = 20,
) -> Path:
    """Save a horizontal ranked bar chart."""

    resolved = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(9, 6))
    if df.empty or label_col not in df or value_col not in df:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_df = df[[label_col, value_col]].copy()
        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[value_col]).sort_values(value_col, ascending=False)
        plot_df = plot_df.head(top_n).sort_values(value_col, ascending=True)
        ax.barh(plot_df[label_col].astype(str), plot_df[value_col], color="#477b52")
        ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(resolved, dpi=150)
    plt.close(fig)
    return resolved


def save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path | str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Save a scatter plot."""

    resolved = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.empty or x_col not in df or y_col not in df:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.any():
            ax.scatter(x[valid], y[valid], alpha=0.65, color="#7a4f9a")
            ax.set_xlabel(xlabel or x_col)
            ax.set_ylabel(ylabel or y_col)
        else:
            ax.text(0.5, 0.5, "No numeric data available", ha="center", va="center")
            ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(resolved, dpi=150)
    plt.close(fig)
    return resolved
