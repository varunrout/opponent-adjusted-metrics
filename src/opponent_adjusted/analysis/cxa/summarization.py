"""Summarization utilities for cxA baselines.

Provides quick validation and summary statistics for baseline CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def summarize_cxa_baselines(
    df: pd.DataFrame,
    *,
    top_n: int = 15,
    verbose: bool = True,
) -> dict:
    """Summarize a cxA baseline DataFrame.

    Returns a dict of computed statistics.
    """
    stats: dict = {}

    stats["row_count"] = len(df)
    stats["unique_shots"] = int(df["shot_id"].nunique())
    stats["unique_passes"] = int(df["pass_event_id"].nunique())

    # Missingness
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    stats["missingness_pct"] = miss.to_dict()

    # xa_keypass summary
    stats["xa_keypass_summary"] = df["xa_keypass"].describe().to_dict()

    # xa_plus summary
    stats["xa_plus_summary"] = df["xa_plus"].describe().to_dict()

    # Per-shot conservation checks
    shot_totals = df.groupby("shot_id", as_index=False).agg(
        shot_value=("shot_value", "first"),
        xa_plus_sum=("xa_plus", "sum"),
        xa_keypass_sum=("xa_keypass", "sum"),
    )
    shot_totals["xa_plus_err"] = (shot_totals["xa_plus_sum"] - shot_totals["shot_value"]).abs()

    err_stats = shot_totals["xa_plus_err"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
    stats["conservation_error"] = err_stats

    bad_count = int((shot_totals["xa_plus_err"] > 1e-6).sum())
    stats["conservation_violations"] = bad_count

    # Top players
    top_players = (
        df.groupby("player_id", dropna=False)["xa_plus"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    stats["top_players_by_xa_plus"] = top_players.to_dict()

    # Top teams
    top_teams = df.groupby("team_id")["xa_plus"].sum().sort_values(ascending=False).head(top_n)
    stats["top_teams_by_xa_plus"] = top_teams.to_dict()

    if verbose:
        print(f"Rows: {stats['row_count']:,}")
        print(f"Unique shots: {stats['unique_shots']:,}")
        print(f"Unique passes: {stats['unique_passes']:,}")

        print("\nMissingness (%):")
        print(miss.head(12).round(2).to_string())

        print("\nxa_keypass summary:")
        print(df["xa_keypass"].describe().to_string())

        print("\nxa_plus summary:")
        print(df["xa_plus"].describe().to_string())

        print("\nPer-shot conservation checks:")
        print(
            shot_totals[["xa_plus_err"]]
            .describe(percentiles=[0.5, 0.9, 0.99])
            .round(8)
            .to_string()
        )

        if bad_count > 0:
            print(f"WARNING: {bad_count:,} shots have xa_plus_sum != shot_value (abs err > 1e-6)")

        print(f"\nTop {top_n} players by xa_plus:")
        print(top_players.to_string())

        print(f"\nTop {top_n} teams by xa_plus:")
        print(top_teams.to_string())

    return stats


def summarize_cxa_baselines_from_csv(
    path: str | Path,
    *,
    top_n: int = 15,
    verbose: bool = True,
) -> dict:
    """Load and summarize a cxA baseline CSV."""
    df = pd.read_csv(path)
    return summarize_cxa_baselines(df, top_n=top_n, verbose=verbose)
