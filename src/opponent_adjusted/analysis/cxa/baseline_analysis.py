"""Exploratory analysis for cxA baselines.

Reads the pass-attribution CSV produced by `scripts/build_cxa_baselines.py` and
exports:
- Aggregate tables by pass type/height/pressure and by team/player
- Pitch heatmaps for mean and total xA+ by pass end location

Outputs follow the standard layout: outputs/analysis/cxa/{csv,plots}/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.analysis.plot_utilities import (
    configure_matplotlib,
    draw_pitch,
    get_analysis_output_dir,
    plot_pitch_heatmap,
    save_figure,
    STYLE,
)
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Player, Team


@dataclass(frozen=True)
class BaselineAnalysisPaths:
    csv_dir: Path
    plots_dir: Path


def _output_dirs() -> BaselineAnalysisPaths:
    return BaselineAnalysisPaths(
        csv_dir=get_analysis_output_dir("cxa", subdir="csv"),
        plots_dir=get_analysis_output_dir("cxa", subdir="plots"),
    )


def _ensure_names(
    session: Optional[Session],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Optionally attach player/team names if a DB session is provided."""

    if session is None:
        return df

    out = df.copy()

    team_ids = sorted({int(x) for x in out["team_id"].dropna().unique().tolist()})
    player_ids = sorted({int(x) for x in out["player_id"].dropna().unique().tolist()})

    team_map: Dict[int, str] = {}
    if team_ids:
        for row in session.execute(select(Team.id, Team.name).where(Team.id.in_(team_ids))).all():
            team_map[int(row.id)] = str(row.name)

    player_map: Dict[int, str] = {}
    if player_ids:
        for row in session.execute(select(Player.id, Player.name).where(Player.id.in_(player_ids))).all():
            player_map[int(row.id)] = str(row.name)

    out["team_name"] = out["team_id"].map(team_map)
    out["player_name"] = out["player_id"].map(player_map)
    return out


def _agg_simple(df: pd.DataFrame, group_cols: list[str], *, top_n: Optional[int] = None) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False, observed=False)
        .agg(
            passes=("pass_event_id", "count"),
            shots=("shot_id", pd.Series.nunique),
            xa_plus_sum=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
            xa_keypass_sum=("xa_keypass", "sum"),
            shot_value_sum=("shot_value", "sum"),
        )
        .reset_index()
    )

    total = float(df["xa_plus"].sum()) if len(df) else 0.0
    agg["xa_plus_share"] = agg["xa_plus_sum"] / total if total > 0 else 0.0

    agg = agg.sort_values(["xa_plus_sum", "passes"], ascending=False)
    if top_n is not None:
        agg = agg.head(top_n)
    return agg


def _bin_mean_sum_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_grid, sum_grid, count_grid) over 2D bins."""

    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(values))
    if not np.any(mask):
        shape = (len(x_bins) - 1, len(y_bins) - 1)
        return (np.zeros(shape), np.zeros(shape), np.zeros(shape))

    xv = x[mask]
    yv = y[mask]
    vv = values[mask]

    counts, _, _ = np.histogram2d(xv, yv, bins=[x_bins, y_bins])
    sums, _, _ = np.histogram2d(xv, yv, bins=[x_bins, y_bins], weights=vv)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return (means, sums, counts)


def export_tables(df: pd.DataFrame, *, out_dir: Path, top_n: int = 30) -> None:
    # Pass attribute slices
    _agg_simple(df, ["pass_type"]).to_csv(out_dir / "cxa_baseline_by_pass_type.csv", index=False)
    _agg_simple(df, ["pass_height"]).to_csv(out_dir / "cxa_baseline_by_pass_height.csv", index=False)
    _agg_simple(df, ["under_pressure"]).to_csv(out_dir / "cxa_baseline_by_pressure.csv", index=False)
    _agg_simple(df, ["is_cross"]).to_csv(out_dir / "cxa_baseline_by_is_cross.csv", index=False)
    _agg_simple(df, ["is_through_ball"]).to_csv(out_dir / "cxa_baseline_by_is_through_ball.csv", index=False)

    # Players / teams
    player_cols = ["player_id"] + (["player_name"] if "player_name" in df.columns else [])
    team_cols = ["team_id"] + (["team_name"] if "team_name" in df.columns else [])

    _agg_simple(df, player_cols, top_n=top_n).to_csv(out_dir / "cxa_baseline_top_players.csv", index=False)
    _agg_simple(df, team_cols, top_n=None).to_csv(out_dir / "cxa_baseline_teams.csv", index=False)

    # Passes per shot distribution
    per_shot = (
        df.groupby("shot_id", as_index=False)
        .agg(
            passes_attributed=("pass_event_id", "count"),
            shot_value=("shot_value", "first"),
            xa_plus_sum=("xa_plus", "sum"),
        )
    )
    per_shot.to_csv(out_dir / "cxa_baseline_per_shot.csv", index=False)


def plot_end_location_heatmaps(
    df: pd.DataFrame,
    *,
    plots_dir: Path,
    x_bins: int = 25,
    y_bins: int = 17,
    min_bin_count: int = 5,
) -> None:
    if df.empty:
        return

    mask = df["pass_end_x"].notna() & df["pass_end_y"].notna() & df["xa_plus"].notna()
    filtered = df[mask]
    if filtered.empty:
        return

    xb = np.linspace(0, settings.pitch_length, x_bins)
    yb = np.linspace(0, settings.pitch_width, y_bins)

    mean_grid, sum_grid, count_grid = _bin_mean_sum_grid(
        filtered["pass_end_x"].to_numpy(dtype=float),
        filtered["pass_end_y"].to_numpy(dtype=float),
        filtered["xa_plus"].to_numpy(dtype=float),
        x_bins=xb,
        y_bins=yb,
    )

    # Mask low-volume bins so plots are not misleading.
    mean_show = mean_grid.copy()
    mean_show[count_grid < float(min_bin_count)] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(STYLE.fig_width * 1.7, STYLE.fig_height))

    for ax in axes:
        draw_pitch(ax)

    hm1 = plot_pitch_heatmap(axes[0], mean_show.T, cmap="PuBu")
    cbar1 = fig.colorbar(hm1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Mean xA+ (per pass end cell)")
    axes[0].set_title("Mean xA+ by pass end location")

    hm2 = plot_pitch_heatmap(axes[1], sum_grid.T, cmap="OrRd")
    cbar2 = fig.colorbar(hm2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Total xA+ (sum)")
    axes[1].set_title("Total xA+ by pass end location")

    save_figure(fig, analysis_type="cxa", name="cxa_baseline_pass_end_heatmaps", subdir="plots")


def run_cxa_baseline_analysis(
    baseline_csv: Path,
    *,
    session: Optional[Session] = None,
    top_n: int = 30,
) -> None:
    """Run baseline analysis from an existing baseline CSV."""

    configure_matplotlib()
    paths = _output_dirs()

    df = pd.read_csv(baseline_csv)
    if df.empty:
        print("No rows in baseline CSV:", baseline_csv)
        return

    df = _ensure_names(session, df)

    export_tables(df, out_dir=paths.csv_dir, top_n=top_n)
    plot_end_location_heatmaps(df, plots_dir=paths.plots_dir)

    print(f"Saved cxA baseline analysis outputs to {paths.csv_dir} and {paths.plots_dir}")
