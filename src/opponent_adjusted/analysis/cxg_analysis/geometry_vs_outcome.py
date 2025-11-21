"""CxG exploratory analysis: shot geometry vs outcomes and StatsBomb xG.

This script:
- Reads shots + shot_features (v1) from the configured database
- Computes empirical goal rates and mean StatsBomb xG vs distance/angle bins
- Produces a couple of diagnostic plots and an aggregated CSV under
  outputs/analysis/cxg/.

Usage (from repo root):

    poetry run python -m opponent_adjusted.analysis.cxg_analysis.geometry_vs_outcome

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import select

from opponent_adjusted.config import settings
from opponent_adjusted.db.session import SessionLocal
from opponent_adjusted.db.models import Shot, ShotFeature, Event
from opponent_adjusted.analysis.plot_utilities import (
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
    STYLE,
    draw_pitch,
    plot_pitch_heatmap,
    compute_goal_rate_grid,
)
import matplotlib.pyplot as plt


@dataclass
class ShotRecord:
    shot_id: int
    is_goal: bool
    shot_distance: float | None
    shot_angle: float | None
    statsbomb_xg: float | None
    location_x: float | None
    location_y: float | None


def load_shot_data(feature_version: str = "v1") -> pd.DataFrame:
    """Load shot + feature data into a pandas DataFrame.

    Columns:
        shot_id, is_goal, shot_distance, shot_angle, statsbomb_xg
    """

    with SessionLocal() as session:
        stmt = (
            select(
                Shot.id.label("shot_id"),
                Shot.outcome,
                Shot.statsbomb_xg,
                ShotFeature.shot_distance,
                ShotFeature.shot_angle,
                Event.location_x,
                Event.location_y,
            )
            .join(ShotFeature, ShotFeature.shot_id == Shot.id)
            .join(Event, Event.id == Shot.event_id)
            .where(ShotFeature.version_tag == feature_version)
        )

        rows = session.execute(stmt).all()

    records: List[ShotRecord] = []
    for r in rows:
        is_goal = (r.outcome or "").lower() == "goal"
        records.append(
            ShotRecord(
                shot_id=r.shot_id,
                is_goal=is_goal,
                shot_distance=float(r.shot_distance) if r.shot_distance is not None else None,
                shot_angle=float(r.shot_angle) if r.shot_angle is not None else None,
                statsbomb_xg=float(r.statsbomb_xg) if r.statsbomb_xg is not None else None,
                location_x=float(r.location_x) if r.location_x is not None else None,
                location_y=float(r.location_y) if r.location_y is not None else None,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    # Drop rows without geometry or location
    df = df.dropna(subset=["shot_distance", "shot_angle", "location_x", "location_y"]).reset_index(drop=True)
    return df


def bin_and_aggregate_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate goal rate and mean xG by distance bin."""

    # Define distance bins (in StatsBomb units; roughly meters if pitch length=120)
    bins = np.arange(0, 40 + 5, 5)  # 0-5, 5-10, ..., 35-40+
    df["distance_bin"] = pd.cut(df["shot_distance"], bins=bins, right=False, include_lowest=True)

    agg = (
        df.groupby("distance_bin", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    agg["goal_rate"] = agg["goals"] / agg["shots"].clip(lower=1)
    # Bin midpoint for plotting
    agg["distance_mid"] = agg["distance_bin"].apply(
        lambda interval: (interval.left + interval.right) / 2.0 if interval is not np.nan else np.nan
    )
    return agg


def bin_and_aggregate_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate goal rate by shot angle bin."""

    # Angle is in radians [0, pi]; use ~10 bins
    bins = np.linspace(0, np.pi, 11)
    df["angle_bin"] = pd.cut(df["shot_angle"], bins=bins, right=False, include_lowest=True)

    agg = (
        df.groupby("angle_bin", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
        )
        .reset_index()
    )
    agg["goal_rate"] = agg["goals"] / agg["shots"].clip(lower=1)
    agg["angle_mid"] = agg["angle_bin"].apply(
        lambda interval: (interval.left + interval.right) / 2.0 if interval is not np.nan else np.nan
    )
    return agg


def plot_distance_vs_goal(agg: pd.DataFrame) -> None:
    """Create and save distance vs goal-rate/xG plot."""

    fig, ax = plt.subplots()

    ax.plot(
        agg["distance_mid"],
        agg["goal_rate"],
        marker="o",
        color=STYLE.primary_color,
        label="Empirical goal rate",
    )

    ax.plot(
        agg["distance_mid"],
        agg["mean_xg"],
        marker="s",
        color=STYLE.secondary_color,
        label="Mean StatsBomb xG",
    )

    ax.set_xlabel("Shot distance (StatsBomb units)")
    ax.set_ylabel("Rate")
    ax.set_title("Goal rate vs distance (with StatsBomb xG)")
    ax.legend()

    save_figure(fig, analysis_type="cxg", name="geometry_distance_vs_goal")


def plot_angle_vs_goal(agg: pd.DataFrame) -> None:
    """Create and save angle vs goal-rate plot."""

    fig, ax = plt.subplots()

    ax.plot(
        agg["angle_mid"],
        agg["goal_rate"],
        marker="o",
        color=STYLE.primary_color,
        label="Empirical goal rate",
    )

    ax.set_xlabel("Shot angle (radians)")
    ax.set_ylabel("Goal rate")
    ax.set_title("Goal rate vs shot angle")
    ax.legend()

    save_figure(fig, analysis_type="cxg", name="geometry_angle_vs_goal")


def export_distance_table(agg: pd.DataFrame) -> None:
    """Export aggregated distance-bin table as CSV under outputs/analysis/cxg/csv."""

    out_dir = get_analysis_output_dir("cxg", subdir="csv")
    path = out_dir / "geometry_distance_bins.csv"
    agg.to_csv(path, index=False)


def plot_pitch_goal_rate_heatmap(df: pd.DataFrame) -> None:
    """Render goal rate by shot location on a pitch heatmap."""

    pitch_length = settings.pitch_length
    pitch_width = settings.pitch_width

    # Use a modest grid resolution to keep visualization readable
    x_bins = np.linspace(0, pitch_length, 25)
    y_bins = np.linspace(0, pitch_width, 17)

    goal_rate = compute_goal_rate_grid(
        df["location_x"].to_numpy(),
        df["location_y"].to_numpy(),
        df["is_goal"].astype(float).to_numpy(),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    fig, ax = plt.subplots()
    draw_pitch(ax)
    heat = plot_pitch_heatmap(ax, goal_rate.T, cmap="OrRd")
    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Goal rate")
    ax.set_title("Shot goal rate heatmap")

    save_figure(fig, analysis_type="cxg", name="geometry_pitch_goal_rate")


def plot_distance_angle_heatmap(df: pd.DataFrame) -> None:
    """Visualize goal rate across distance/angle bins."""

    # Distance bins capture the bulk of open-play shots (~0-60 SB units)
    dist_bins = np.linspace(0, 60, 25)
    angle_bins = np.linspace(0, np.pi, 25)

    counts, _, _ = np.histogram2d(
        df["shot_distance"],
        df["shot_angle"],
        bins=[dist_bins, angle_bins],
        range=[[dist_bins[0], dist_bins[-1]], [angle_bins[0], angle_bins[-1]]],
    )
    goals, _, _ = np.histogram2d(
        df["shot_distance"],
        df["shot_angle"],
        bins=[dist_bins, angle_bins],
        range=[[dist_bins[0], dist_bins[-1]], [angle_bins[0], angle_bins[-1]]],
        weights=df["is_goal"].astype(float),
    )

    goal_rate = np.full_like(goals, np.nan, dtype=float)
    np.divide(goals, counts, out=goal_rate, where=counts > 0)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        np.rad2deg(angle_bins),
        dist_bins,
        goal_rate,
        cmap="plasma",
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Goal rate")
    ax.set_xlabel("Shot angle (degrees)")
    ax.set_ylabel("Shot distance (StatsBomb units)")
    ax.set_title("Goal rate heatmap by distance/angle")

    save_figure(fig, analysis_type="cxg", name="geometry_angle_distance_heatmap")


def run_geometry_vs_outcome_analysis(feature_version: str = "v1") -> None:
    """Entry point for the geometry vs outcome analysis."""

    configure_matplotlib()
    df = load_shot_data(feature_version=feature_version)

    if df.empty:
        print("No shot data found for feature version", feature_version)
        return

    dist_agg = bin_and_aggregate_distance(df)
    angle_agg = bin_and_aggregate_angle(df)

    plot_distance_vs_goal(dist_agg)
    plot_angle_vs_goal(angle_agg)
    export_distance_table(dist_agg)
    plot_pitch_goal_rate_heatmap(df)
    plot_distance_angle_heatmap(df)


if __name__ == "__main__":
    run_geometry_vs_outcome_analysis()
