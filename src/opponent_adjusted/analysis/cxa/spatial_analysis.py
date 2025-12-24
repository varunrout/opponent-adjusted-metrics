"""Spatial analysis for cxA.

Provides:
- Pass origin zone analysis (where do high-xA passes start?)
- Pass end zone analysis (already in baseline, enhanced here)
- Corridor analysis (start zone → end zone combinations)
- Zone-binned xA tables

Outputs: CSV tables and plots under outputs/analysis/cxa/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from opponent_adjusted.db.models import Event


# ---------------------------------------------------------------------------
# Zone Definitions
# ---------------------------------------------------------------------------

# Pitch zones (StatsBomb coordinates: 120 x 80)
# X zones: Defensive Third (0-40), Middle Third (40-80), Attacking Third (80-120)
# Y zones: Left (0-26.67), Center (26.67-53.33), Right (53.33-80)

X_ZONE_EDGES = [0, 40, 80, 120]
Y_ZONE_EDGES = [0, 26.67, 53.33, 80]

X_ZONE_LABELS = ["Def Third", "Mid Third", "Att Third"]
Y_ZONE_LABELS = ["Left", "Center", "Right"]


def _get_zone(x: float, y: float) -> Tuple[str, str]:
    """Get zone labels for a coordinate."""
    x_zone = "Unknown"
    for i, (low, high) in enumerate(zip(X_ZONE_EDGES[:-1], X_ZONE_EDGES[1:])):
        if low <= x < high:
            x_zone = X_ZONE_LABELS[i]
            break
    if x >= X_ZONE_EDGES[-1]:
        x_zone = X_ZONE_LABELS[-1]

    y_zone = "Unknown"
    for i, (low, high) in enumerate(zip(Y_ZONE_EDGES[:-1], Y_ZONE_EDGES[1:])):
        if low <= y < high:
            y_zone = Y_ZONE_LABELS[i]
            break
    if y >= Y_ZONE_EDGES[-1]:
        y_zone = Y_ZONE_LABELS[-1]

    return x_zone, y_zone


def _zone_label(x_zone: str, y_zone: str) -> str:
    """Combine zone labels."""
    return f"{x_zone} - {y_zone}"


# ---------------------------------------------------------------------------
# Data Enrichment
# ---------------------------------------------------------------------------


def enrich_with_pass_start(
    df: pd.DataFrame,
    session: Session,
) -> pd.DataFrame:
    """Add pass start coordinates from events table."""

    pass_event_ids = df["pass_event_id"].dropna().unique().tolist()

    # Fetch start locations
    start_locs: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

    for row in session.execute(
        select(Event.id, Event.location_x, Event.location_y).where(
            Event.id.in_([int(x) for x in pass_event_ids])
        )
    ).all():
        start_locs[int(row.id)] = (row.location_x, row.location_y)

    df = df.copy()
    df["pass_start_x"] = df["pass_event_id"].map(lambda x: start_locs.get(int(x), (None, None))[0] if pd.notna(x) else None)
    df["pass_start_y"] = df["pass_event_id"].map(lambda x: start_locs.get(int(x), (None, None))[1] if pd.notna(x) else None)

    return df


def add_zone_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add zone labels for start and end locations."""

    df = df.copy()

    # Start zones
    def get_start_zone(row):
        if pd.isna(row["pass_start_x"]) or pd.isna(row["pass_start_y"]):
            return "Unknown", "Unknown"
        return _get_zone(float(row["pass_start_x"]), float(row["pass_start_y"]))

    start_zones = df.apply(get_start_zone, axis=1)
    df["start_x_zone"] = [z[0] for z in start_zones]
    df["start_y_zone"] = [z[1] for z in start_zones]
    df["start_zone"] = df.apply(lambda r: _zone_label(r["start_x_zone"], r["start_y_zone"]), axis=1)

    # End zones
    def get_end_zone(row):
        if pd.isna(row["pass_end_x"]) or pd.isna(row["pass_end_y"]):
            return "Unknown", "Unknown"
        return _get_zone(float(row["pass_end_x"]), float(row["pass_end_y"]))

    end_zones = df.apply(get_end_zone, axis=1)
    df["end_x_zone"] = [z[0] for z in end_zones]
    df["end_y_zone"] = [z[1] for z in end_zones]
    df["end_zone"] = df.apply(lambda r: _zone_label(r["end_x_zone"], r["end_y_zone"]), axis=1)

    return df


# ---------------------------------------------------------------------------
# Zone Analysis
# ---------------------------------------------------------------------------


def compute_zone_stats(df: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    """Compute xA statistics by zone."""

    stats = (
        df.groupby(zone_col, dropna=False)
        .agg(
            passes=("pass_event_id", "count"),
            shots=("shot_id", pd.Series.nunique),
            xa_plus_sum=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
            xa_keypass_sum=("xa_keypass", "sum"),
        )
        .reset_index()
    )

    total_xa = float(df["xa_plus"].sum()) if len(df) > 0 else 0.0
    stats["xa_plus_share"] = stats["xa_plus_sum"] / total_xa if total_xa > 0 else 0.0

    return stats.sort_values("xa_plus_sum", ascending=False)


def compute_corridor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute start zone → end zone transition matrix with xA values."""

    corridors = (
        df.groupby(["start_zone", "end_zone"], dropna=False)
        .agg(
            passes=("pass_event_id", "count"),
            xa_plus_sum=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
        )
        .reset_index()
    )

    return corridors.sort_values("xa_plus_sum", ascending=False)


def compute_corridor_pivot(df: pd.DataFrame, value_col: str = "xa_plus_sum") -> pd.DataFrame:
    """Create pivot table of start zone → end zone."""

    corridors = compute_corridor_matrix(df)

    pivot = corridors.pivot_table(
        index="start_zone",
        columns="end_zone",
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )

    return pivot


# ---------------------------------------------------------------------------
# Heatmap Plotting
# ---------------------------------------------------------------------------


def _bin_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin data to 2D grid.

    Returns (aggregated_grid, count_grid).
    """
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(values))
    if not np.any(mask):
        shape = (len(x_bins) - 1, len(y_bins) - 1)
        return np.zeros(shape), np.zeros(shape)

    xv = x[mask]
    yv = y[mask]
    vv = values[mask]

    counts, _, _ = np.histogram2d(xv, yv, bins=[x_bins, y_bins])
    sums, _, _ = np.histogram2d(xv, yv, bins=[x_bins, y_bins], weights=vv)

    if agg == "mean":
        result = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    else:
        result = sums

    return result, counts


def plot_start_location_heatmap(
    df: pd.DataFrame,
    *,
    x_bins: int = 20,
    y_bins: int = 14,
    min_count: int = 3,
) -> plt.Figure:
    """Heatmap of mean xA+ by pass start location."""

    fig, axes = plt.subplots(1, 2, figsize=(STYLE.fig_width * 1.7, STYLE.fig_height))

    for ax in axes:
        draw_pitch(ax)

    mask = df["pass_start_x"].notna() & df["pass_start_y"].notna() & df["xa_plus"].notna()
    filtered = df[mask]

    if filtered.empty:
        return fig

    xb = np.linspace(0, settings.pitch_length, x_bins)
    yb = np.linspace(0, settings.pitch_width, y_bins)

    mean_grid, count_grid = _bin_to_grid(
        filtered["pass_start_x"].to_numpy(dtype=float),
        filtered["pass_start_y"].to_numpy(dtype=float),
        filtered["xa_plus"].to_numpy(dtype=float),
        xb,
        yb,
        agg="mean",
    )

    sum_grid, _ = _bin_to_grid(
        filtered["pass_start_x"].to_numpy(dtype=float),
        filtered["pass_start_y"].to_numpy(dtype=float),
        filtered["xa_plus"].to_numpy(dtype=float),
        xb,
        yb,
        agg="sum",
    )

    # Mask low-count cells
    mean_show = mean_grid.copy()
    mean_show[count_grid < float(min_count)] = np.nan

    hm1 = plot_pitch_heatmap(axes[0], mean_show.T, cmap="YlOrRd")
    cbar1 = fig.colorbar(hm1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Mean xA+")
    axes[0].set_title("Mean xA+ by Pass Start Location")

    hm2 = plot_pitch_heatmap(axes[1], sum_grid.T, cmap="Blues")
    cbar2 = fig.colorbar(hm2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Total xA+")
    axes[1].set_title("Total xA+ by Pass Start Location")

    return fig


def plot_corridor_heatmap(corridor_pivot: pd.DataFrame, df: pd.DataFrame = None) -> plt.Figure:
    """Pitch-based visualization of pass corridors with flow arrows.
    
    Shows arrows from start zone centroids to end zone centroids,
    with arrow thickness/color representing xA values.
    """
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.4, STYLE.fig_height * 1.2))
    draw_pitch(ax)
    
    # Zone centroids (StatsBomb coordinates: 120 x 80)
    # X zones: Def (0-40), Mid (40-80), Att (80-120)
    # Y zones: Left (0-26.67), Center (26.67-53.33), Right (53.33-80)
    zone_centroids = {
        "Def Third - Left": (20, 13.3),
        "Def Third - Center": (20, 40),
        "Def Third - Right": (20, 66.7),
        "Mid Third - Left": (60, 13.3),
        "Mid Third - Center": (60, 40),
        "Mid Third - Right": (60, 66.7),
        "Att Third - Left": (100, 13.3),
        "Att Third - Center": (100, 40),
        "Att Third - Right": (100, 66.7),
    }
    
    # Draw zone boundaries
    for x in X_ZONE_EDGES[1:-1]:
        ax.axvline(x=x, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    for y in Y_ZONE_EDGES[1:-1]:
        ax.axhline(y=y, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    # Get corridor data from pivot
    max_xa = corridor_pivot.values.max() if corridor_pivot.values.max() > 0 else 1.0
    
    # Collect all corridors with xA > 0
    corridors = []
    for start_zone in corridor_pivot.index:
        for end_zone in corridor_pivot.columns:
            xa_val = corridor_pivot.loc[start_zone, end_zone]
            if xa_val > 0 and start_zone in zone_centroids and end_zone in zone_centroids:
                corridors.append({
                    "start": start_zone,
                    "end": end_zone,
                    "xa": xa_val,
                })
    
    # Sort corridors by xA (draw smaller ones first)
    corridors.sort(key=lambda x: x["xa"])
    
    # Draw arrows for each corridor
    cmap = plt.cm.YlOrRd
    
    for corridor in corridors:
        start_coords = zone_centroids[corridor["start"]]
        end_coords = zone_centroids[corridor["end"]]
        xa_val = corridor["xa"]
        
        # Normalize for color and width
        norm_xa = xa_val / max_xa
        
        # Arrow width based on xA
        width = max(0.5, norm_xa * 4)
        
        # Color based on xA
        color = cmap(norm_xa)
        
        # Calculate arrow offset to avoid overlap
        dx = end_coords[0] - start_coords[0]
        dy = end_coords[1] - start_coords[1]
        
        # Add slight curve for visibility using FancyArrowPatch
        ax.annotate(
            "",
            xy=end_coords,
            xytext=start_coords,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=width,
                mutation_scale=10 + norm_xa * 10,
                connectionstyle="arc3,rad=0.1",
                alpha=0.7,
            ),
        )
    
    # Add zone labels
    for zone, (x, y) in zone_centroids.items():
        zone_label = zone.split(" - ")[0][:3] + "\n" + zone.split(" - ")[1][:1]
        ax.text(
            x, y, zone_label,
            ha="center", va="center",
            fontsize=7, alpha=0.4,
            fontweight="bold",
        )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_xa))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Total xA+")
    
    ax.set_title("Pass Corridors: xA Flow Between Zones", fontsize=11, fontweight="bold")
    ax.set_xlabel("Direction of Play →", fontsize=9)
    
    # Add legend for arrow thickness
    ax.text(
        0.02, 0.02,
        "Arrow thickness = xA volume",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.6,
    )
    
    return fig


# ---------------------------------------------------------------------------
# Categorical Breakdowns
# ---------------------------------------------------------------------------


def compute_pass_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute xA breakdown by pass type combinations."""

    df = df.copy()

    # Create combined category
    def get_category(row):
        parts = []
        if row.get("is_cross"):
            parts.append("Cross")
        if row.get("is_through_ball"):
            parts.append("Through")
        if row.get("pass_type") and pd.notna(row["pass_type"]):
            parts.append(str(row["pass_type"]))
        if not parts:
            parts.append("Regular")
        return " | ".join(parts)

    df["pass_category"] = df.apply(get_category, axis=1)

    breakdown = (
        df.groupby("pass_category", dropna=False)
        .agg(
            passes=("pass_event_id", "count"),
            xa_plus_sum=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
        )
        .reset_index()
    )

    return breakdown.sort_values("xa_plus_sum", ascending=False)


def compute_body_part_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute xA breakdown by body part."""

    breakdown = (
        df.groupby("pass_body_part", dropna=False)
        .agg(
            passes=("pass_event_id", "count"),
            xa_plus_sum=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
        )
        .reset_index()
    )

    return breakdown.sort_values("xa_plus_sum", ascending=False)


# ---------------------------------------------------------------------------
# Main Analysis Runner
# ---------------------------------------------------------------------------


def run_spatial_analysis(
    baseline_csv: Path,
    session: Session,
) -> Dict[str, pd.DataFrame]:
    """Run full spatial analysis pipeline.

    Returns dict of DataFrames for further inspection.
    """

    configure_matplotlib()
    csv_dir = get_analysis_output_dir("cxa", subdir="csv")
    plots_dir = get_analysis_output_dir("cxa", subdir="plots")

    # Load data
    print("Loading baseline data...")
    df = pd.read_csv(baseline_csv)

    # Enrich with start locations
    print("Enriching with pass start locations...")
    df = enrich_with_pass_start(df, session)

    # Add zone labels
    print("Adding zone labels...")
    df = add_zone_labels(df)

    # 1. Zone statistics
    print("Computing zone statistics...")
    start_zone_stats = compute_zone_stats(df, "start_zone")
    start_zone_stats.to_csv(csv_dir / "cxa_spatial_start_zones.csv", index=False)

    end_zone_stats = compute_zone_stats(df, "end_zone")
    end_zone_stats.to_csv(csv_dir / "cxa_spatial_end_zones.csv", index=False)

    # X-zone only (thirds)
    start_x_stats = compute_zone_stats(df, "start_x_zone")
    start_x_stats.to_csv(csv_dir / "cxa_spatial_start_thirds.csv", index=False)

    end_x_stats = compute_zone_stats(df, "end_x_zone")
    end_x_stats.to_csv(csv_dir / "cxa_spatial_end_thirds.csv", index=False)

    # 2. Corridor analysis
    print("Computing corridor analysis...")
    corridors = compute_corridor_matrix(df)
    corridors.to_csv(csv_dir / "cxa_spatial_corridors.csv", index=False)

    corridor_pivot = compute_corridor_pivot(df)
    corridor_pivot.to_csv(csv_dir / "cxa_spatial_corridor_pivot.csv")

    # 3. Categorical breakdowns
    print("Computing categorical breakdowns...")
    pass_type_breakdown = compute_pass_type_breakdown(df)
    pass_type_breakdown.to_csv(csv_dir / "cxa_spatial_pass_categories.csv", index=False)

    body_part_breakdown = compute_body_part_breakdown(df)
    body_part_breakdown.to_csv(csv_dir / "cxa_spatial_body_part.csv", index=False)

    # 4. Plots
    print("Generating spatial plots...")

    fig = plot_start_location_heatmap(df)
    save_figure(fig, "cxa", "cxa_spatial_start_heatmap")
    plt.close(fig)

    fig = plot_corridor_heatmap(corridor_pivot)
    save_figure(fig, "cxa", "cxa_spatial_corridor_heatmap")
    plt.close(fig)

    print(f"Saved spatial analysis outputs to {csv_dir} and {plots_dir}")

    return {
        "enriched_df": df,
        "start_zone_stats": start_zone_stats,
        "end_zone_stats": end_zone_stats,
        "corridors": corridors,
        "corridor_pivot": corridor_pivot,
        "pass_type_breakdown": pass_type_breakdown,
        "body_part_breakdown": body_part_breakdown,
    }
