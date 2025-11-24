"""Common plotting utilities and style configuration for analysis.

This module centralizes colors, figure sizes, fonts, and export helpers so
all analysis code produces consistent, publication-quality plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from opponent_adjusted.config import settings


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotStyle:
    """Style configuration for analysis plots."""

    # Colors
    primary_color: str = "#1f77b4"  # blue
    secondary_color: str = "#ff7f0e"  # orange
    tertiary_color: str = "#2ca02c"  # green
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"

    # Figure sizes
    fig_width: float = 8.0
    fig_height: float = 5.0

    # Fonts
    font_family: str = "DejaVu Sans"
    font_size_small: int = 9
    font_size_medium: int = 11
    font_size_large: int = 13

    # DPI for exports
    dpi: int = 120


STYLE = PlotStyle()


def configure_matplotlib() -> None:
    """Apply global Matplotlib rcParams based on STYLE.

    Call this once at the start of an analysis script/notebook.
    """

    mpl.rcParams.update(
        {
            "figure.figsize": (STYLE.fig_width, STYLE.fig_height),
            "figure.facecolor": STYLE.background_color,
            "axes.facecolor": STYLE.background_color,
            "axes.grid": True,
            "grid.color": STYLE.grid_color,
            "font.family": STYLE.font_family,
            "font.size": STYLE.font_size_medium,
            "axes.titlesize": STYLE.font_size_large,
            "axes.labelsize": STYLE.font_size_medium,
            "legend.fontsize": STYLE.font_size_small,
        }
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def get_analysis_output_dir(analysis_type: str, subdir: str = "plots") -> Path:
    """Return (and create) the standard output directory for an analysis.

    Layout convention:
        outputs/analysis/{analysis_type}/{subdir}/

    Example:
        get_analysis_output_dir("cxg", "plots")
        -> outputs/analysis/cxg/plots
    """

    root = settings.data_root.parent / "outputs" / "analysis" / analysis_type / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_figure(
    fig: mpl.figure.Figure,
    analysis_type: str,
    name: str,
    subdir: str = "plots",
    tight: bool = True,
    transparent: bool = False,
) -> Path:
    """Save a Matplotlib figure using the standard output layout.

    Args:
        fig: Matplotlib Figure instance.
        analysis_type: High-level analysis name (e.g., "cxg").
        name: Base file name without extension.
        subdir: Subdirectory under the analysis type (default: "plots").
        tight: Whether to apply tight_layout before saving.
        transparent: Save with transparent background.

    Returns:
        Path to the saved PNG file.
    """

    out_dir = get_analysis_output_dir(analysis_type, subdir=subdir)
    path = out_dir / f"{name}.png"

    if tight:
        fig.tight_layout()

    fig.savefig(path, dpi=STYLE.dpi, transparent=transparent)
    return path


# ---------------------------------------------------------------------------
# Optional pitch map helper
# ---------------------------------------------------------------------------


def draw_pitch(
    ax: mpl.axes.Axes,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
    line_color: str = "#cccccc",
) -> mpl.axes.Axes:
    """Draw a simple StatsBomb-style pitch on the given Axes.

    Coordinates default to the StatsBomb pitch (120 x 80) from settings.
    """

    if pitch_length is None:
        pitch_length = settings.pitch_length
    if pitch_width is None:
        pitch_width = settings.pitch_width

    # Pitch outline
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)

    ax.plot(
        [0, 0, pitch_length, pitch_length, 0],
        [0, pitch_width, pitch_width, 0, 0],
        color=line_color,
    )

    # Halfway line
    ax.axvline(pitch_length / 2.0, color=line_color, linewidth=0.8, linestyle="--")

    # Center circle (approximate)
    center_x = pitch_length / 2.0
    center_y = pitch_width / 2.0
    center_circle = mpl.patches.Circle(
        (center_x, center_y), 10, fill=False, color=line_color, linewidth=0.8
    )
    ax.add_patch(center_circle)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # Match common football plotting convention
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_pitch_heatmap(
    ax: mpl.axes.Axes,
    grid: Any,
    *,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
    cmap: str = "magma",
    alpha: float = 0.85,
    vmin: float | None = None,
    vmax: float | None = None,
) -> mpl.image.AxesImage:
    """Render a 2D grid onto the pitch as a heatmap."""

    import numpy as np  # Local import to avoid hard dependency when unused

    if pitch_length is None:
        pitch_length = settings.pitch_length
    if pitch_width is None:
        pitch_width = settings.pitch_width

    # Ensure grid is a numpy array for imshow
    data = np.asarray(grid)

    image = ax.imshow(
        data,
        extent=(0, pitch_length, pitch_width, 0),
        origin="upper",
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    return image


def compute_goal_rate_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
) -> np.ndarray:
    """Compute a 2D goal-rate grid for plotting on a pitch."""

    if pitch_length is None:
        pitch_length = settings.pitch_length
    if pitch_width is None:
        pitch_width = settings.pitch_width

    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(values))
    if not np.any(mask):
        return np.zeros((len(x_bins) - 1, len(y_bins) - 1))

    x_valid = x[mask]
    y_valid = y[mask]
    v_valid = values[mask]

    counts, _, _ = np.histogram2d(
        x_valid,
        y_valid,
        bins=[x_bins, y_bins],
        range=[[0, pitch_length], [0, pitch_width]],
    )
    goals, _, _ = np.histogram2d(
        x_valid,
        y_valid,
        bins=[x_bins, y_bins],
        range=[[0, pitch_length], [0, pitch_width]],
        weights=v_valid,
    )

    goal_rate = np.divide(goals, counts, out=np.zeros_like(goals), where=counts > 0)
    return goal_rate


__all__ = [
    "PlotStyle",
    "STYLE",
    "configure_matplotlib",
    "get_analysis_output_dir",
    "save_figure",
    "draw_pitch",
    "plot_pitch_heatmap",
    "compute_goal_rate_grid",
]
