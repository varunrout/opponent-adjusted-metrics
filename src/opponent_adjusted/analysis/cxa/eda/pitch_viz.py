"""Pitch Visualization Utilities for cXA EDA.

Football pitch plotting utilities for spatial analysis:
- Pass heatmaps
- Shot maps
- Assist locations
- Action flow visualization
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def draw_pitch(ax, pitch_color="white", line_color="black", orientation="horizontal"):
    """Draw a football pitch on the given axes.

    Uses StatsBomb coordinate system: 120x80
    """
    # Pitch dimensions (StatsBomb)
    pitch_length = 120
    pitch_width = 80

    # Set background
    ax.set_facecolor(pitch_color)

    if orientation == "horizontal":
        # Pitch outline
        ax.plot([0, pitch_length], [0, 0], color=line_color, linewidth=1.5)
        ax.plot([0, pitch_length], [pitch_width, pitch_width], color=line_color, linewidth=1.5)
        ax.plot([0, 0], [0, pitch_width], color=line_color, linewidth=1.5)
        ax.plot([pitch_length, pitch_length], [0, pitch_width], color=line_color, linewidth=1.5)

        # Center line
        ax.plot(
            [pitch_length / 2, pitch_length / 2], [0, pitch_width], color=line_color, linewidth=1
        )

        # Center circle
        center_circle = plt.Circle(
            (pitch_length / 2, pitch_width / 2), 9.15, fill=False, color=line_color, linewidth=1
        )
        ax.add_patch(center_circle)

        # Left penalty area
        ax.plot([0, 18], [18, 18], color=line_color, linewidth=1)
        ax.plot([0, 18], [62, 62], color=line_color, linewidth=1)
        ax.plot([18, 18], [18, 62], color=line_color, linewidth=1)

        # Left 6-yard box
        ax.plot([0, 6], [30, 30], color=line_color, linewidth=1)
        ax.plot([0, 6], [50, 50], color=line_color, linewidth=1)
        ax.plot([6, 6], [30, 50], color=line_color, linewidth=1)

        # Right penalty area
        ax.plot([102, pitch_length], [18, 18], color=line_color, linewidth=1)
        ax.plot([102, pitch_length], [62, 62], color=line_color, linewidth=1)
        ax.plot([102, 102], [18, 62], color=line_color, linewidth=1)

        # Right 6-yard box
        ax.plot([114, pitch_length], [30, 30], color=line_color, linewidth=1)
        ax.plot([114, pitch_length], [50, 50], color=line_color, linewidth=1)
        ax.plot([114, 114], [30, 50], color=line_color, linewidth=1)

        # Goals
        ax.plot([0, 0], [36, 44], color=line_color, linewidth=3)
        ax.plot([pitch_length, pitch_length], [36, 44], color=line_color, linewidth=3)

        # Penalty spots
        ax.scatter([12, 108], [40, 40], color=line_color, s=20)

        ax.set_xlim(-2, pitch_length + 2)
        ax.set_ylim(-2, pitch_width + 2)

    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def plot_pass_map(
    df: pd.DataFrame,
    title: str = "Pass Map",
    alpha: float = 0.3,
    color: str = "steelblue",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot passes on a pitch."""

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, pitch_color="#f0f0f0")

    # Plot passes
    for _, row in df.iterrows():
        ax.annotate(
            "",
            xy=(row["end_x"], row["end_y"]),
            xytext=(row["start_x"], row["start_y"]),
            arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=0.5),
        )

    ax.set_title(title, fontsize=14)

    return fig


def plot_shot_map(
    df: pd.DataFrame, title: str = "Shot Map", figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot shots on a pitch with size proportional to xG."""

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, pitch_color="#f0f0f0")

    goals = df[df["is_goal"] == 1]
    non_goals = df[df["is_goal"] == 0]

    # Plot non-goals
    ax.scatter(
        non_goals["shot_x"],
        non_goals["shot_y"],
        s=non_goals["statsbomb_xg"] * 500 + 20,
        c="steelblue",
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
        label="No Goal",
    )

    # Plot goals
    ax.scatter(
        goals["shot_x"],
        goals["shot_y"],
        s=goals["statsbomb_xg"] * 500 + 20,
        c="coral",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        marker="*",
        label="Goal",
    )

    ax.legend(loc="upper left")
    ax.set_title(title, fontsize=14)

    return fig


def plot_heatmap_on_pitch(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    title: str = "Heatmap",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot a heatmap overlaid on a pitch."""

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, pitch_color="white", line_color="gray")

    # Create heatmap
    if weights is not None:
        hb = ax.hexbin(x, y, C=weights, gridsize=20, cmap=cmap, reduce_C_function=np.sum, alpha=0.8)
    else:
        hb = ax.hexbin(x, y, gridsize=20, cmap=cmap, alpha=0.8)

    plt.colorbar(hb, ax=ax, label="Count" if weights is None else "Total")
    ax.set_title(title, fontsize=14)

    return fig


def plot_action_flow(
    df: pd.DataFrame,
    shot_row: pd.Series,
    title: str = "Action Flow",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot a single shot's action sequence on a pitch."""

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, pitch_color="#f0f0f0")

    colors = {"Pass": "steelblue", "Carry": "coral", "Dribble": "green"}

    # Plot actions in sequence
    for i, (_, action) in enumerate(df.iterrows()):
        action_type = action.get("action_type", "Pass")
        color = colors.get(action_type, "gray")

        # Draw arrow
        ax.annotate(
            "",
            xy=(action["end_x"], action["end_y"]),
            xytext=(action["start_x"], action["start_y"]),
            arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.7),
        )

        # Number the action
        mid_x = (action["start_x"] + action["end_x"]) / 2
        mid_y = (action["start_y"] + action["end_y"]) / 2
        ax.text(
            mid_x,
            mid_y,
            str(i + 1),
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle", facecolor="white", edgecolor=color),
        )

    # Plot shot
    ax.scatter(
        shot_row["shot_x"], shot_row["shot_y"], s=200, c="red", marker="X", zorder=5, label="Shot"
    )

    # Legend
    for action_type, color in colors.items():
        ax.plot([], [], color=color, linewidth=2, label=action_type)
    ax.legend(loc="upper left")

    ax.set_title(title, fontsize=14)

    return fig


def plot_zones_analysis(
    df: pd.DataFrame,
    value_col: str,
    title: str = "Zone Analysis",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot pitch divided into zones with aggregated values."""

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, pitch_color="white", line_color="gray")

    # Define zones (6x4 grid)
    x_zones = np.linspace(0, 120, 7)
    y_zones = np.linspace(0, 80, 5)

    # Calculate zone values
    df = df.copy()
    df["x_zone"] = pd.cut(df["end_x"], bins=x_zones, labels=range(6))
    df["y_zone"] = pd.cut(df["end_y"], bins=y_zones, labels=range(4))

    zone_values = (
        df.groupby(["x_zone", "y_zone"], observed=True)[value_col].sum().unstack(fill_value=0)
    )

    # Plot zones
    cmap = plt.cm.YlOrRd
    max_val = zone_values.values.max()

    for i in range(6):
        for j in range(4):
            val = zone_values.iloc[i, j] if i < len(zone_values) and j < zone_values.shape[1] else 0
            color = cmap(val / max_val if max_val > 0 else 0)

            rect = patches.Rectangle(
                (x_zones[i], y_zones[j]),
                x_zones[i + 1] - x_zones[i],
                y_zones[j + 1] - y_zones[j],
                facecolor=color,
                alpha=0.6,
                edgecolor="gray",
            )
            ax.add_patch(rect)

            # Add value text
            ax.text(
                (x_zones[i] + x_zones[i + 1]) / 2,
                (y_zones[j] + y_zones[j + 1]) / 2,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_title(title, fontsize=14)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_val))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=value_col)

    return fig
