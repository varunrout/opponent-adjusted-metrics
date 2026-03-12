"""Backward-compatible re-export — all plot utilities now live in utils.plot_utilities."""

from opponent_adjusted.utils.plot_utilities import (  # noqa: F401
    PlotStyle,
    STYLE,
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
    draw_pitch,
    plot_pitch_heatmap,
    compute_goal_rate_grid,
)

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
