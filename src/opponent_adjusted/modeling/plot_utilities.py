"""Backward-compatible re-export — all plot utilities now live in utils.plot_utilities."""

from opponent_adjusted.utils.plot_utilities import (  # noqa: F401
    DEFAULT_PALETTE,
    configure_modeling_style,
    load_metrics_json,
    metrics_map_to_frame,
    format_metric_name,
    annotate_bars,
)

__all__ = [
    "DEFAULT_PALETTE",
    "configure_modeling_style",
    "load_metrics_json",
    "metrics_map_to_frame",
    "format_metric_name",
    "annotate_bars",
]
