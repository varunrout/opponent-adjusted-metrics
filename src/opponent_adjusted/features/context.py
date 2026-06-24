"""Backward-compatible context feature imports."""

from opponent_adjusted.features.cxg.context import (
    calculate_game_state,
    calculate_minute_bucket_label,
    calculate_possession_features,
    calculate_pressure_features,
    calculate_pressure_proxy_score,
)

__all__ = [
    "calculate_game_state",
    "calculate_minute_bucket_label",
    "calculate_possession_features",
    "calculate_pressure_features",
    "calculate_pressure_proxy_score",
]
