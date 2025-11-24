"""CxG analysis helpers and exploratory scripts package."""

from .core import (
    ShotLevelCxGRecord,
    PlayerCxGSummary,
    TeamCxGSummary,
    compute_shot_level_cxg,
    summarize_player_cxg,
    summarize_team_cxg,
)

__all__ = [
    "ShotLevelCxGRecord",
    "PlayerCxGSummary",
    "TeamCxGSummary",
    "compute_shot_level_cxg",
    "summarize_player_cxg",
    "summarize_team_cxg",
]
