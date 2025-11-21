"""Analysis utilities for CxG and opponent-adjusted metrics.

This package provides higher-level analysis functions built on top of the
core data model and prediction tables. These are intended for notebooks,
exploratory scripts, and reporting tools rather than the core API.
"""

from .cxg_analysis import (
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
