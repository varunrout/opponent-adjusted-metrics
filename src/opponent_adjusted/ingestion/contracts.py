"""Small internal contracts for StatsBomb subset ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class CompetitionConfig(TypedDict, total=False):
    competition_id: int
    season_id: int
    name: str
    season: str
    include_events: bool
    purpose: str


class SubsetConfig(TypedDict):
    competitions: list[CompetitionConfig]


class FetchSummary(TypedDict):
    config: str
    output_dir: str
    competitions_selected: int
    competitions_written: int
    matches_written: int
    matches_skipped_existing: int
    events_written: int
    events_skipped_existing: int
    missing: list[dict[str, Any]]


def load_subset_config(path: Path) -> SubsetConfig:
    """Load and minimally validate the configured subset JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Subset config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Subset config must be an object: {path}")
    competitions = config.get("competitions", [])
    if not isinstance(competitions, list):
        raise ValueError(f"Subset config competitions must be a list: {path}")
    return config  # type: ignore[return-value]
