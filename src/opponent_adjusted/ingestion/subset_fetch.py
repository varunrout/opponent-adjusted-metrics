"""Orchestration for fetching and storing a configured StatsBomb subset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from opponent_adjusted.ingestion.contracts import FetchSummary
from opponent_adjusted.storage.interfaces import RawStatsBombStore


class StatsBombSourceProtocol(Protocol):
    """Structural source contract used by the orchestration layer."""

    def get_competitions(self) -> list | dict | None: ...

    def get_matches(self, competition_id: int, season_id: int) -> list | dict | None: ...

    def get_events(self, match_id: int) -> list | dict | None: ...

    def pace_after_event_fetch(self) -> None: ...


def _filter_competitions(
    all_competitions: list[dict[str, Any]], configured: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    wanted = {(int(item["competition_id"]), int(item["season_id"])) for item in configured}
    return [
        comp
        for comp in all_competitions
        if (int(comp["competition_id"]), int(comp["season_id"])) in wanted
    ]


def run_subset_fetch(
    config: Mapping[str, Any],
    *,
    source: StatsBombSourceProtocol,
    store: RawStatsBombStore,
    include_events: bool,
    force: bool,
    config_label: str,
    output_label: str,
) -> FetchSummary:
    """Fetch configured source payloads and pass them to a raw-data store."""
    competitions_config = config.get("competitions", [])
    if not competitions_config:
        raise ValueError(f"No competitions configured in {config_label}")

    summary: FetchSummary = {
        "config": config_label,
        "output_dir": output_label,
        "competitions_selected": 0,
        "competitions_written": 0,
        "matches_written": 0,
        "matches_skipped_existing": 0,
        "events_written": 0,
        "events_skipped_existing": 0,
        "missing": [],
    }

    all_competitions = source.get_competitions()
    if not isinstance(all_competitions, list):
        raise RuntimeError("Could not fetch StatsBomb competitions.json")

    selected_competitions = _filter_competitions(all_competitions, competitions_config)
    summary["competitions_selected"] = len(selected_competitions)
    if not selected_competitions:
        raise RuntimeError("Configured competitions were not found in StatsBomb competitions.json")

    if store.write_competitions(selected_competitions, force=force):
        summary["competitions_written"] += 1

    for competition in selected_competitions:
        competition_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])
        matches = source.get_matches(competition_id, season_id)
        if not isinstance(matches, list):
            summary["missing"].append(
                {"scope": "matches", "competition_id": competition_id, "season_id": season_id}
            )
            continue

        if store.write_matches(competition_id, season_id, matches, force=force):
            summary["matches_written"] += 1
        else:
            summary["matches_skipped_existing"] += 1

        if not include_events:
            continue

        should_fetch_events = next(
            (
                bool(item.get("include_events", True))
                for item in competitions_config
                if int(item["competition_id"]) == competition_id
                and int(item["season_id"]) == season_id
            ),
            True,
        )
        if not should_fetch_events:
            continue

        for match in matches:
            match_id = int(match["match_id"])

            if not force and store.has_events(match_id):
                summary["events_skipped_existing"] += 1
                continue

            events = source.get_events(match_id)
            if not isinstance(events, list):
                summary["missing"].append({"scope": "events", "match_id": match_id})
                continue
            if store.write_events(match_id, events, force=force):
                summary["events_written"] += 1
                source.pace_after_event_fetch()
            else:
                summary["events_skipped_existing"] += 1

    return summary
