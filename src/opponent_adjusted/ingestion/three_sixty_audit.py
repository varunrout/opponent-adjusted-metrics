"""Empirical coverage audit for separate StatsBomb 360 files."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from opponent_adjusted.ingestion.subset_fetch import (
    TARGET_THREE_SIXTY_COMPETITIONS,
    is_match_three_sixty_available,
)

JsonPayload = list | dict

TOURNAMENT_LABELS: dict[tuple[int, int], str] = {
    (43, 106): "FIFA World Cup 2022",
    (55, 43): "UEFA Euro 2020",
    (55, 282): "UEFA Euro 2024",
}


def _event_ids(events: JsonPayload) -> set[str]:
    if not isinstance(events, list):
        raise ValueError("Events payload must be a JSON list")
    ids: set[str] = set()
    for event in events:
        if isinstance(event, dict):
            value = event.get("id")
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def _frame_event_ids(three_sixty_payload: JsonPayload) -> set[str]:
    if not isinstance(three_sixty_payload, list):
        raise ValueError("Three-sixty payload must be a JSON list")
    ids: set[str] = set()
    for record in three_sixty_payload:
        if isinstance(record, dict):
            value = record.get("event_uuid")
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def _coverage_pct(matched: int, total_events: int) -> float:
    if total_events <= 0:
        return 0.0
    return round((matched / total_events) * 100.0, 6)


def build_three_sixty_coverage_report(
    *,
    source_ref: str,
    matches_by_competition: Mapping[tuple[int, int], list[dict[str, Any]]],
    load_events: Callable[[int], JsonPayload],
    load_three_sixty: Callable[[int], JsonPayload],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build tournament and overall coverage from raw events and three-sixty payloads."""
    report_matches: list[dict[str, Any]] = []
    competitions_out: list[dict[str, Any]] = []
    missing_expected_files: list[int] = []
    corrupt_invalid_files: list[int] = []

    for competition in sorted(TARGET_THREE_SIXTY_COMPETITIONS):
        competition_id, season_id = competition
        matches = matches_by_competition.get(competition, [])

        comp_events_total = 0
        comp_events_with_360 = 0
        comp_matches_total = 0
        comp_available = 0
        comp_files_present = 0
        comp_missing = 0
        comp_corrupt = 0

        for match in matches:
            match_id = int(match["match_id"])
            comp_matches_total += 1

            events_payload = load_events(match_id)
            event_ids = _event_ids(events_payload)
            event_count = len(event_ids)
            comp_events_total += event_count

            match_status_360 = str(match.get("match_status_360", ""))
            available = is_match_three_sixty_available(match)

            frame_ids: set[str] = set()
            file_present = False
            if available:
                comp_available += 1
                try:
                    three_sixty_payload = load_three_sixty(match_id)
                    frame_ids = _frame_event_ids(three_sixty_payload)
                    file_present = True
                    comp_files_present += 1
                except FileNotFoundError:
                    comp_missing += 1
                    missing_expected_files.append(match_id)
                except ValueError:
                    comp_corrupt += 1
                    corrupt_invalid_files.append(match_id)

            matched_event_count = len(event_ids & frame_ids)
            frame_event_count = len(frame_ids)
            events_without_360_count = len(event_ids - frame_ids)
            three_sixty_without_event_count = len(frame_ids - event_ids)
            comp_events_with_360 += matched_event_count

            report_matches.append(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "match_id": match_id,
                    "match_status_360": match_status_360,
                    "is_360_available": available,
                    "three_sixty_file_present": file_present,
                    "event_count": event_count,
                    "frame_event_count": frame_event_count,
                    "matched_event_count": matched_event_count,
                    "events_without_360_count": events_without_360_count,
                    "360_without_event_count": three_sixty_without_event_count,
                    "coverage_pct": _coverage_pct(matched_event_count, event_count),
                }
            )

        competitions_out.append(
            {
                "competition_id": competition_id,
                "season_id": season_id,
                "name": TOURNAMENT_LABELS.get(competition, f"{competition_id}/{season_id}"),
                "matches_total": comp_matches_total,
                "matches_360_available": comp_available,
                "files_present": comp_files_present,
                "missing_expected_files": comp_missing,
                "corrupt_invalid_files": comp_corrupt,
                "events_total": comp_events_total,
                "events_with_360": comp_events_with_360,
                "coverage_pct": _coverage_pct(comp_events_with_360, comp_events_total),
            }
        )

    overall_matches_total = sum(row["matches_total"] for row in competitions_out)
    overall_available = sum(row["matches_360_available"] for row in competitions_out)
    overall_files_present = sum(row["files_present"] for row in competitions_out)
    overall_events_total = sum(row["events_total"] for row in competitions_out)
    overall_events_with_360 = sum(row["events_with_360"] for row in competitions_out)

    return {
        "source_ref": source_ref,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "competitions": competitions_out,
        "overall": {
            "matches_total": overall_matches_total,
            "matches_360_available": overall_available,
            "files_present": overall_files_present,
            "missing_expected_files": len(missing_expected_files),
            "corrupt_invalid_files": len(corrupt_invalid_files),
            "events_total": overall_events_total,
            "events_with_360": overall_events_with_360,
            "coverage_pct": _coverage_pct(overall_events_with_360, overall_events_total),
        },
        "missing_expected_360_match_ids": sorted(set(missing_expected_files)),
        "corrupt_invalid_360_match_ids": sorted(set(corrupt_invalid_files)),
        "matches": report_matches,
    }
