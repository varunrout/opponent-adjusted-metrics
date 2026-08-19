"""Tests for StatsBomb 360 empirical coverage auditing."""

from __future__ import annotations

from opponent_adjusted.ingestion.three_sixty_audit import build_three_sixty_coverage_report


def test_three_sixty_audit_intersection_and_totals():
    matches_by_competition = {
        (43, 106): [
            {"match_id": 1, "match_status_360": "available"},
            {"match_id": 2, "match_status_360": "unavailable"},
        ],
        (55, 43): [
            {"match_id": 3, "match_available_360": "2024-01-01T00:00:00.000"},
        ],
        (55, 282): [],
    }

    def load_events(match_id: int):
        payloads = {
            1: [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            2: [{"id": "d"}, {"id": "e"}],
            3: [{"id": "x"}, {"id": "y"}],
        }
        return payloads[match_id]

    def load_three_sixty(match_id: int):
        payloads = {
            1: [{"event_uuid": "a"}, {"event_uuid": "z"}],
            3: [{"event_uuid": "x"}],
        }
        return payloads[match_id]

    report = build_three_sixty_coverage_report(
        source_ref="b0bc9f22dd77c206ddedc1d742893b3bbe64baec",
        matches_by_competition=matches_by_competition,
        load_events=load_events,
        load_three_sixty=load_three_sixty,
        generated_at="2026-08-19T00:00:00+00:00",
    )

    wc22 = next(item for item in report["competitions"] if item["competition_id"] == 43)
    assert wc22["matches_total"] == 2
    assert wc22["matches_360_available"] == 1
    assert wc22["files_present"] == 1
    assert wc22["events_total"] == 5
    assert wc22["events_with_360"] == 1

    euro20 = next(
        item
        for item in report["competitions"]
        if item["competition_id"] == 55 and item["season_id"] == 43
    )
    assert euro20["matches_total"] == 1
    assert euro20["matches_360_available"] == 1
    assert euro20["files_present"] == 1
    assert euro20["events_total"] == 2
    assert euro20["events_with_360"] == 1

    first_match = next(item for item in report["matches"] if item["match_id"] == 1)
    assert first_match["event_count"] == 3
    assert first_match["frame_event_count"] == 2
    assert first_match["matched_event_count"] == 1
    assert first_match["events_without_360_count"] == 2
    assert first_match["360_without_event_count"] == 1


def test_three_sixty_audit_tracks_missing_and_corrupt_files():
    matches_by_competition = {
        (43, 106): [{"match_id": 11, "match_status_360": "available"}],
        (55, 43): [{"match_id": 12, "match_status_360": "available"}],
        (55, 282): [],
    }

    def load_events(match_id: int):
        return [{"id": f"{match_id}-event"}]

    def load_three_sixty(match_id: int):
        if match_id == 11:
            raise FileNotFoundError("missing")
        return {"not": "a-list"}

    report = build_three_sixty_coverage_report(
        source_ref="b0bc9f22dd77c206ddedc1d742893b3bbe64baec",
        matches_by_competition=matches_by_competition,
        load_events=load_events,
        load_three_sixty=load_three_sixty,
    )

    assert report["missing_expected_360_match_ids"] == [11]
    assert report["corrupt_invalid_360_match_ids"] == [12]
    assert report["overall"]["missing_expected_files"] == 1
    assert report["overall"]["corrupt_invalid_files"] == 1
