"""Focused tests for three-sixty ingestion entrypoints."""

from __future__ import annotations

import json

from scripts import fetch_statsbomb_subset


def test_fetch_subset_local_with_three_sixty_only(tmp_path, monkeypatch):
    config_path = tmp_path / "subset.json"
    config_path.write_text(
        json.dumps(
            {
                "competitions": [
                    {"competition_id": 43, "season_id": 106, "include_events": True},
                    {"competition_id": 2, "season_id": 27, "include_events": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_fetch(url: str):
        if url.endswith("/competitions.json"):
            return [
                {"competition_id": 43, "season_id": 106},
                {"competition_id": 2, "season_id": 27},
            ]
        if url.endswith("/matches/43/106.json"):
            return [{"match_id": 11, "match_status_360": "available"}]
        if url.endswith("/matches/2/27.json"):
            return [{"match_id": 22, "match_status_360": "available"}]
        if url.endswith("/three-sixty/11.json"):
            return [{"event_uuid": "e-11"}]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(fetch_statsbomb_subset, "_fetch_with_retries", fake_fetch)

    summary = fetch_statsbomb_subset.fetch_subset(
        config_path,
        tmp_path / "data",
        include_events=False,
        include_three_sixty=True,
        force=False,
    )

    assert summary["three_sixty_candidates"] == 1
    assert summary["three_sixty_available_matches"] == 1
    assert summary["three_sixty_written"] == 1
    assert summary["missing"] == []
    assert (tmp_path / "data" / "three-sixty" / "11.json").exists()
    assert not (tmp_path / "data" / "three-sixty" / "22.json").exists()
