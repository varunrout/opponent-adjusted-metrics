"""Tests for StatsBomb subset orchestration."""

from dataclasses import dataclass, field

from opponent_adjusted.ingestion.subset_fetch import run_subset_fetch


@dataclass
class FakeSource:
    competitions: list
    matches: dict
    events: dict
    event_calls: list[int] = field(default_factory=list)
    pace_calls: int = 0

    def get_competitions(self):
        return self.competitions

    def get_matches(self, competition_id: int, season_id: int):
        return self.matches.get((competition_id, season_id))

    def get_events(self, match_id: int):
        self.event_calls.append(match_id)
        return self.events.get(match_id)

    def pace_after_event_fetch(self):
        self.pace_calls += 1


@dataclass
class FakeStore:
    writes: list[tuple] = field(default_factory=list)
    existing: set[tuple] = field(default_factory=set)

    def has_events(self, match_id: int) -> bool:
        return match_id in self.existing

    def _write(self, kind: str, key, payload, force: bool):
        self.writes.append((kind, key, payload, force))
        if key in self.existing and not force:
            return False
        return True

    def write_competitions(self, payload, *, force=False):
        return self._write("competitions", "competitions", payload, force)

    def write_matches(self, competition_id, season_id, payload, *, force=False):
        return self._write("matches", (competition_id, season_id), payload, force)

    def write_events(self, match_id, payload, *, force=False):
        return self._write("events", match_id, payload, force)


def _config():
    return {
        "competitions": [
            {"competition_id": 43, "season_id": 3, "include_events": True},
            {"competition_id": 55, "season_id": 43, "include_events": False},
        ]
    }


def test_orchestration_filters_and_persists_matches_and_events():
    source = FakeSource(
        competitions=[
            {"competition_id": 43, "season_id": 3},
            {"competition_id": 55, "season_id": 43},
            {"competition_id": 99, "season_id": 1},
        ],
        matches={
            (43, 3): [{"match_id": 7}],
            (55, 43): [{"match_id": 8}],
        },
        events={7: [{"id": "event-7"}], 8: [{"id": "event-8"}]},
    )
    store = FakeStore()

    summary = run_subset_fetch(
        _config(),
        source=source,
        store=store,
        include_events=True,
        force=False,
        config_label="config.json",
        output_label="data/statsbomb",
    )

    assert summary == {
        "config": "config.json",
        "output_dir": "data/statsbomb",
        "competitions_selected": 2,
        "competitions_written": 1,
        "matches_written": 2,
        "matches_skipped_existing": 0,
        "events_written": 1,
        "events_skipped_existing": 0,
        "missing": [],
    }
    assert source.event_calls == [7]
    assert source.pace_calls == 1
    assert [(kind, key) for kind, key, _payload, _force in store.writes] == [
        ("competitions", "competitions"),
        ("matches", (43, 3)),
        ("events", 7),
        ("matches", (55, 43)),
    ]


def test_orchestration_records_missing_matches_and_events():
    source = FakeSource(
        competitions=[{"competition_id": 43, "season_id": 3}],
        matches={(43, 3): [{"match_id": 7}, {"match_id": 8}]},
        events={7: None, 8: [{"id": "event-8"}]},
    )
    store = FakeStore()

    summary = run_subset_fetch(
        _config() | {"competitions": [{"competition_id": 43, "season_id": 3}]},
        source=source,
        store=store,
        include_events=True,
        force=False,
        config_label="config.json",
        output_label="data/statsbomb",
    )

    assert summary["missing"] == [{"scope": "events", "match_id": 7}]
    assert summary["events_written"] == 1

    missing_matches_source = FakeSource(
        competitions=[{"competition_id": 43, "season_id": 3}],
        matches={(43, 3): None},
        events={},
    )
    missing_summary = run_subset_fetch(
        {"competitions": [{"competition_id": 43, "season_id": 3}]},
        source=missing_matches_source,
        store=FakeStore(),
        include_events=True,
        force=False,
        config_label="config.json",
        output_label="data/statsbomb",
    )
    assert missing_summary["missing"] == [
        {"scope": "matches", "competition_id": 43, "season_id": 3}
    ]


def test_orchestration_counts_skips_and_force_pass_through():
    source = FakeSource(
        competitions=[{"competition_id": 43, "season_id": 3}],
        matches={(43, 3): [{"match_id": 7}]},
        events={7: [{"id": "event-7"}]},
    )
    store = FakeStore(existing={"competitions", (43, 3), 7})

    summary = run_subset_fetch(
        {"competitions": [{"competition_id": 43, "season_id": 3}]},
        source=source,
        store=store,
        include_events=True,
        force=False,
        config_label="config.json",
        output_label="data/statsbomb",
    )

    assert summary["competitions_written"] == 0
    assert summary["matches_skipped_existing"] == 1
    assert summary["events_skipped_existing"] == 1
    assert all(force is False for _kind, _key, _payload, force in store.writes)
    assert source.event_calls == []
    assert source.pace_calls == 0


def test_existing_event_skips_before_fetch_and_does_not_mark_missing():
    source = FakeSource(
        competitions=[{"competition_id": 43, "season_id": 3}],
        matches={(43, 3): [{"match_id": 7}]},
        events={7: None},
    )
    store = FakeStore(existing={7})

    summary = run_subset_fetch(
        {"competitions": [{"competition_id": 43, "season_id": 3}]},
        source=source,
        store=store,
        include_events=True,
        force=False,
        config_label="config.json",
        output_label="data/statsbomb",
    )

    assert source.event_calls == []
    assert summary["events_skipped_existing"] == 1
    assert summary["missing"] == []
    assert source.pace_calls == 0


def test_existing_event_force_true_fetches_and_overwrites():
    source = FakeSource(
        competitions=[{"competition_id": 43, "season_id": 3}],
        matches={(43, 3): [{"match_id": 7}]},
        events={7: [{"id": "event-7"}]},
    )
    store = FakeStore(existing={7})

    summary = run_subset_fetch(
        {"competitions": [{"competition_id": 43, "season_id": 3}]},
        source=source,
        store=store,
        include_events=True,
        force=True,
        config_label="config.json",
        output_label="data/statsbomb",
    )

    assert source.event_calls == [7]
    assert summary["events_written"] == 1
    assert summary["events_skipped_existing"] == 0
    event_writes = [w for w in store.writes if w[0] == "events"]
    assert len(event_writes) == 1
    assert event_writes[0][3] is True
    assert source.pace_calls == 1


def test_orchestration_global_include_events_false_skips_event_source():
    source = FakeSource(
        competitions=[{"competition_id": 43, "season_id": 3}],
        matches={(43, 3): [{"match_id": 7}]},
        events={7: [{"id": "event-7"}]},
    )

    summary = run_subset_fetch(
        {"competitions": [{"competition_id": 43, "season_id": 3}]},
        source=source,
        store=FakeStore(),
        include_events=False,
        force=False,
        config_label="config.json",
        output_label="data/statsbomb",
    )

    assert summary["matches_written"] == 1
    assert summary["events_written"] == 0
    assert source.event_calls == []
