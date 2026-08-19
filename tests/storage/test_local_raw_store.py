"""Tests for local raw StatsBomb storage."""

import json

from opponent_adjusted.storage.local import LocalRawStatsBombStore


def test_local_store_preserves_statsbomb_layout_and_json_format(tmp_path):
    store = LocalRawStatsBombStore(tmp_path / "statsbomb")

    assert store.has_events(7) is False
    assert store.has_three_sixty(7) is False
    assert store.write_competitions([{"competition_id": 43}]) is True
    assert store.write_matches(43, 3, [{"match_id": 7}]) is True
    assert store.write_events(7, [{"id": "event-1"}]) is True
    assert store.write_three_sixty(7, [{"event_uuid": "event-1"}]) is True
    assert store.has_events(7) is True
    assert store.has_three_sixty(7) is True

    assert (tmp_path / "statsbomb" / "competitions.json").exists()
    assert (tmp_path / "statsbomb" / "matches" / "43" / "3.json").exists()
    assert (tmp_path / "statsbomb" / "events" / "7.json").exists()
    assert (tmp_path / "statsbomb" / "three-sixty" / "7.json").exists()
    assert json.loads((tmp_path / "statsbomb" / "events" / "7.json").read_text()) == [
        {"id": "event-1"}
    ]
    assert '  "id": "event-1"' in (tmp_path / "statsbomb" / "events" / "7.json").read_text()


def test_local_store_skips_existing_and_force_overwrites(tmp_path):
    store = LocalRawStatsBombStore(tmp_path)

    assert store.write_competitions([{"competition_id": 1}]) is True
    assert store.write_competitions([{"competition_id": 2}]) is False
    assert json.loads((tmp_path / "competitions.json").read_text())[0]["competition_id"] == 1

    assert store.write_competitions([{"competition_id": 2}], force=True) is True
    assert json.loads((tmp_path / "competitions.json").read_text())[0]["competition_id"] == 2

    assert store.write_three_sixty(77, [{"event_uuid": "a"}]) is True
    assert store.write_three_sixty(77, [{"event_uuid": "b"}]) is False
    assert store.write_three_sixty(77, [{"event_uuid": "b"}], force=True) is True
    assert json.loads((tmp_path / "three-sixty" / "77.json").read_text())[0]["event_uuid"] == "b"
