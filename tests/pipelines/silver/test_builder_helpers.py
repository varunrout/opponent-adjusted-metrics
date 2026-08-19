from __future__ import annotations

from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

from opponent_adjusted.pipelines.silver import builder


class _FakeBlob:
    def __init__(self, name: str, bucket: "_FakeBucket"):
        self.name = name
        self.bucket = bucket
        self.md5_hash = None
        self.size = None

    def upload_from_filename(self, filename: str, if_generation_match: int):
        if if_generation_match != 0:
            raise AssertionError("expected create-only upload")
        if self.name in self.bucket.objects:
            raise PreconditionFailed("exists")
        payload = Path(filename).read_bytes()
        self.bucket.objects[self.name] = payload
        self.md5_hash = builder._md5_b64(Path(filename))
        self.size = len(payload)

    def reload(self):
        payload = self.bucket.objects[self.name]
        tmp = self.bucket.local_file_lookup[self.name]
        self.md5_hash = builder._md5_b64(tmp)
        self.size = len(payload)


class _FakeBucket:
    def __init__(self):
        self.name = "fake-bucket"
        self.objects: dict[str, bytes] = {}
        self.local_file_lookup: dict[str, Path] = {}

    def blob(self, name: str):
        return _FakeBlob(name, self)


def test_table_path_partitioned_and_non_partitioned(tmp_path):
    p1 = builder._table_path(tmp_path, "events", 43, 106)
    p2 = builder._table_path(tmp_path, "matches", None, None)
    assert "competition_id=43" in p1.as_posix()
    assert p2.as_posix().endswith("matches/part-00000.parquet")


def test_upload_create_only_skips_identical_existing(tmp_path):
    bucket = _FakeBucket()
    path = tmp_path / "file.parquet"
    path.write_bytes(b"abc")

    status1 = builder._upload_create_only(bucket, "obj", path)
    assert status1 == "uploaded"

    bucket.local_file_lookup["obj"] = path
    status2 = builder._upload_create_only(bucket, "obj", path)
    assert status2 == "skipped_identical"


def test_upload_create_only_fails_on_conflicting_existing(tmp_path):
    bucket = _FakeBucket()
    original = tmp_path / "a.parquet"
    changed = tmp_path / "b.parquet"
    original.write_bytes(b"abc")
    changed.write_bytes(b"xyz")

    assert builder._upload_create_only(bucket, "obj", original) == "uploaded"
    bucket.local_file_lookup["obj"] = original

    with pytest.raises(RuntimeError, match="Existing object differs"):
        builder._upload_create_only(bucket, "obj", changed)


def test_starting_xi_rows_preserve_lineup_facts_and_lineage():
    event = {
        "tactics": {
            "formation": 4231,
            "lineup": [
                {
                    "player": {"id": 1, "name": "Goalkeeper"},
                    "position": {"id": 1, "name": "Goalkeeper"},
                    "jersey_number": 1,
                },
                {
                    "player": {"id": 2, "name": "Defender"},
                    "position": {"id": 3, "name": "Right Center Back"},
                    "jersey_number": 5,
                },
            ],
        }
    }

    rows = builder._starting_xi_player_rows(
        event,
        event_id="starting-xi-event",
        match_id=100,
        competition_id=43,
        season_id=106,
        event_index=4,
        team_id=10,
        team_name="Example FC",
        data_version="raw-version",
        silver_schema_version="statsbomb_silver_v1_2",
    )

    assert len(rows) == 2
    assert [row["lineup_ordinal"] for row in rows] == [0, 1]
    assert rows[1] == {
        "event_id": "starting-xi-event",
        "match_id": 100,
        "competition_id": 43,
        "season_id": 106,
        "event_index": 4,
        "team_id": 10,
        "team_name": "Example FC",
        "formation": 4231,
        "lineup_ordinal": 1,
        "player_id": 2,
        "player_name": "Defender",
        "position_id": 3,
        "position_name": "Right Center Back",
        "jersey_number": 5,
        "data_version": "raw-version",
        "silver_schema_version": "statsbomb_silver_v1_2",
    }


@pytest.mark.parametrize("event", [{}, {"tactics": {}}, {"tactics": {"lineup": [{}]}}])
def test_malformed_starting_xi_does_not_fabricate_rows(event):
    with pytest.raises(ValueError):
        builder._starting_xi_player_rows(
            event,
            event_id="starting-xi-event",
            match_id=100,
            competition_id=43,
            season_id=106,
            event_index=4,
            team_id=10,
            team_name="Example FC",
            data_version="raw-version",
            silver_schema_version="statsbomb_silver_v1_2",
        )
