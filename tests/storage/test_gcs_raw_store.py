"""Unit tests for immutable versioned GCS raw StatsBomb storage."""

from __future__ import annotations

import json

import pytest
from google.api_core.exceptions import PreconditionFailed

from opponent_adjusted.storage.gcs import GCSRawStatsBombStore


class _FakeBlob:
    def __init__(self, name: str, bucket: "_FakeBucket") -> None:
        self.name = name
        self._bucket = bucket

    def exists(self, client=None):
        return self.name in self._bucket.objects

    def upload_from_string(self, data, *, content_type: str, if_generation_match: int):
        if if_generation_match != 0:
            raise AssertionError("expected create-only precondition if_generation_match=0")
        if self.name in self._bucket.objects:
            raise PreconditionFailed("object exists")
        self._bucket.objects[self.name] = {
            "data": data,
            "content_type": content_type,
            "if_generation_match": if_generation_match,
        }


class _FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.objects: dict[str, dict] = {}

    def blob(self, name: str):
        return _FakeBlob(name, self)


class _FakeClient:
    def __init__(self) -> None:
        self._buckets: dict[str, _FakeBucket] = {}

    def bucket(self, name: str):
        if name not in self._buckets:
            self._buckets[name] = _FakeBucket(name)
        return self._buckets[name]


def test_gcs_store_writes_versioned_competitions_matches_events_keys():
    client = _FakeClient()
    store = GCSRawStatsBombStore("bucket", "a" * 40, client=client)

    assert store.write_competitions([{"competition_id": 1}]) is True
    assert store.write_matches(43, 3, [{"match_id": 1000001}]) is True
    assert store.write_events(1000001, [{"id": "event-1"}]) is True

    bucket = client.bucket("bucket")
    assert f"raw/statsbomb/{'a' * 40}/competitions.json" in bucket.objects
    assert f"raw/statsbomb/{'a' * 40}/matches/43/3.json" in bucket.objects
    assert f"raw/statsbomb/{'a' * 40}/events/1000001.json" in bucket.objects


def test_gcs_store_has_events_and_json_content_type_and_serialization():
    client = _FakeClient()
    store = GCSRawStatsBombStore("bucket", "b" * 40, client=client)

    assert store.has_events(77) is False
    assert store.write_events(77, [{"id": "e-77"}]) is True
    assert store.has_events(77) is True

    obj = client.bucket("bucket").objects[f"raw/statsbomb/{'b' * 40}/events/77.json"]
    assert obj["content_type"] == "application/json"
    assert json.loads(obj["data"].decode("utf-8")) == [{"id": "e-77"}]


def test_gcs_store_create_only_semantics_no_overwrite_when_existing():
    client = _FakeClient()
    store = GCSRawStatsBombStore("bucket", "c" * 40, client=client)

    assert store.write_events(88, [{"id": "first"}]) is True
    assert store.write_events(88, [{"id": "second"}]) is False

    obj = client.bucket("bucket").objects[f"raw/statsbomb/{'c' * 40}/events/88.json"]
    assert json.loads(obj["data"].decode("utf-8")) == [{"id": "first"}]


def test_gcs_store_force_overwrite_not_supported_for_immutable_landing():
    client = _FakeClient()
    store = GCSRawStatsBombStore("bucket", "d" * 40, client=client)

    with pytest.raises(ValueError, match="force overwrite is not supported"):
        store.write_competitions([{"competition_id": 1}], force=True)


def test_gcs_store_data_versions_map_to_distinct_prefixes():
    client = _FakeClient()
    store_a = GCSRawStatsBombStore("bucket", "e" * 40, client=client)
    store_b = GCSRawStatsBombStore("bucket", "f" * 40, client=client)

    assert store_a.write_events(123, [{"id": "a"}]) is True
    assert store_b.write_events(123, [{"id": "b"}]) is True

    bucket_objects = client.bucket("bucket").objects
    assert f"raw/statsbomb/{'e' * 40}/events/123.json" in bucket_objects
    assert f"raw/statsbomb/{'f' * 40}/events/123.json" in bucket_objects
