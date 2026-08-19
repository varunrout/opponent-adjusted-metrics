from __future__ import annotations

import json
from types import SimpleNamespace

from opponent_adjusted.pipelines.silver import publish_core


class _FakeQueryJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class _FakeLoadJob:
    def result(self):
        return None


class _FakeBigQueryClient:
    def __init__(self, counts):
        self.counts = counts
        self.load_calls = []

    def get_dataset(self, _dataset_ref):
        return SimpleNamespace(location="europe-west2")

    def query(self, query, job_config=None, location=None):
        if "COUNT(1) AS c" in query and "WHERE data_version" in query:
            table_ref = query.split("FROM `")[1].split("`", 1)[0]
            table = table_ref.rsplit(".", 1)[1]
            c = self.counts.get(table, 0)
            return _FakeQueryJob([{"c": c}])
        return _FakeQueryJob([{"c": 1}])

    def load_table_from_uri(self, uris, table_ref, location, job_config):
        self.load_calls.append((uris, table_ref, location, job_config.write_disposition))
        table = table_ref.rsplit(".", 1)[1]
        self.counts[table] = 3
        return _FakeLoadJob()


class _FakeBlob:
    def __init__(self, payload):
        self.payload = payload

    def download_as_bytes(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeStorageClient:
    def __init__(self, manifest):
        self._manifest = manifest

    def bucket(self, _name):
        return SimpleNamespace(blob=lambda _path: _FakeBlob(self._manifest))

    def list_blobs(self, _bucket_name, prefix):
        table = prefix.split("/")[-2]
        return [SimpleNamespace(name=f"{prefix}part-00000.parquet" if table else "x")]


def test_publish_skips_existing_and_loads_missing(monkeypatch):
    manifest = {
        "tables": {
            "events": {"row_count": 3},
            "shots": {"row_count": 3},
        }
    }

    fake_bq = _FakeBigQueryClient(counts={"events": 3, "shots": 0})
    fake_st = _FakeStorageClient(manifest)

    monkeypatch.setattr(publish_core, "CONTRACTS", {"events": object(), "shots": object()})
    monkeypatch.setattr(publish_core, "table_bq_schema", lambda _t: [])
    monkeypatch.setattr(publish_core.bigquery, "Client", lambda project: fake_bq)
    monkeypatch.setattr(publish_core.storage, "Client", lambda project: fake_st)

    result = publish_core.publish_oam_core(
        publish_core.PublishConfig(
            project_id="oam-varun-260819",
            dataset="oam_core",
            location="europe-west2",
            bucket_name="oam-varun-260819-data",
            output_prefix="staged/statsbomb/x/statsbomb_silver_v1",
            data_version="x",
            silver_schema_version="statsbomb_silver_v1",
        )
    )

    assert result.load_actions["events"] == "skipped_existing"
    assert result.load_actions["shots"] == "loaded"
    assert any("shots" in call[1] for call in fake_bq.load_calls)
