from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

import pytest

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
        self.load_calls.append((uris, table_ref, location, job_config))
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


def test_load_job_enables_parquet_list_inference_for_repeated_columns(monkeypatch):
    """Regression test for the ADAPTER_BUG: without list inference, BigQuery's Parquet loader
    silently produces empty arrays for REPEATED columns (found on visible_area/related_event_ids)
    even with an explicit REPEATED schema supplied."""
    manifest = {"tables": {"three_sixty_frames": {"row_count": 3}}}

    fake_bq = _FakeBigQueryClient(counts={"three_sixty_frames": 0})
    fake_st = _FakeStorageClient(manifest)

    monkeypatch.setattr(publish_core, "CONTRACTS", {"three_sixty_frames": object()})
    monkeypatch.setattr(publish_core, "table_bq_schema", lambda _t: [])
    monkeypatch.setattr(publish_core.bigquery, "Client", lambda project: fake_bq)
    monkeypatch.setattr(publish_core.storage, "Client", lambda project: fake_st)

    publish_core.publish_oam_core(
        publish_core.PublishConfig(
            project_id="oam-varun-260819",
            dataset="oam_core",
            location="europe-west2",
            bucket_name="oam-varun-260819-data",
            output_prefix="staged/statsbomb/x/statsbomb_silver_v1_2",
            data_version="x",
            silver_schema_version="statsbomb_silver_v1_2",
        )
    )

    assert len(fake_bq.load_calls) == 1
    job_config = fake_bq.load_calls[0][3]
    assert job_config.parquet_options is not None
    assert job_config.parquet_options.enable_list_inference is True


def _config() -> publish_core.PublishConfig:
    return publish_core.PublishConfig(
        project_id="project",
        dataset="oam_core",
        location="europe-west2",
        bucket_name="bucket",
        output_prefix="staged/version",
        data_version="data-version",
        silver_schema_version="statsbomb_silver_v1_2",
    )


def test_join_checks_scope_all_participants_and_use_parameters():
    checks = publish_core._join_checks("project.oam_core")

    assert set(checks) == {
        "shots_join_events_matches",
        "three_sixty_frames_join_events_matches",
        "three_sixty_players_join_frames",
        "possessions_join_events",
        "passes_join_events",
        "starting_xi_players_join_events_matches",
    }
    for query in checks.values():
        assert "@data_version" in query
        assert "@silver_schema_version" in query
        assert "data-version" not in query
        assert "statsbomb_silver_v1_2" not in query

    assert "s.data_version = e.data_version" in checks["shots_join_events_matches"]
    assert (
        "e.silver_schema_version = m.silver_schema_version" in checks["shots_join_events_matches"]
    )
    assert "p.data_version = f.data_version" in checks["three_sixty_players_join_frames"]
    assert "p.silver_schema_version = e.silver_schema_version" in checks["possessions_join_events"]
    assert "p.data_version = e.data_version" in checks["passes_join_events"]
    assert (
        "x.silver_schema_version = e.silver_schema_version"
        in checks["starting_xi_players_join_events_matches"]
    )
    assert "e.event_type_name = 'Starting XI'" in checks["starting_xi_players_join_events_matches"]


def test_preflight_fails_immutable_mismatch_before_loads(monkeypatch):
    monkeypatch.setattr(publish_core, "CONTRACTS", {"events": object(), "shots": object()})
    monkeypatch.setattr(
        publish_core,
        "_count_rows_for_version",
        lambda _bq, _ref, _data, _schema: 2,
    )
    uri_calls = []
    monkeypatch.setattr(
        publish_core,
        "_table_uris",
        lambda *_args: uri_calls.append(True) or ["gs://bucket/events.parquet"],
    )

    with pytest.raises(RuntimeError, match="immutable mismatch for events"):
        publish_core._build_publication_plan(
            bq=object(),
            st=object(),
            config=_config(),
            dataset_ref="project.oam_core",
            manifest={"tables": {"events": {"row_count": 3}, "shots": {"row_count": 3}}},
        )

    assert uri_calls == []


def test_preflight_fails_missing_uris_before_loads(monkeypatch):
    monkeypatch.setattr(publish_core, "CONTRACTS", {"events": object()})
    monkeypatch.setattr(
        publish_core,
        "_count_rows_for_version",
        lambda _bq, _ref, _data, _schema: 0,
    )
    monkeypatch.setattr(publish_core, "_table_uris", lambda *_args: [])

    with pytest.raises(RuntimeError, match="No parquet objects found for table events"):
        publish_core._build_publication_plan(
            bq=object(),
            st=object(),
            config=_config(),
            dataset_ref="project.oam_core",
            manifest={"tables": {"events": {"row_count": 3}}},
        )


def test_preflight_is_resumable_for_partially_completed_publication(monkeypatch):
    monkeypatch.setattr(publish_core, "CONTRACTS", {"events": object(), "shots": object()})
    counts = iter([3, 0])
    monkeypatch.setattr(
        publish_core,
        "_count_rows_for_version",
        lambda _bq, _ref, _data, _schema: next(counts),
    )
    monkeypatch.setattr(publish_core, "_table_uris", lambda *_args: ["gs://bucket/shots.parquet"])

    plan = publish_core._build_publication_plan(
        bq=object(),
        st=object(),
        config=_config(),
        dataset_ref="project.oam_core",
        manifest={"tables": {"events": {"row_count": 3}, "shots": {"row_count": 3}}},
    )

    assert [(entry.table_name, entry.action) for entry in plan] == [
        ("events", "skipped_existing"),
        ("shots", "loaded"),
    ]


def test_publish_final_completeness_recounts_every_contract_table(monkeypatch):
    manifest = {"tables": {"events": {"row_count": 3}, "shots": {"row_count": 2}}}
    fake_bq = _FakeBigQueryClient(counts={})
    fake_st = _FakeStorageClient(manifest)
    plan = [
        publish_core.PublicationPlanEntry("events", 3, 3, "skipped_existing", []),
        publish_core.PublicationPlanEntry("shots", 2, 2, "skipped_existing", []),
    ]
    recounted = []

    def count_rows(_bq, table_ref, _data, _schema):
        recounted.append(table_ref.rsplit(".", 1)[1])
        return {"events": 3, "shots": 2}[recounted[-1]]

    monkeypatch.setattr(publish_core, "CONTRACTS", {"events": object(), "shots": object()})
    monkeypatch.setattr(publish_core.bigquery, "Client", lambda project: fake_bq)
    monkeypatch.setattr(publish_core.storage, "Client", lambda project: fake_st)
    monkeypatch.setattr(publish_core, "_build_publication_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(publish_core, "_count_rows_for_version", count_rows)
    monkeypatch.setattr(publish_core, "_join_checks", lambda _dataset_ref: {})

    result = publish_core.publish_oam_core(_config())

    assert recounted == ["events", "shots"]
    assert result.table_row_counts == {"events": 3, "shots": 2}


@pytest.mark.parametrize(
    ("expected_row_count", "raises"),
    [(0, False), (3, True)],
)
def test_final_completeness_handles_missing_table_by_expected_count(
    monkeypatch, expected_row_count, raises
):
    fake_bq = _FakeBigQueryClient(counts={})
    fake_st = _FakeStorageClient({"tables": {"events": {"row_count": expected_row_count}}})
    plan = [publish_core.PublicationPlanEntry("events", expected_row_count, 0, "skipped_empty", [])]

    def missing_table(*_args):
        raise publish_core.NotFound("missing")

    monkeypatch.setattr(publish_core, "CONTRACTS", {"events": object()})
    monkeypatch.setattr(publish_core.bigquery, "Client", lambda project: fake_bq)
    monkeypatch.setattr(publish_core.storage, "Client", lambda project: fake_st)
    monkeypatch.setattr(publish_core, "_build_publication_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(publish_core, "_count_rows_for_version", missing_table)
    monkeypatch.setattr(publish_core, "_join_checks", lambda _dataset_ref: {})

    if raises:
        with pytest.raises(RuntimeError, match="Final completeness missing table for events"):
            publish_core.publish_oam_core(_config())
    else:
        result = publish_core.publish_oam_core(_config())
        assert result.table_row_counts == {"events": 0}


def test_publisher_has_no_destructive_sql_or_overwrite_disposition():
    source = inspect.getsource(publish_core)

    assert "DELETE" not in source
    assert "TRUNCATE" not in source
    assert "WRITE_TRUNCATE" not in source
