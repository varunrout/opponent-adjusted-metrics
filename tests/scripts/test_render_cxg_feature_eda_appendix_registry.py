"""Regression test for the chart-registry overwrite bug in the EDA appendix renderer.

`AppendixRenderer.materialize_registry` used to do
`CREATE OR REPLACE TABLE ... AS SELECT <this run only>`, which silently
destroyed every other run_id's rows on every upload -- same bug and fix as
`CxGChartRenderer._materialize_render_registry` (tests/analysis/test_cxg_charts_registry.py).
Uses a fake BigQuery client -- never touches the real `oam_analysis` tables.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "render_cxg_feature_eda_appendix.py"

_spec = importlib.util.spec_from_file_location("render_cxg_feature_eda_appendix", MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

AppendixRenderer = _module.AppendixRenderer


class _FakeQueryJob:
    def __init__(self, affected: int = 0) -> None:
        self.num_dml_affected_rows = affected

    def result(self):
        return []


class _FakeLoadJob:
    def result(self):
        return None


class FakeBigQueryClient:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.create_table_calls = 0

    def create_table(self, table, exists_ok: bool = False) -> None:
        self.create_table_calls += 1

    def query(self, sql: str, job_config=None, location=None) -> _FakeQueryJob:
        assert "DELETE FROM" in sql, f"expected a scoped DELETE, got: {sql}"
        run_id = None
        if job_config is not None:
            for param in job_config.query_parameters:
                if param.name == "run_id":
                    run_id = param.value
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.get("run_id") != run_id]
        return _FakeQueryJob(affected=before - len(self.rows))

    def load_table_from_json(self, json_rows, table_ref, job_config=None, location=None) -> _FakeLoadJob:
        self.rows.extend(json_rows)
        return _FakeLoadJob()


def _renderer_with_fake_bq(run_id: str, fake_bq: FakeBigQueryClient) -> AppendixRenderer:
    renderer = object.__new__(AppendixRenderer)
    renderer.run_id = run_id
    renderer.bq = fake_bq
    return renderer


def _rendered(names: list[str]) -> list[dict]:
    return [
        {"feature_family": "opponent_adjusted", "column_name": name, "html_uri": f"gs://bucket/{name}.html", "png_uri": f"gs://bucket/{name}.png"}
        for name in names
    ]


def test_materialize_registry_is_additive_across_run_ids():
    fake_bq = FakeBigQueryClient()

    renderer_a = _renderer_with_fake_bq("run-A", fake_bq)
    renderer_a.materialize_registry(_rendered(["feat1", "feat2"]))

    renderer_b = _renderer_with_fake_bq("run-B", fake_bq)
    renderer_b.materialize_registry(_rendered(["feat3"]))

    run_a_after = [r for r in fake_bq.rows if r["run_id"] == "run-A"]
    run_b_after = [r for r in fake_bq.rows if r["run_id"] == "run-B"]
    assert len(run_a_after) == 2, "run-A rows were destroyed by run-B's materialize (the original bug)"
    assert {r["column_name"] for r in run_a_after} == {"feat1", "feat2"}
    assert len(run_b_after) == 1
    assert len(fake_bq.rows) == 3


def test_materialize_registry_is_idempotent_for_same_run_id():
    fake_bq = FakeBigQueryClient()
    renderer = _renderer_with_fake_bq("run-A", fake_bq)

    renderer.materialize_registry(_rendered(["feat1", "feat2"]))
    renderer.materialize_registry(_rendered(["feat1", "feat2", "feat3"]))

    run_a_rows = [r for r in fake_bq.rows if r["run_id"] == "run-A"]
    assert len(run_a_rows) == 3


def test_materialize_registry_noop_on_empty_rendered_list():
    fake_bq = FakeBigQueryClient()
    renderer = _renderer_with_fake_bq("run-A", fake_bq)
    renderer.materialize_registry([])
    assert fake_bq.rows == []
    assert fake_bq.create_table_calls == 0
