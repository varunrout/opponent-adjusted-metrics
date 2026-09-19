"""Regression test for the chart-registry overwrite bug.

`CxGChartRenderer._materialize_render_registry` used to do
`CREATE OR REPLACE TABLE ... AS SELECT <this run only>`, which silently
destroyed every other run_id's rows on every upload. The fix must be
additive per run_id (delete-then-insert scoped to `run_id`, never touching
other run_ids). Uses a fake BigQuery client -- never touches the real
`oam_analysis` tables, per instruction.
"""

from __future__ import annotations

from opponent_adjusted.analysis.cxg_charts import CxGChartRenderer


class _FakeQueryJob:
    def __init__(self, affected: int = 0) -> None:
        self.num_dml_affected_rows = affected

    def result(self):
        return []


class _FakeLoadJob:
    def result(self):
        return None


class FakeBigQueryClient:
    """In-memory stand-in for google.cloud.bigquery.Client, just enough surface
    for create_table / DELETE-by-run_id / load_table_from_json (append)."""

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


def _renderer_with_fake_bq(fake_bq: FakeBigQueryClient) -> CxGChartRenderer:
    renderer = object.__new__(CxGChartRenderer)
    renderer.project_id = "proj"
    renderer.analysis_dataset = "oam_analysis"
    renderer.location = "europe-west2"
    renderer.bq = fake_bq
    return renderer


def _manifest(run_id: str, chart_names: list[str]) -> dict:
    return {
        "run_id": run_id,
        "charts": [
            {"chart_name": name, "html_uri": f"gs://bucket/{run_id}/{name}.html", "png_uri": f"gs://bucket/{run_id}/{name}.png"}
            for name in chart_names
        ],
    }


def test_materialize_render_registry_is_additive_across_run_ids():
    fake_bq = FakeBigQueryClient()
    renderer = _renderer_with_fake_bq(fake_bq)

    renderer._materialize_render_registry(_manifest("run-A", ["chart1", "chart2"]))
    run_a_rows = [r for r in fake_bq.rows if r["run_id"] == "run-A"]
    assert len(run_a_rows) == 2

    renderer._materialize_render_registry(_manifest("run-B", ["chart3"]))

    # The regression this test guards against: run-A's rows must survive run-B's materialize.
    run_a_after = [r for r in fake_bq.rows if r["run_id"] == "run-A"]
    run_b_after = [r for r in fake_bq.rows if r["run_id"] == "run-B"]
    assert len(run_a_after) == 2, "run-A rows were destroyed by run-B's materialize (the original bug)"
    assert {r["chart_name"] for r in run_a_after} == {"chart1", "chart2"}
    assert len(run_b_after) == 1
    assert len(fake_bq.rows) == 3


def test_materialize_render_registry_is_idempotent_for_same_run_id():
    fake_bq = FakeBigQueryClient()
    renderer = _renderer_with_fake_bq(fake_bq)

    renderer._materialize_render_registry(_manifest("run-A", ["chart1", "chart2"]))
    renderer._materialize_render_registry(_manifest("run-A", ["chart1", "chart2", "chart3"]))

    run_a_rows = [r for r in fake_bq.rows if r["run_id"] == "run-A"]
    assert len(run_a_rows) == 3, "re-materializing the same run_id should replace, not duplicate, its own rows"


def test_materialize_render_registry_creates_table_if_missing():
    fake_bq = FakeBigQueryClient()
    renderer = _renderer_with_fake_bq(fake_bq)
    renderer._materialize_render_registry(_manifest("run-A", ["chart1"]))
    assert fake_bq.create_table_calls == 1
