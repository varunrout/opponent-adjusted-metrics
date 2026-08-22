"""Copy-forward the chart registry to a new run_id after the Phase C requalification pass.

No NEW chart definitions are needed -- `feature_correlation_heatmap`, `pca_scree`, and
`bivariate_significance_grid` already exist for BOTH `cxg_event` and `cxg_plus` in the
latest run_id (confirmed live before writing this script), each reading directly from
`cxg_feature_correlation_v1` / `cxg_pca_components_v1` / `cxg_bivariate_interaction_v1`
filtered by track. Those underlying tables' CONTENT changed (Phase C requalification added
rows), but the chart rows' identity (table, track) did not, so a plain copy-forward under a
fresh run_id is sufficient -- same pattern as `materialize_cxg_v2_model_chart_registry.py`,
never `CREATE OR REPLACE` the registry itself.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def main() -> None:
    new_run_id = sys.argv[1] if len(sys.argv) > 1 else f"cxg-analysis-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    client = bigquery.Client(project=PROJECT)

    latest_row = list(client.query(
        f"SELECT run_id FROM {q('cxg_chart_registry_v1')} GROUP BY run_id ORDER BY MAX(materialized_at) DESC LIMIT 1",
        location=LOCATION,
    ).result())
    if not latest_row:
        raise SystemExit("No existing cxg_chart_registry_v1 rows found to copy forward from.")
    latest_run_id = latest_row[0].run_id

    existing_rows = list(client.query(
        f"SELECT feature_family, chart_name, chart_type, chart_library, backing_table, artifact_uri "
        f"FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", latest_run_id)]),
        location=LOCATION,
    ).result())
    if not existing_rows:
        raise RuntimeError(f"No existing chart_registry rows found for {latest_run_id!r} to copy forward")

    now = datetime.now(UTC).isoformat()
    selects = []
    for r in existing_rows:
        uri = f"'{r.artifact_uri}'" if r.artifact_uri else "NULL"
        selects.append(
            "SELECT "
            f"'{new_run_id}' AS run_id, '{r.feature_family}' AS feature_family, '{r.chart_name}' AS chart_name, "
            f"'{r.chart_type}' AS chart_type, '{r.chart_library}' AS chart_library, "
            f"'{r.backing_table}' AS backing_table, {uri} AS artifact_uri, TIMESTAMP('{now}') AS materialized_at"
        )

    client.query(
        f"DELETE FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", new_run_id)]),
        location=LOCATION,
    ).result()
    client.query(f"INSERT INTO {q('cxg_chart_registry_v1')}\n" + "\nUNION ALL\n".join(selects), location=LOCATION).result()

    print(json.dumps({
        "new_run_id": new_run_id, "copied_forward_from": latest_run_id,
        "copied_forward": len(existing_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
