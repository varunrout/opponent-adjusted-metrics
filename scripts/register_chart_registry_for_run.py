"""Copy the latest known-good `cxg_chart_registry_v1` batch forward to a new run_id.

Generalizes the pattern used one-off in `materialize_cxg_opponent_adjusted_chart_registry.py`
(which copied forward from one hardcoded historical run_id) so `oam-analyse` can register a
fresh run_id automatically on every invocation, without needing to know the prior run_id in
advance. Auto-detects the latest run_id by `MAX(materialized_at)`.

Safety check carried over from the prior chart-coverage task: any row whose `backing_table`
no longer exists in BigQuery is dropped rather than copied forward (this is exactly the
`cxg_correlation_v1` situation found and handled manually before -- a chart type whose
backing table was retired by the split-policy revert must never be silently re-registered).

Uses the fixed delete-then-insert-per-run_id pattern already built and tested for this exact
table (`tests/analysis/test_cxg_charts_registry.py` covers the underlying logic in
`CxGChartRenderer._materialize_render_registry`; this script performs the equivalent
operation directly since it's populating the *input* registry, not the *rendered* output
registry).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def _table_exists(client: bigquery.Client, table_ref: str) -> bool:
    try:
        client.get_table(table_ref)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy latest chart_registry batch forward to a new run_id")
    parser.add_argument("new_run_id")
    args = parser.parse_args()
    new_run_id = args.new_run_id

    client = bigquery.Client(project=PROJECT)

    latest_row = list(
        client.query(
            f"SELECT run_id FROM {q('cxg_chart_registry_v1')} GROUP BY run_id ORDER BY MAX(materialized_at) DESC LIMIT 1",
            location=LOCATION,
        ).result()
    )
    if not latest_row:
        raise SystemExit("No existing cxg_chart_registry_v1 rows found to copy forward from.")
    latest_run_id = latest_row[0].run_id
    if latest_run_id == new_run_id:
        print(f"[register_chart_registry_for_run] {new_run_id} already the latest run_id; nothing to do.")
        return

    rows = list(
        client.query(
            f"SELECT feature_family, chart_name, chart_type, chart_library, backing_table, artifact_uri "
            f"FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", latest_run_id)]
            ),
            location=LOCATION,
        ).result()
    )

    kept, dropped = [], []
    for r in rows:
        if _table_exists(client, r.backing_table):
            kept.append(r)
        else:
            dropped.append(r.chart_name)

    if not kept:
        raise SystemExit(f"All {len(rows)} rows from {latest_run_id} had missing backing tables; nothing to register.")

    # Scoped delete-then-insert for the new run_id (never touches any other run_id's rows).
    client.query(
        f"DELETE FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", new_run_id)]
        ),
        location=LOCATION,
    ).result()

    now = datetime.now(UTC).isoformat()
    selects = []
    for r in kept:
        uri = f"'{r.artifact_uri}'" if r.artifact_uri else "NULL"
        selects.append(
            "SELECT "
            f"'{new_run_id}' AS run_id, "
            f"'{r.feature_family}' AS feature_family, "
            f"'{r.chart_name}' AS chart_name, "
            f"'{r.chart_type}' AS chart_type, "
            f"'{r.chart_library}' AS chart_library, "
            f"'{r.backing_table}' AS backing_table, "
            f"{uri} AS artifact_uri, "
            f"TIMESTAMP('{now}') AS materialized_at"
        )
    client.query(f"INSERT INTO {q('cxg_chart_registry_v1')}\n" + "\nUNION ALL\n".join(selects), location=LOCATION).result()

    print(
        f"[register_chart_registry_for_run] copied {len(kept)} rows from {latest_run_id} -> {new_run_id}"
        + (f"; dropped {len(dropped)} rows with missing backing tables: {dropped}" if dropped else "")
    )


if __name__ == "__main__":
    main()
