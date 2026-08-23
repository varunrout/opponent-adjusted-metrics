"""Register v3 model chart rows in `cxg_chart_registry_v1` for a fresh run, BOTH tracks.
Same copy-forward pattern as every prior chart-registration script this session: existing
rows for the latest run_id copied forward under a NEW run_id, then these new rows appended.
Never CREATE OR REPLACE.

Reuses the existing `baseline_calibration`/`baseline_divergence_bar` chart TYPES (further
generalized in `cxg_charts.py` to detect a `<track>_v3` feature_family and read from the v3
tables, and to no longer hardcode "CxG+" in the divergence-bar title) -- per the task's
instruction to reuse the existing chart types, not invent new ones for v3. CxG event-wide
gets its own calibration/divergence charts for the first time here.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

NEW_CHARTS = [
    ("cxg_event_v3", "cxg_event_v3_calibration", "baseline_calibration", "cxg_event_v3_predictions"),
    ("cxg_event_v3", "cxg_event_v3_divergence_overall", "baseline_divergence_bar", "cxg_event_v3_divergence"),
    ("cxg_plus_v3", "cxg_plus_v3_calibration", "baseline_calibration", "cxg_plus_v3_predictions"),
    ("cxg_plus_v3", "cxg_plus_v3_divergence_cluster", "baseline_divergence_bar", "cxg_plus_v3_divergence"),
    ("cxg_plus_v3", "cxg_plus_v3_divergence_archetype", "baseline_divergence_bar", "cxg_plus_v3_divergence"),
]


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
    for feature_family, chart_name, chart_type, backing_table in NEW_CHARTS:
        selects.append(
            "SELECT "
            f"'{new_run_id}' AS run_id, '{feature_family}' AS feature_family, '{chart_name}' AS chart_name, "
            f"'{chart_type}' AS chart_type, 'plotly' AS chart_library, "
            f"'{PROJECT}.oam_ml.{backing_table}' AS backing_table, NULL AS artifact_uri, TIMESTAMP('{now}') AS materialized_at"
        )

    client.query(
        f"DELETE FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", new_run_id)]),
        location=LOCATION,
    ).result()
    client.query(f"INSERT INTO {q('cxg_chart_registry_v1')}\n" + "\nUNION ALL\n".join(selects), location=LOCATION).result()

    print(json.dumps({
        "new_run_id": new_run_id, "copied_forward_from": latest_run_id,
        "copied_forward": len(existing_rows), "new_v3_charts": len(NEW_CHARTS),
        "total_for_new_run_id": len(existing_rows) + len(NEW_CHARTS),
    }, indent=2))


if __name__ == "__main__":
    main()
