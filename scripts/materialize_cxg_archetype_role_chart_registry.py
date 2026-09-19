"""Register the 3 new archetype/role validation chart rows in `cxg_chart_registry_v1` for a
fresh run. Same copy-forward pattern as every prior chart-registration script this session
(e.g. `materialize_cxg_baseline_chart_registry.py`): the existing rows for the latest known
run_id are copied forward under a NEW run_id, then these 3 new rows are appended. The old
batch is left untouched as a historical record -- never CREATE OR REPLACE.

The 4 backfilled EDA charts (opponent_adjusted_eda_histogram/null_profile_bar/summary_box/
target_lift_bar) and the updated bivariate_significance_grid charts are already registered
from prior runs -- re-rendering them under the new run_id picks up the backfilled tables /
code changes automatically, no new registry rows needed for those.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

from google.cloud import bigquery

NEW_CHARTS = [
    ("opponent_adjusted", "opponent_adjusted_archetype_goal_rate_bar", "archetype_goal_rate_bar", "cxg_plus_360_model_matrix_v1"),
    ("opponent_adjusted", "opponent_adjusted_archetype_role_heatmap", "archetype_role_heatmap", "cxg_defensive_360_features"),
    ("opponent_adjusted", "opponent_adjusted_role_displacement_bar", "role_displacement_bar", "cxg_defensive_360_features"),
]


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: materialize_cxg_archetype_role_chart_registry.py <new_run_id>")
    new_run_id = sys.argv[1]

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

    existing_rows = list(
        client.query(
            f"SELECT feature_family, chart_name, chart_type, chart_library, backing_table, artifact_uri "
            f"FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
            job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", latest_run_id)]),
            location=LOCATION,
        ).result()
    )
    if not existing_rows:
        raise RuntimeError(f"No existing chart_registry rows found for {latest_run_id!r} to copy forward")

    now = datetime.now(UTC).isoformat()
    selects = []
    for r in existing_rows:
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
    for feature_family, chart_name, chart_type, backing_table in NEW_CHARTS:
        dataset = "oam_features" if backing_table == "cxg_defensive_360_features" else ANALYSIS_DATASET
        selects.append(
            "SELECT "
            f"'{new_run_id}' AS run_id, "
            f"'{feature_family}' AS feature_family, "
            f"'{chart_name}' AS chart_name, "
            f"'{chart_type}' AS chart_type, "
            "'plotly' AS chart_library, "
            f"'{PROJECT}.{dataset}.{backing_table}' AS backing_table, "
            "NULL AS artifact_uri, "
            f"TIMESTAMP('{now}') AS materialized_at"
        )

    client.query(
        f"DELETE FROM {q('cxg_chart_registry_v1')} WHERE run_id = @run_id",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", new_run_id)]),
        location=LOCATION,
    ).result()
    client.query(f"INSERT INTO {q('cxg_chart_registry_v1')}\n" + "\nUNION ALL\n".join(selects), location=LOCATION).result()

    print(json.dumps({
        "new_run_id": new_run_id,
        "copied_forward_from": latest_run_id,
        "copied_forward": len(existing_rows),
        "new_archetype_role_charts": len(NEW_CHARTS),
        "total_for_new_run_id": len(existing_rows) + len(NEW_CHARTS),
    }, indent=2))


if __name__ == "__main__":
    main()
