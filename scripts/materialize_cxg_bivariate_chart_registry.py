"""Register bivariate-interaction chart rows in `cxg_chart_registry_v1` for a fresh run.

Same copy-forward pattern as `materialize_cxg_opponent_adjusted_chart_registry.py`: the
existing rows for the latest known run_id are copied forward under a NEW run_id (auto-detected
by MAX(materialized_at), not hardcoded), then the bivariate rows are appended to that same new
run_id. The old batch is left untouched as a historical record -- never CREATE OR REPLACE.

New chart types (added to `cxg_charts.py`):
  - bivariate_significance_grid  -- Tier 1 exhaustive-pairwise FDR-adjusted-p heatmap, one per track.
  - bivariate_stratified_bar     -- Tier 2 / Tier 3 headline stratified goal-rate bar, one each.

Feature families are synthetic (not real feature families -- same pattern as
`cxg_event_correlation` / `cxg_plus_pca` from the correlation/PCA task): `cxg_event_bivariate`,
`cxg_plus_bivariate`. Tier is encoded in `chart_name` for the stratified-bar charts since a
track can have both a tier-2 and a tier-3 headline chart.
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
    # chart_name is the render/GCS file-naming key (see CxGChartRenderer.run) -- must be
    # unique across every row, so it's track-prefixed the same way the correlation/PCA
    # track-charts are (cxg_event_correlation_heatmap / cxg_plus_correlation_heatmap etc).
    ("cxg_event_bivariate", "cxg_event_bivariate_tier1_significance_grid", "bivariate_significance_grid", "cxg_bivariate_interaction_v1"),
    ("cxg_plus_bivariate", "cxg_plus_bivariate_tier1_significance_grid", "bivariate_significance_grid", "cxg_bivariate_interaction_v1"),
    ("cxg_plus_bivariate", "cxg_plus_bivariate_tier2_stratified_bar", "bivariate_stratified_bar", "cxg_bivariate_stratified_v1"),
    ("cxg_plus_bivariate", "cxg_plus_bivariate_tier3_stratified_bar", "bivariate_stratified_bar", "cxg_bivariate_stratified_v1"),
]


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: materialize_cxg_bivariate_chart_registry.py <new_run_id>")
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
        selects.append(
            "SELECT "
            f"'{new_run_id}' AS run_id, "
            f"'{feature_family}' AS feature_family, "
            f"'{chart_name}' AS chart_name, "
            f"'{chart_type}' AS chart_type, "
            "'plotly' AS chart_library, "
            f"'{PROJECT}.{ANALYSIS_DATASET}.{backing_table}' AS backing_table, "
            "NULL AS artifact_uri, "
            f"TIMESTAMP('{now}') AS materialized_at"
        )

    # Scoped delete-then-insert for the new run_id only (never touches any other run_id's rows).
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
        "new_bivariate_charts": len(NEW_CHARTS),
        "total_for_new_run_id": len(existing_rows) + len(NEW_CHARTS),
    }, indent=2))


if __name__ == "__main__":
    main()
