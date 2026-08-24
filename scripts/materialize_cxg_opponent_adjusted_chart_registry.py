"""Register `opponent_adjusted` chart rows in `cxg_chart_registry_v1` for a fresh run.

The existing 24 registry rows (5 families x standard chart set, minus a couple of
family-specific extras) are all tagged with the frozen `cxg-analysis-20260820T201934Z`
run_id and must not be mutated or re-tagged in place (that would be rewriting history).
Since `CxGChartRenderer._load_chart_registry()` filters by run_id when one is passed,
rendering "the full set for a fresh run_id" (per task instruction, since there is no
per-family CLI filter) means: copy the 24 existing rows forward under a NEW run_id,
append the 4 new `opponent_adjusted` rows under that same new run_id, and render that
batch. The old batch is left untouched as a historical record.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

OLD_RUN_ID = "cxg-analysis-20260820T201934Z"

NEW_CHARTS = [
    ("opponent_adjusted_null_profile_bar", "bar", "cxg_null_profile_v1"),
    ("opponent_adjusted_summary_box", "box", "cxg_summary_stats_v1"),
    ("opponent_adjusted_eda_histogram", "histogram", "cxg_eda_distribution_bins_v1"),
    ("opponent_adjusted_target_lift_bar", "bar", "cxg_univariate_target_v1"),
]


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def main() -> None:
    parser_run_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not parser_run_id:
        raise SystemExit("usage: materialize_cxg_opponent_adjusted_chart_registry.py <new_run_id>")
    new_run_id = parser_run_id

    client = bigquery.Client(project=PROJECT)

    existing_rows = list(
        client.query(
            f"SELECT feature_family, chart_name, chart_type, chart_library, backing_table, artifact_uri "
            f"FROM {q('cxg_chart_registry_v1')} WHERE run_id = @old_run_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("old_run_id", "STRING", OLD_RUN_ID)]
            ),
            location=LOCATION,
        ).result()
    )
    if not existing_rows:
        raise RuntimeError(f"No existing chart_registry rows found for {OLD_RUN_ID!r} to copy forward")

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
    for chart_name, chart_type, backing_table in NEW_CHARTS:
        selects.append(
            "SELECT "
            f"'{new_run_id}' AS run_id, "
            "'opponent_adjusted' AS feature_family, "
            f"'{chart_name}' AS chart_name, "
            f"'{chart_type}' AS chart_type, "
            "'plotly' AS chart_library, "
            f"'{PROJECT}.{ANALYSIS_DATASET}.{backing_table}' AS backing_table, "
            "NULL AS artifact_uri, "
            f"TIMESTAMP('{now}') AS materialized_at"
        )

    client.query(
        f"INSERT INTO {q('cxg_chart_registry_v1')}\n" + "\nUNION ALL\n".join(selects), location=LOCATION
    ).result()

    total = list(
        client.query(
            f"SELECT COUNT(*) AS n FROM {q('cxg_chart_registry_v1')} WHERE run_id = @new_run_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("new_run_id", "STRING", new_run_id)]
            ),
            location=LOCATION,
        ).result()
    )[0].n
    print(json.dumps({"new_run_id": new_run_id, "copied_forward": len(existing_rows), "new_opponent_adjusted": len(NEW_CHARTS), "total_for_new_run_id": int(total)}, indent=2))


if __name__ == "__main__":
    main()
