"""v2 modeling spec, Step 1 (folded-in Task #11): `second_nearest_defender_style_archetype`.

A lookup/join task, NOT a new clustering fit -- reuses the exact k=4 archetype labels
already fitted and materialized by Phase B (`oam_analysis.cxg_defender_style_clusters_v1`,
`player_id -> style_archetype`). The only new work is resolving the SECOND-nearest
defender's identity per CxG+ shot (Phase B/`cxg_defensive_involvement_v1` only ever kept
rank 1) and reusing `defstyle.shot_join.resolve_shot_archetype` UNCHANGED for the lookup +
honest null-reason handling -- same function, same null-reason vocabulary, just called with
the second-nearest player_id instead of the nearest.

Second-nearest identity: same `shot_freeze_frame_players` source, same filters
(`teammate=FALSE`, non-null player_id/coordinates), same raw-Euclidean distance metric, same
ordinal tie-break as `cxg_defensive_involvement_v1`'s own `INVOLVEMENT_QUERY` -- only the
`QUALIFY ROW_NUMBER() ... = 1` changed to `= 2`. Not a new distance definition.

Gold write: additive only, mirrors Phase B's own `merge_gold_column` exactly --
`ALTER TABLE ADD COLUMN IF NOT EXISTS` + a scoped staging-table UPDATE on
`cxg_defensive_360_features`. Never CREATE OR REPLACE.
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

from opponent_adjusted.analysis.defstyle.shot_join import coverage_summary, resolve_shot_archetype

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

GOLD_TABLE = f"{PROJECT}.{FEATURE_DATASET}.cxg_defensive_360_features"
NEW_COLUMN = "second_nearest_defender_style_archetype"
STAGING_TABLE = f"{PROJECT}.{ANALYSIS_DATASET}._cxg_defstyle_second_nearest_staging"

AUDIT_DIR = ROOT / "audit_outputs" / "cxg_analysis" / "v2_model_training"

# Second-nearest defender per CxG+ 360-eligible shot -- identical filters/metric/tie-break to
# cxg_defensive_involvement_v1's INVOLVEMENT_QUERY (materialize_cxg_defensive_involvement.py),
# only the QUALIFY rank changed from 1 to 2.
SECOND_NEAREST_QUERY = f"""
WITH defenders AS (
  SELECT
    f.shot_event_id, f.player_id, f.freeze_frame_player_ordinal,
    SQRT(POW(f.x - s.location_x, 2) + POW(f.y - s.location_y, 2)) AS distance
  FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players` f
  JOIN `{PROJECT}.{CORE_DATASET}.shots` s
    ON f.shot_event_id = s.event_id
    AND s.data_version = @data_version AND s.silver_schema_version = @schema_version
  WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
    AND f.teammate = FALSE AND f.player_id IS NOT NULL
    AND f.x IS NOT NULL AND f.y IS NOT NULL
    AND s.location_x IS NOT NULL AND s.location_y IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY f.shot_event_id ORDER BY distance ASC, f.freeze_frame_player_ordinal ASC
  ) = 2
)
SELECT m.event_id, d.player_id AS second_nearest_defender_player_id
FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_plus_360_model_matrix_v1` m
LEFT JOIN defenders d ON m.event_id = d.shot_event_id
"""

ARCHETYPE_LOOKUP_QUERY = f"""
SELECT player_id, style_archetype
FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_defender_style_clusters_v1`
"""


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params() -> list:
    return [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]


def main() -> None:
    client = _client()

    print("[step1] resolving second-nearest defender identity per CxG+ shot...")
    shot_rows = [
        dict(r.items())
        for r in client.query(SECOND_NEAREST_QUERY, job_config=bigquery.QueryJobConfig(query_parameters=_params()), location=LOCATION).result()
    ]
    print(f"[step1] {len(shot_rows)} CxG+ shots")

    print("[step1] loading existing (unrefit) k=4 archetype fit: player_id -> style_archetype...")
    archetype_rows = list(client.query(ARCHETYPE_LOOKUP_QUERY, location=LOCATION).result())
    # Every player row in cxg_defender_style_clusters_v1 is present here; style_archetype is
    # None for below-threshold players (matches resolve_shot_archetype's NOT_CLUSTERED vs
    # BELOW_THRESHOLD distinction: presence in this dict with value None means "clustered
    # table has this player but they weren't assigned").
    archetype_by_player = {int(r.player_id): r.style_archetype for r in archetype_rows}
    print(f"[step1] {len(archetype_by_player)} players in the existing fit")

    resolutions = [
        resolve_shot_archetype(row["second_nearest_defender_player_id"], archetype_by_player)
        for row in shot_rows
    ]
    coverage = coverage_summary(resolutions)
    coverage["by_archetype"] = {
        a: sum(1 for r in resolutions if r.archetype == a)
        for a in sorted({r.archetype for r in resolutions if r.archetype is not None})
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "second_nearest_shot_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))

    assignments = [
        {"event_id": row["event_id"], NEW_COLUMN: resolution.archetype}
        for row, resolution in zip(shot_rows, resolutions, strict=True)
        if resolution.archetype is not None
    ]

    print(f"[step1] merging {len(assignments)} archetype assignments into Gold (additive)...")
    client.query(f"ALTER TABLE `{GOLD_TABLE}` ADD COLUMN IF NOT EXISTS `{NEW_COLUMN}` STRING", location=LOCATION).result()

    staging_schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField(NEW_COLUMN, "STRING"),
    ]
    client.load_table_from_json(
        assignments, STAGING_TABLE,
        job_config=bigquery.LoadJobConfig(schema=staging_schema, write_disposition="WRITE_TRUNCATE"),
        location=LOCATION,
    ).result()
    try:
        # Reset first so a re-run can't leave a stale value on a shot that no longer resolves.
        client.query(f"UPDATE `{GOLD_TABLE}` SET `{NEW_COLUMN}` = NULL WHERE TRUE", location=LOCATION).result()
        client.query(
            f"""
            UPDATE `{GOLD_TABLE}` t
            SET t.`{NEW_COLUMN}` = s.`{NEW_COLUMN}`
            FROM `{STAGING_TABLE}` s
            WHERE t.event_id = s.event_id
            """,
            location=LOCATION,
        ).result()
    finally:
        client.query(f"DROP TABLE IF EXISTS `{STAGING_TABLE}`", location=LOCATION).result()

    verify = list(client.query(
        f"SELECT COUNT(*) AS n, COUNTIF(`{NEW_COLUMN}` IS NOT NULL) AS n_assigned FROM `{GOLD_TABLE}`",
        location=LOCATION,
    ).result())[0]
    print(json.dumps({"gold_total_rows": int(verify.n), "gold_assigned": int(verify.n_assigned)}, indent=2))


if __name__ == "__main__":
    main()
