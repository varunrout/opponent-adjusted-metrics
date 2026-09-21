"""Player-season CxG/CxA rollup: materializes `oam_serving.player_season_cxg_cxa_v1`,
the table that closes Hard gate 2 (`docs/dashboard_design_spec_v2.md` section 9) and
unblocks Track B's quadrant scatter. Design and decisions fully written up in
docs/analysis/quadrant_scatter_v1.md (the reviewed doc this script implements exactly,
not redesigns).

Grain: one row per (player_id, competition_id, season_id, split). Rolls up four
already-materialized sources to that grain -- no scoring, no refitting, pure SQL
aggregation:
  - `oam_ml.cxg_event_v3_predictions` / `cxg_plus_v3_predictions` (per-shot CxG,
    joined to `oam_core.shots` for player/competition/season identity)
  - `oam_serving.cxa_event_combined_v1` / `cxa_plus_combined_v1` (per-pass CxA,
    restricted to chance-creating passes -- `cxa_combined_score IS NOT NULL` --
    joined to `oam_core.events` for the PASSER's identity)

**Critical join-safety requirement, found live in `bigquery_store.py`'s own comment
(the real 3x-duplication incident already documented there): `oam_core.shots` and
`oam_core.events` each carry 3 full lineage-versioned copies of every row, one per
`silver_schema_version`.** Every join against either table in this script filters to
`silver_schema_version = SILVER_SCHEMA_VERSION` -- confirmed live (this script's own
verification step) that this filter alone is sufficient (single `data_version` value
per silver_schema_version, checked directly, not assumed).

Null vs zero discipline: `*_n_shots`/`*_n_passes_created` are always a real integer
(0 when a player-season has zero coverage for that metric). `*_mean`/`*_total`/
`*_total_xg` are NULL when their `_n` is 0 -- there is no "average" to report over an
empty set, and NULL is not the same claim as 0.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ML_DATASET = "oam_ml"
CORE_DATASET = "oam_core"
SERVING_DATASET = "oam_serving"
LOCATION = "europe-west2"

# Matches bigquery_store.py's own constant exactly -- not imported (this script
# stays decoupled from the API package, same boundary every other materialize_*
# script in this project already respects).
SILVER_SCHEMA_VERSION = "statsbomb_silver_v1_2"

MATERIALIZED_AT = datetime.now(UTC).isoformat()
SOURCE_DOCS = [
    "docs/analysis/quadrant_scatter_v1.md",
    "docs/dashboard_design_spec_v2.md",
    "docs/analysis/cxa_combined_scorer_design_v1.md",
]

TABLE = "player_season_cxg_cxa_v1"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "player_season_cxg_cxa" / "v1"

# The rollup query, parameterized only by the constants above (no per-row Python
# logic -- this is pure SQL aggregation, unlike the CxA combined scorer's ML refit).
ROLLUP_SQL = f"""
WITH cxg_event AS (
    SELECT
        s.player_id, s.competition_id, s.season_id, p.split,
        COUNT(*) AS n_shots,
        AVG(p.v3_predicted_prob) AS mean_cxg,
        SUM(p.v3_predicted_prob) AS total_cxg,
        SUM(s.statsbomb_xg) AS total_xg,
        COUNTIF(p.is_goal) AS goals
    FROM `{PROJECT}.{ML_DATASET}.cxg_event_v3_predictions` p
    JOIN `{PROJECT}.{CORE_DATASET}.shots` s
      ON s.event_id = p.event_id AND s.silver_schema_version = '{SILVER_SCHEMA_VERSION}'
    WHERE s.player_id IS NOT NULL
    GROUP BY s.player_id, s.competition_id, s.season_id, p.split
),
cxg_plus AS (
    SELECT
        s.player_id, s.competition_id, s.season_id, p.split,
        COUNT(*) AS n_shots,
        AVG(p.v3_predicted_prob) AS mean_cxg,
        SUM(p.v3_predicted_prob) AS total_cxg,
        SUM(s.statsbomb_xg) AS total_xg,
        COUNTIF(p.is_goal) AS goals
    FROM `{PROJECT}.{ML_DATASET}.cxg_plus_v3_predictions` p
    JOIN `{PROJECT}.{CORE_DATASET}.shots` s
      ON s.event_id = p.event_id AND s.silver_schema_version = '{SILVER_SCHEMA_VERSION}'
    WHERE s.player_id IS NOT NULL
    GROUP BY s.player_id, s.competition_id, s.season_id, p.split
),
cxa_event AS (
    SELECT
        e.player_id, e.competition_id, e.season_id, c.split,
        COUNT(*) AS n_passes_created,
        AVG(c.cxa_combined_score) AS mean_cxa,
        SUM(c.cxa_combined_score) AS total_cxa
    FROM `{PROJECT}.{SERVING_DATASET}.cxa_event_combined_v1` c
    JOIN `{PROJECT}.{CORE_DATASET}.events` e
      ON e.event_id = c.pass_event_id AND e.silver_schema_version = '{SILVER_SCHEMA_VERSION}'
    WHERE c.cxa_combined_score IS NOT NULL AND e.player_id IS NOT NULL
    GROUP BY e.player_id, e.competition_id, e.season_id, c.split
),
cxa_plus AS (
    SELECT
        e.player_id, e.competition_id, e.season_id, c.split,
        COUNT(*) AS n_passes_created,
        AVG(c.cxa_combined_score) AS mean_cxa,
        SUM(c.cxa_combined_score) AS total_cxa
    FROM `{PROJECT}.{SERVING_DATASET}.cxa_plus_combined_v1` c
    JOIN `{PROJECT}.{CORE_DATASET}.events` e
      ON e.event_id = c.pass_event_id AND e.silver_schema_version = '{SILVER_SCHEMA_VERSION}'
    WHERE c.cxa_combined_score IS NOT NULL AND e.player_id IS NOT NULL
    GROUP BY e.player_id, e.competition_id, e.season_id, c.split
),
player_names AS (
    SELECT player_id, ANY_VALUE(player_name) AS player_name
    FROM `{PROJECT}.{CORE_DATASET}.events`
    WHERE player_id IS NOT NULL AND silver_schema_version = '{SILVER_SCHEMA_VERSION}'
    GROUP BY player_id
),
team_by_season AS (
    SELECT player_id, competition_id, season_id,
           ANY_VALUE(team_id) AS team_id, ANY_VALUE(team_name) AS team_name
    FROM `{PROJECT}.{CORE_DATASET}.events`
    WHERE player_id IS NOT NULL AND silver_schema_version = '{SILVER_SCHEMA_VERSION}'
    GROUP BY player_id, competition_id, season_id
),
keys AS (
    SELECT player_id, competition_id, season_id, split FROM cxg_event
    UNION DISTINCT
    SELECT player_id, competition_id, season_id, split FROM cxg_plus
    UNION DISTINCT
    SELECT player_id, competition_id, season_id, split FROM cxa_event
    UNION DISTINCT
    SELECT player_id, competition_id, season_id, split FROM cxa_plus
)
SELECT
    k.player_id,
    pn.player_name,
    tb.team_id,
    tb.team_name,
    k.competition_id,
    k.season_id,
    k.split,

    COALESCE(cge.n_shots, 0) AS cxg_event_n_shots,
    cge.mean_cxg AS cxg_event_mean,
    cge.total_cxg AS cxg_event_total,
    cge.total_xg AS cxg_event_total_xg,
    COALESCE(cge.goals, 0) AS cxg_event_goals,

    COALESCE(cgp.n_shots, 0) AS cxg_plus_n_shots,
    cgp.mean_cxg AS cxg_plus_mean,
    cgp.total_cxg AS cxg_plus_total,
    cgp.total_xg AS cxg_plus_total_xg,
    COALESCE(cgp.goals, 0) AS cxg_plus_goals,

    COALESCE(cae.n_passes_created, 0) AS cxa_event_n_passes_created,
    cae.mean_cxa AS cxa_event_mean,
    cae.total_cxa AS cxa_event_total,

    COALESCE(cap.n_passes_created, 0) AS cxa_plus_n_passes_created,
    cap.mean_cxa AS cxa_plus_mean,
    cap.total_cxa AS cxa_plus_total,

    '{MATERIALIZED_AT}' AS materialized_at,
    {json.dumps(SOURCE_DOCS)} AS source_docs
FROM keys k
LEFT JOIN player_names pn USING (player_id)
LEFT JOIN team_by_season tb USING (player_id, competition_id, season_id)
LEFT JOIN cxg_event cge USING (player_id, competition_id, season_id, split)
LEFT JOIN cxg_plus cgp USING (player_id, competition_id, season_id, split)
LEFT JOIN cxa_event cae USING (player_id, competition_id, season_id, split)
LEFT JOIN cxa_plus cap USING (player_id, competition_id, season_id, split)
"""


def verify_source_multiplicity(client: bigquery.Client) -> None:
    """The exact live check this script's own docstring claims was done: confirm
    `silver_schema_version = SILVER_SCHEMA_VERSION` alone is sufficient to avoid
    the 3x-duplication bug bigquery_store.py's comment documents, for BOTH
    `shots` and `events` -- i.e. a single `data_version` value exists per
    silver_schema_version in each. Refuses to proceed if this ever stops holding."""
    for table in ("shots", "events"):
        query = f"""
            SELECT COUNT(DISTINCT data_version) AS n_versions
            FROM `{PROJECT}.{CORE_DATASET}.{table}`
            WHERE silver_schema_version = '{SILVER_SCHEMA_VERSION}'
        """
        n_versions = list(client.query(query, location=LOCATION).result())[0]["n_versions"]
        print(f"  [{table}] distinct data_version count within {SILVER_SCHEMA_VERSION}: {n_versions}")
        if n_versions != 1:
            raise RuntimeError(
                f"{table}: expected exactly 1 data_version within {SILVER_SCHEMA_VERSION}, "
                f"found {n_versions} -- the silver_schema_version filter alone is no longer "
                f"sufficient to prevent row duplication; refusing to materialize."
            )


def verify_rollup(client: bigquery.Client) -> dict:
    """Run the rollup as a plain SELECT (no write) and check its row-level
    arithmetic against the four source tables' own known population sizes, BEFORE
    writing anything -- same discipline as materialize_cxa_combined_v1.py's own
    pre-write verification."""
    check_query = f"""
        WITH player_rollup AS ({ROLLUP_SQL})
        SELECT
            COUNTIF(split = 'test') AS n_rows_test,
            SUM(IF(split = 'test', cxg_event_n_shots, 0)) AS sum_cxg_event_n_shots_test,
            SUM(IF(split = 'test', cxg_plus_n_shots, 0)) AS sum_cxg_plus_n_shots_test,
            SUM(IF(split = 'test', cxa_event_n_passes_created, 0)) AS sum_cxa_event_n_test,
            SUM(IF(split = 'test', cxa_plus_n_passes_created, 0)) AS sum_cxa_plus_n_test,
        FROM player_rollup
    """
    check = dict(list(client.query(check_query, location=LOCATION).result())[0].items())

    source_counts_query = f"""
        SELECT
            (SELECT COUNTIF(split = 'test') FROM `{PROJECT}.{ML_DATASET}.cxg_event_v3_predictions`) AS cxg_event_test_n,
            (SELECT COUNTIF(split = 'test') FROM `{PROJECT}.{ML_DATASET}.cxg_plus_v3_predictions`) AS cxg_plus_test_n,
            (SELECT COUNTIF(split = 'test' AND cxa_combined_score IS NOT NULL)
               FROM `{PROJECT}.{SERVING_DATASET}.cxa_event_combined_v1`) AS cxa_event_test_n,
            (SELECT COUNTIF(split = 'test' AND cxa_combined_score IS NOT NULL)
               FROM `{PROJECT}.{SERVING_DATASET}.cxa_plus_combined_v1`) AS cxa_plus_test_n
    """
    source = dict(list(client.query(source_counts_query, location=LOCATION).result())[0].items())

    print(f"  rollup test-split rows (player-seasons): {check['n_rows_test']}")
    print(f"  cxg_event: rollup sum={check['sum_cxg_event_n_shots_test']} vs source n={source['cxg_event_test_n']}")
    print(f"  cxg_plus:  rollup sum={check['sum_cxg_plus_n_shots_test']} vs source n={source['cxg_plus_test_n']}")
    print(f"  cxa_event: rollup sum={check['sum_cxa_event_n_test']} vs source n={source['cxa_event_test_n']}")
    print(f"  cxa_plus:  rollup sum={check['sum_cxa_plus_n_test']} vs source n={source['cxa_plus_test_n']}")

    problems = []
    # Each rollup sum should be <= the source count (a player_id IS NULL row on
    # the source side is legitimately dropped, never gained) and should equal it
    # whenever the source has no null-player rows -- checked as an inequality
    # (never MORE than the source) which catches a join-explosion regardless.
    for label, rollup_sum, source_n in (
        ("cxg_event", check["sum_cxg_event_n_shots_test"], source["cxg_event_test_n"]),
        ("cxg_plus", check["sum_cxg_plus_n_shots_test"], source["cxg_plus_test_n"]),
        ("cxa_event", check["sum_cxa_event_n_test"], source["cxa_event_test_n"]),
        ("cxa_plus", check["sum_cxa_plus_n_test"], source["cxa_plus_test_n"]),
    ):
        if rollup_sum > source_n:
            problems.append(f"{label}: rollup sum ({rollup_sum}) exceeds source count ({source_n}) -- join exploded")
    if problems:
        raise RuntimeError("rollup verification FAILED, refusing to write: " + "; ".join(problems))
    print("  rollup verification PASSED -- no join exploded any source count, writing table.")
    return {"check": check, "source": source}


def write_table(client: bigquery.Client) -> None:
    ddl = f"CREATE OR REPLACE TABLE `{PROJECT}.{SERVING_DATASET}.{TABLE}` AS\n{ROLLUP_SQL}"
    job = client.query(ddl, location=LOCATION)
    job.result()
    print(f"wrote {job.num_dml_affected_rows if job.num_dml_affected_rows is not None else '(CTAS)'} -> "
          f"{PROJECT}.{SERVING_DATASET}.{TABLE}")


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== verifying source join safety (silver_schema_version multiplicity) ===")
    verify_source_multiplicity(client)

    print("=== verifying rollup arithmetic before writing ===")
    verification = verify_rollup(client)

    print("=== writing table ===")
    write_table(client)

    row_count_query = f"SELECT COUNT(*) AS n FROM `{PROJECT}.{SERVING_DATASET}.{TABLE}`"
    n_total = list(client.query(row_count_query, location=LOCATION).result())[0]["n"]
    print(f"  final table row count (all splits): {n_total}")

    summary = {
        "materialized_at": MATERIALIZED_AT,
        "table": f"{PROJECT}.{SERVING_DATASET}.{TABLE}",
        "n_total_rows": n_total,
        "verification": verification,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
