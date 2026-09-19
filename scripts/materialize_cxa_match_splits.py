"""Materialize `oam_analysis.cxa_match_splits_v1` -- CxA/CxA+'s train/validation/test
match assignment.

Design decision (see docs/cxa_split_policy_and_parallel_plan.md, "Required Split
Design"): CxA's 610-match corpus is the exact same match population CxG already
split in `oam_analysis.cxg_match_splits_v1` (confirmed live: both are 610 rows/matches
under `silver_schema_version = statsbomb_silver_v1_2`). This script copies that same
`match_id -> split` assignment and `split_seed` rather than drawing an independent
random split over the same matches -- an independent split would risk a match landing
in CxG-train but CxA-test, which would block any future cross-project feature or
evaluation work without leakage. It then adds CxA's own pass-level aggregates per
match (a different grain than CxG's shot-level counts, same partition).

Must run after both training-matrix materializations
(materialize_cxa_event_v1_training_matrix.py,
materialize_cxa_plus_v1_training_matrix.py).
"""

from __future__ import annotations

import json

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"

CXG_SPLITS_TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_match_splits_v1"
EVENT_MATRIX = f"{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix"
PLUS_MATRIX = f"{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix"
TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxa_match_splits_v1"

SQL = f"""
CREATE OR REPLACE TABLE `{TABLE}` AS
WITH cxg_splits AS (
  SELECT match_id, split, split_seed, has_360_match, run_id AS source_run_id
  FROM `{CXG_SPLITS_TABLE}`
),
pass_agg AS (
  SELECT
    match_id,
    COUNT(*) AS pass_count,
    COUNTIF(is_completed) AS completed_pass_count,
    COUNTIF(y_create) AS create_count
  FROM `{EVENT_MATRIX}`
  GROUP BY match_id
),
plus_agg AS (
  SELECT
    match_id,
    COUNT(*) AS plus_pass_count,
    COUNTIF(y_create) AS plus_create_count
  FROM `{PLUS_MATRIX}`
  GROUP BY match_id
)
SELECT
  cs.match_id,
  cs.split,
  cs.split_seed,
  cs.has_360_match,
  cs.source_run_id,
  COALESCE(pa.pass_count, 0) AS pass_count,
  COALESCE(pa.completed_pass_count, 0) AS completed_pass_count,
  COALESCE(pa.create_count, 0) AS create_count,
  SAFE_DIVIDE(pa.create_count, pa.pass_count) AS create_rate,
  COALESCE(pl.plus_pass_count, 0) AS plus_pass_count,
  COALESCE(pl.plus_create_count, 0) AS plus_create_count,
  SAFE_DIVIDE(pl.plus_create_count, pl.plus_pass_count) AS plus_create_rate,
  CURRENT_TIMESTAMP() AS materialized_at
FROM cxg_splits cs
LEFT JOIN pass_agg pa ON pa.match_id = cs.match_id
LEFT JOIN plus_agg pl ON pl.match_id = cs.match_id
"""

VALIDATION_SQL = f"""
SELECT
  split,
  COUNT(*) AS matches,
  SUM(pass_count) AS pass_count,
  SUM(create_count) AS create_count,
  ROUND(SAFE_DIVIDE(SUM(create_count), SUM(pass_count)) * 100, 3) AS create_rate_pct,
  SUM(plus_pass_count) AS plus_pass_count,
  SUM(plus_create_count) AS plus_create_count,
  ROUND(SAFE_DIVIDE(SUM(plus_create_count), SUM(plus_pass_count)) * 100, 3) AS plus_create_rate_pct
FROM `{TABLE}`
GROUP BY split
ORDER BY split
"""


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    client.query(SQL, location=LOCATION).result()
    rows = [dict(r.items()) for r in client.query(VALIDATION_SQL, location=LOCATION).result()]
    print(json.dumps({"table": TABLE, "by_split": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
