"""Phase 0 spike: shot_freeze_frame_players identity data-shape investigation.

Read-only BigQuery audit for the ODI (On-pitch Defensive Index) feature spike.
Answers, with real counts (not estimates):

1. Non-null population rate of player_id in shot_freeze_frame_players, overall
   and broken out by competition/season, and by 360-cohort membership.
2. For teammate = False rows (defenders), fraction with non-null player_id
   AND non-null position_id.
3. Cross-check: do player_ids appearing in shot_freeze_frame_players for the
   defending team match that team's starting_xi_players / substitutions
   roster for the match (sample of matches)?
4. Row-for-row comparison (same shot_event_id) between shot_freeze_frame_players
   and the generic three_sixty_players frame for the same shots: do (x, y)
   coordinates approximately correspond between the two frame sources?

No table is written or mutated. Results are dumped to JSON under
audit_outputs/cxg_analysis/odi_defprofile_spike/raw/ for the markdown report
to be authored from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
DATASET = "oam_core"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

OUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxg_analysis" / "odi_defprofile_spike" / "raw"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params() -> list[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]


def _run(client: bigquery.Client, query: str, extra_params: list | None = None) -> list[dict]:
    params = _params() + (extra_params or [])
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return [dict(r.items()) for r in client.query(query, job_config=job_config, location=LOCATION).result()]


def _dump(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf8")
    print(f"wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Q1: overall + by-competition population rate, split by 360-cohort membership
# ---------------------------------------------------------------------------
Q1_OVERALL = f"""
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT f.shot_event_id) AS distinct_shots,
  COUNTIF(f.player_id IS NOT NULL) AS player_id_nonnull_rows,
  SAFE_DIVIDE(COUNTIF(f.player_id IS NOT NULL), COUNT(*)) AS player_id_nonnull_rate
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players` f
WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
"""

Q1_BY_COMPETITION = f"""
SELECT
  co.competition_name, co.season_name, co.competition_id, co.season_id,
  COUNT(*) AS total_rows,
  COUNT(DISTINCT f.shot_event_id) AS distinct_shots,
  COUNTIF(f.player_id IS NOT NULL) AS player_id_nonnull_rows,
  SAFE_DIVIDE(COUNTIF(f.player_id IS NOT NULL), COUNT(*)) AS player_id_nonnull_rate
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players` f
JOIN `{PROJECT}.{DATASET}.competitions` co
  ON f.competition_id = co.competition_id AND f.season_id = co.season_id
  AND co.data_version = @data_version AND co.silver_schema_version = @schema_version
WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2
"""

Q1_BY_360_COHORT = f"""
SELECT
  COALESCE(b.has_360_frame, FALSE) AS has_360_frame,
  COUNT(*) AS total_rows,
  COUNT(DISTINCT f.shot_event_id) AS distinct_shots,
  COUNTIF(f.player_id IS NOT NULL) AS player_id_nonnull_rows,
  SAFE_DIVIDE(COUNTIF(f.player_id IS NOT NULL), COUNT(*)) AS player_id_nonnull_rate
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players` f
LEFT JOIN `{PROJECT}.{FEATURE_DATASET}.cxg_shot_base_features` b
  ON f.shot_event_id = b.event_id
WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
GROUP BY 1
ORDER BY 1
"""

# ---------------------------------------------------------------------------
# Q2: defender (teammate = False) rows — player_id AND position_id non-null
# ---------------------------------------------------------------------------
Q2_DEFENDERS = f"""
SELECT
  COALESCE(b.has_360_frame, FALSE) AS has_360_frame,
  COUNT(*) AS defender_rows,
  COUNT(DISTINCT f.shot_event_id) AS distinct_shots,
  COUNTIF(f.player_id IS NOT NULL) AS player_id_nonnull,
  COUNTIF(f.position_id IS NOT NULL) AS position_id_nonnull,
  COUNTIF(f.player_id IS NOT NULL AND f.position_id IS NOT NULL) AS both_nonnull,
  SAFE_DIVIDE(COUNTIF(f.player_id IS NOT NULL AND f.position_id IS NOT NULL), COUNT(*)) AS both_nonnull_rate
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players` f
LEFT JOIN `{PROJECT}.{FEATURE_DATASET}.cxg_shot_base_features` b
  ON f.shot_event_id = b.event_id
WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
  AND f.teammate = FALSE
GROUP BY 1
ORDER BY 1
"""

# ---------------------------------------------------------------------------
# Q3: roster coherence cross-check on a sample of matches
# ---------------------------------------------------------------------------
Q3_SAMPLE_MATCHES = f"""
SELECT DISTINCT match_id
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND player_id IS NOT NULL AND teammate = FALSE
ORDER BY match_id
LIMIT 15
"""

Q3_ROSTER_CHECK = f"""
WITH shot_defenders AS (
  SELECT
    f.match_id, f.shot_event_id, f.player_id AS defender_player_id,
    s.team_id AS shooting_team_id
  FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players` f
  JOIN `{PROJECT}.{DATASET}.shots` s
    ON f.shot_event_id = s.event_id
    AND s.data_version = @data_version AND s.silver_schema_version = @schema_version
  WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
    AND f.teammate = FALSE AND f.player_id IS NOT NULL
    AND f.match_id IN UNNEST(@match_ids)
),
match_teams AS (
  SELECT match_id, home_team_id, away_team_id
  FROM `{PROJECT}.{DATASET}.matches`
  WHERE data_version = @data_version AND silver_schema_version = @schema_version
    AND match_id IN UNNEST(@match_ids)
),
defending_team AS (
  SELECT
    sd.match_id, sd.shot_event_id, sd.defender_player_id,
    IF(sd.shooting_team_id = mt.home_team_id, mt.away_team_id, mt.home_team_id) AS defending_team_id
  FROM shot_defenders sd
  JOIN match_teams mt USING (match_id)
),
match_roster AS (
  SELECT match_id, team_id, player_id FROM `{PROJECT}.{DATASET}.starting_xi_players`
  WHERE data_version = @data_version AND silver_schema_version = @schema_version
    AND match_id IN UNNEST(@match_ids)
  UNION DISTINCT
  SELECT match_id, team_id, replacement_id AS player_id FROM `{PROJECT}.{DATASET}.substitutions`
  WHERE data_version = @data_version AND silver_schema_version = @schema_version
    AND match_id IN UNNEST(@match_ids) AND replacement_id IS NOT NULL
)
SELECT
  COUNT(*) AS total_defender_rows,
  COUNTIF(mr.player_id IS NOT NULL) AS matched_to_defending_team_roster,
  SAFE_DIVIDE(COUNTIF(mr.player_id IS NOT NULL), COUNT(*)) AS match_rate,
  COUNT(DISTINCT dt.match_id) AS matches_checked
FROM defending_team dt
LEFT JOIN match_roster mr
  ON dt.match_id = mr.match_id AND dt.defending_team_id = mr.team_id AND dt.defender_player_id = mr.player_id
"""

# ---------------------------------------------------------------------------
# Q4: row-for-row (x, y) comparison against three_sixty_players for same shots
# ---------------------------------------------------------------------------
Q4_SAMPLE_SHOTS = f"""
SELECT DISTINCT f.shot_event_id, f.match_id
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players` f
JOIN `{PROJECT}.{DATASET}.three_sixty_players` t
  ON f.shot_event_id = t.event_uuid AND f.match_id = t.match_id
  AND t.data_version = @data_version AND t.silver_schema_version = @schema_version
WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
  AND f.player_id IS NOT NULL
ORDER BY f.shot_event_id
LIMIT 200
"""

Q4_FREEZE_POINTS = f"""
SELECT shot_event_id, teammate, x, y
FROM `{PROJECT}.{DATASET}.shot_freeze_frame_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND shot_event_id IN UNNEST(@shot_ids)
"""

Q4_360_POINTS = f"""
SELECT event_uuid AS shot_event_id, teammate, x, y
FROM `{PROJECT}.{DATASET}.three_sixty_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND event_uuid IN UNNEST(@shot_ids)
"""


def _nearest_match_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[float]:
    """Greedy nearest-neighbour distances from each point in a to closest unused point in b."""
    remaining = list(b)
    distances = []
    for px, py in a:
        if not remaining:
            break
        best_i, best_d = None, math.inf
        for i, (qx, qy) in enumerate(remaining):
            d = math.hypot(px - qx, py - qy)
            if d < best_d:
                best_d, best_i = d, i
        distances.append(best_d)
        remaining.pop(best_i)
    return distances


def main() -> None:
    client = _client()

    print("Q1 overall...")
    _dump("q1_overall", _run(client, Q1_OVERALL))

    print("Q1 by competition...")
    _dump("q1_by_competition", _run(client, Q1_BY_COMPETITION))

    print("Q1 by 360 cohort...")
    _dump("q1_by_360_cohort", _run(client, Q1_BY_360_COHORT))

    print("Q2 defenders...")
    _dump("q2_defenders", _run(client, Q2_DEFENDERS))

    print("Q3 sample matches...")
    sample_matches = _run(client, Q3_SAMPLE_MATCHES)
    match_ids = [r["match_id"] for r in sample_matches]
    _dump("q3_sample_matches", sample_matches)

    print("Q3 roster check...")
    roster_rows = _run(
        client, Q3_ROSTER_CHECK, [bigquery.ArrayQueryParameter("match_ids", "INT64", match_ids)]
    )
    _dump("q3_roster_check", roster_rows)

    print("Q4 sample shots...")
    sample_shots = _run(client, Q4_SAMPLE_SHOTS)
    shot_ids = [r["shot_event_id"] for r in sample_shots]
    _dump("q4_sample_shots", sample_shots)

    print("Q4 freeze points / 360 points...")
    freeze_points = _run(
        client, Q4_FREEZE_POINTS, [bigquery.ArrayQueryParameter("shot_ids", "STRING", shot_ids)]
    )
    points_360 = _run(
        client, Q4_360_POINTS, [bigquery.ArrayQueryParameter("shot_ids", "STRING", shot_ids)]
    )

    by_shot_freeze: dict[str, list[tuple[float, float]]] = {}
    for r in freeze_points:
        if r["x"] is not None and r["y"] is not None:
            by_shot_freeze.setdefault(r["shot_event_id"], []).append((r["x"], r["y"]))
    by_shot_360: dict[str, list[tuple[float, float]]] = {}
    for r in points_360:
        if r["x"] is not None and r["y"] is not None:
            by_shot_360.setdefault(r["shot_event_id"], []).append((r["x"], r["y"]))

    comparison_rows = []
    all_dists: list[float] = []
    for shot_id in shot_ids:
        a = by_shot_freeze.get(shot_id, [])
        b = by_shot_360.get(shot_id, [])
        if not a or not b:
            comparison_rows.append(
                {"shot_event_id": shot_id, "freeze_count": len(a), "sixty_count": len(b), "note": "missing_side"}
            )
            continue
        dists = _nearest_match_distance(a, b)
        all_dists.extend(dists)
        comparison_rows.append(
            {
                "shot_event_id": shot_id,
                "freeze_count": len(a),
                "sixty_count": len(b),
                "mean_nn_distance": sum(dists) / len(dists) if dists else None,
                "max_nn_distance": max(dists) if dists else None,
            }
        )
    _dump("q4_comparison", comparison_rows)

    if all_dists:
        all_dists_sorted = sorted(all_dists)
        n = len(all_dists_sorted)
        summary = {
            "n_points": n,
            "mean_distance": sum(all_dists_sorted) / n,
            "median_distance": all_dists_sorted[n // 2],
            "p90_distance": all_dists_sorted[int(n * 0.9)] if n > 10 else all_dists_sorted[-1],
            "max_distance": all_dists_sorted[-1],
            "shots_compared": len([r for r in comparison_rows if "mean_nn_distance" in r]),
            "shots_missing_one_side": len([r for r in comparison_rows if r.get("note") == "missing_side"]),
        }
    else:
        summary = {"n_points": 0}
    _dump("q4_summary", [summary])

    print("Done.")


if __name__ == "__main__":
    main()
