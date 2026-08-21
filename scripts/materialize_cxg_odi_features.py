"""Materialize `oam_analysis.cxg_odi_features_v1` (ODI Phase 1, step 4).

One row per shot in the 360-eligible CxG+ cohort (`has_360_frame` in
`oam_features.cxg_shot_base_features`, 3,960 shots), joinable to
`cxg_plus_360_model_matrix_v1` / `cxg_analysis_360_v1` on `event_id`.

For each cohort shot, computes:
  - nearest-defender ODI (from `cxg_defensive_involvement_v1`'s own
    nearest-defender identity for that shot)
  - mean back-line ODI (defenders in the shot's own freeze-frame whose
    `position_name` matches a "Back" marker -- see
    `odi.contracts.BACKLINE_POSITION_MARKERS`)
  - GK ODI (freeze-frame defender with `position_name == "Goalkeeper"`)

ODI eligibility (cold-start gating) uses `odi.roster.MatchRoster` to resolve
each defender's continuous on-pitch tenure in the shot's period, and
`odi.aggregator.compute_odi` for the trailing-15-minute rolling sum. Every
row is populated (3,960 total, reconciling exactly to the 360-eligible
cohort) even when every feature is null -- nulls always carry an explicit
`*_null_reason` rather than being silently blank.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from google.cloud import bigquery

from opponent_adjusted.analysis.odi.aggregator import InvolvementRow, compute_odi
from opponent_adjusted.analysis.odi.contracts import BACKLINE_POSITION_MARKERS
from opponent_adjusted.analysis.odi.roster import MatchRoster, StarterRecord, SubstitutionRecord

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
FEATURE_VERSION = "cxg_odi_features_v1"

TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_odi_features_v1"
INVOLVEMENT_TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_defensive_involvement_v1"

_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.(\d+))?$")


def _event_clock_s(minute: int | None, second: int | None, timestamp: str | None) -> float | None:
    if not isinstance(minute, int) or not isinstance(second, int):
        return None
    if minute < 0 or second < 0:
        return None
    fraction = 0.0
    if timestamp is not None:
        match = _TIMESTAMP_RE.fullmatch(timestamp)
        if match is None:
            return None
        if match.group(1):
            fraction = float(f"0.{match.group(1)}")
    return minute * 60.0 + second + fraction


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params(extra: list | None = None) -> list:
    base = [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]
    return base + (extra or [])


COHORT_QUERY = f"""
SELECT b.event_id, b.match_id, b.team_id AS shooting_team_id, e.period, e.minute, e.second, e.timestamp
FROM `{PROJECT}.{FEATURE_DATASET}.cxg_shot_base_features` b
JOIN `{PROJECT}.{CORE_DATASET}.events` e
  ON b.event_id = e.event_id AND e.data_version = @data_version AND e.silver_schema_version = @schema_version
WHERE b.has_360_frame AND b.data_version = @data_version AND b.silver_schema_version = @schema_version
"""

MATCH_TEAMS_QUERY = f"""
SELECT match_id, home_team_id, away_team_id
FROM `{PROJECT}.{CORE_DATASET}.matches`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND match_id IN UNNEST(@match_ids)
"""

COHORT_DEFENDERS_QUERY = f"""
SELECT shot_event_id, player_id, position_name
FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND teammate = FALSE AND player_id IS NOT NULL
  AND shot_event_id IN UNNEST(@shot_ids)
"""

STARTERS_QUERY = f"""
SELECT match_id, team_id, player_id, position_name
FROM `{PROJECT}.{CORE_DATASET}.starting_xi_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND match_id IN UNNEST(@match_ids)
"""

SUBS_QUERY = f"""
SELECT match_id, team_id, period, minute, second, timestamp, player_id AS player_off_id, replacement_id AS player_on_id
FROM `{PROJECT}.{CORE_DATASET}.substitutions`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND match_id IN UNNEST(@match_ids)
"""

PERIOD_STARTS_QUERY = f"""
SELECT match_id, period, MIN(minute * 60 + second) AS period_start_s
FROM `{PROJECT}.{CORE_DATASET}.events`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND match_id IN UNNEST(@match_ids) AND minute IS NOT NULL AND second IS NOT NULL AND period IS NOT NULL
GROUP BY match_id, period
"""

SUB_POSITION_MODE_QUERY = f"""
SELECT match_id, player_id, position_name, COUNT(*) AS n
FROM `{PROJECT}.{CORE_DATASET}.events`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND match_id IN UNNEST(@match_ids) AND player_id IS NOT NULL AND position_name IS NOT NULL
GROUP BY match_id, player_id, position_name
"""

FREEZE_PRESENT_QUERY = f"""
SELECT DISTINCT shot_event_id
FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND shot_event_id IN UNNEST(@shot_ids)
"""

INVOLVEMENT_ALL_QUERY = f"""
SELECT player_id, match_id, shot_event_id, period, shot_timestamp_seconds, statsbomb_xg, is_goal
FROM `{INVOLVEMENT_TABLE}`
WHERE match_id IN UNNEST(@match_ids)
"""


def _rows(client, query, job_config):
    return list(client.query(query, job_config=job_config, location=LOCATION).result())


def _is_backline(position_name: str | None) -> bool:
    return position_name is not None and any(m in position_name for m in BACKLINE_POSITION_MARKERS)


def main() -> None:
    client = _client()

    print("fetching 360-eligible cohort...")
    cohort_rows = _rows(client, COHORT_QUERY, bigquery.QueryJobConfig(query_parameters=_params()))
    print(f"cohort size: {len(cohort_rows)}")
    match_ids = sorted({r.match_id for r in cohort_rows})
    shot_ids = [r.event_id for r in cohort_rows]

    match_ids_param = bigquery.ArrayQueryParameter("match_ids", "INT64", match_ids)
    shot_ids_param = bigquery.ArrayQueryParameter("shot_ids", "STRING", shot_ids)

    print("fetching match teams / starters / subs / period starts / freeze presence / involvement...")
    match_teams = {
        r.match_id: (r.home_team_id, r.away_team_id)
        for r in _rows(client, MATCH_TEAMS_QUERY, bigquery.QueryJobConfig(query_parameters=_params([match_ids_param])))
    }
    starter_rows = _rows(client, STARTERS_QUERY, bigquery.QueryJobConfig(query_parameters=_params([match_ids_param])))
    sub_rows = _rows(client, SUBS_QUERY, bigquery.QueryJobConfig(query_parameters=_params([match_ids_param])))
    period_start_rows = _rows(
        client, PERIOD_STARTS_QUERY, bigquery.QueryJobConfig(query_parameters=_params([match_ids_param]))
    )
    sub_position_rows = _rows(
        client, SUB_POSITION_MODE_QUERY, bigquery.QueryJobConfig(query_parameters=_params([match_ids_param]))
    )
    freeze_present = {
        r.shot_event_id
        for r in _rows(client, FREEZE_PRESENT_QUERY, bigquery.QueryJobConfig(query_parameters=_params([shot_ids_param])))
    }
    cohort_defender_rows = _rows(
        client, COHORT_DEFENDERS_QUERY, bigquery.QueryJobConfig(query_parameters=_params([shot_ids_param]))
    )
    involvement_rows_raw = _rows(
        client, INVOLVEMENT_ALL_QUERY, bigquery.QueryJobConfig(query_parameters=[match_ids_param])
    )

    # --- assemble per-match roster inputs ---
    starters_by_match: dict[int, list[StarterRecord]] = defaultdict(list)
    for r in starter_rows:
        starters_by_match[r.match_id].append(StarterRecord(r.team_id, r.player_id, r.position_name))

    subs_by_match: dict[int, list[SubstitutionRecord]] = defaultdict(list)
    for r in sub_rows:
        clock_s = _event_clock_s(r.minute, r.second, r.timestamp)
        if r.period is None or clock_s is None:
            continue
        subs_by_match[r.match_id].append(
            SubstitutionRecord(r.team_id, r.period, clock_s, r.player_off_id, r.player_on_id)
        )

    period_starts_by_match: dict[int, dict[int, float]] = defaultdict(dict)
    for r in period_start_rows:
        period_starts_by_match[r.match_id][r.period] = float(r.period_start_s)

    sub_position_counts: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in sub_position_rows:
        sub_position_counts[(r.match_id, r.player_id)][r.position_name] += r.n
    sub_position_mode: dict[int, dict[int, str | None]] = defaultdict(dict)
    for (match_id, player_id), counts in sub_position_counts.items():
        sub_position_mode[match_id][player_id] = max(counts.items(), key=lambda kv: kv[1])[0]

    rosters: dict[int, MatchRoster] = {}
    for match_id in match_ids:
        rosters[match_id] = MatchRoster(
            starters=starters_by_match.get(match_id, []),
            subs=subs_by_match.get(match_id, []),
            period_start_clock_s=period_starts_by_match.get(match_id, {}),
            substitute_position_lookup=sub_position_mode.get(match_id, {}),
        )

    # --- involvement rows, grouped by match for the aggregator ---
    involvement_by_match: dict[int, list[InvolvementRow]] = defaultdict(list)
    for r in involvement_rows_raw:
        if r.shot_timestamp_seconds is None:
            continue
        involvement_by_match[r.match_id].append(
            InvolvementRow(
                r.player_id, r.match_id, r.shot_event_id, r.period, r.shot_timestamp_seconds, r.statsbomb_xg, r.is_goal
            )
        )
    nearest_defender_by_shot = {}
    for r in involvement_rows_raw:
        nearest_defender_by_shot.setdefault(r.shot_event_id, r)

    defenders_by_shot: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for r in cohort_defender_rows:
        defenders_by_shot[r.shot_event_id].append((r.player_id, r.position_name))

    # --- assemble output rows ---
    rows_out: list[dict] = []
    for c in cohort_rows:
        event_id = c.event_id
        match_id = c.match_id
        period = c.period
        shot_clock = _event_clock_s(c.minute, c.second, c.timestamp)
        has_ff = event_id in freeze_present
        home_id, away_id = match_teams.get(match_id, (None, None))
        defending_team_id = (
            away_id if c.shooting_team_id == home_id else home_id
        ) if home_id is not None and away_id is not None else None
        roster = rosters[match_id]
        involvement = involvement_by_match.get(match_id, [])

        row = {
            "event_id": event_id,
            "match_id": match_id,
            "team_id": c.shooting_team_id,
            "period": period,
            "shot_timestamp_seconds": shot_clock,
            "has_freeze_frame": has_ff,
        }

        def _odi_for(player_id: int) -> tuple[float | None, bool, int, str | None]:
            if period is None or shot_clock is None or defending_team_id is None:
                return None, False, 0, "shot_timing_or_team_unresolved"
            tenure = roster.period_tenure_s(player_id, defending_team_id, period, shot_clock)
            result = compute_odi(player_id, match_id, period, shot_clock, event_id, involvement, tenure)
            return result.odi, result.eligible, result.involvement_count, result.null_reason

        # Period 5 (penalty shootout) is excluded from `cxg_defensive_involvement_v1`
        # entirely (matches the repository's existing shootout-exclusion convention;
        # see `goal_beneficiary_team_id` in event_context.py). A handful of the
        # 360-eligible cohort's shots are shootout penalties (has_360_frame does not
        # itself exclude period 5), so those rows are explicitly null-reasoned here
        # rather than silently falling into "no_resolvable_defender" / "cold_start".
        if period == 5:
            row.update(
                nearest_defender_player_id=None,
                nearest_defender_odi=None,
                nearest_defender_odi_eligible=False,
                nearest_defender_odi_involvement_count=0,
                nearest_defender_odi_null_reason="shootout_stage_excluded",
                backline_defender_count=0,
                backline_odi_eligible_count=0,
                mean_backline_odi=None,
                mean_backline_odi_null_reason="shootout_stage_excluded",
                gk_player_id=None,
                gk_odi=None,
                gk_odi_eligible=False,
                gk_odi_null_reason="shootout_stage_excluded",
                data_version=DATA_VERSION,
                silver_schema_version=SCHEMA_VERSION,
                feature_version=FEATURE_VERSION,
                materialized_at=datetime.now(UTC).isoformat(),
            )
            rows_out.append(row)
            continue

        # nearest defender
        if not has_ff:
            row.update(
                nearest_defender_player_id=None,
                nearest_defender_odi=None,
                nearest_defender_odi_eligible=False,
                nearest_defender_odi_involvement_count=0,
                nearest_defender_odi_null_reason="no_freeze_frame",
            )
        else:
            nd = nearest_defender_by_shot.get(event_id)
            if nd is None:
                row.update(
                    nearest_defender_player_id=None,
                    nearest_defender_odi=None,
                    nearest_defender_odi_eligible=False,
                    nearest_defender_odi_involvement_count=0,
                    nearest_defender_odi_null_reason="no_resolvable_defender",
                )
            else:
                odi, eligible, count, reason = _odi_for(nd.player_id)
                row.update(
                    nearest_defender_player_id=nd.player_id,
                    nearest_defender_odi=odi,
                    nearest_defender_odi_eligible=eligible,
                    nearest_defender_odi_involvement_count=count,
                    nearest_defender_odi_null_reason=reason,
                )

        # back-line
        shot_defenders = defenders_by_shot.get(event_id, [])
        backline = [(pid, pos) for pid, pos in shot_defenders if _is_backline(pos)]
        if not backline:
            row.update(
                backline_defender_count=0,
                backline_odi_eligible_count=0,
                mean_backline_odi=None,
                mean_backline_odi_null_reason="no_backline_defenders_in_freeze_frame" if has_ff else "no_freeze_frame",
            )
        else:
            values = []
            eligible_n = 0
            for pid, _pos in backline:
                odi, eligible, _count, _reason = _odi_for(pid)
                if eligible and odi is not None:
                    values.append(odi)
                    eligible_n += 1
            row.update(
                backline_defender_count=len(backline),
                backline_odi_eligible_count=eligible_n,
                mean_backline_odi=statistics.mean(values) if values else None,
                mean_backline_odi_null_reason=None if values else "no_eligible_backline_defenders",
            )

        # GK
        gks = [(pid, pos) for pid, pos in shot_defenders if pos == "Goalkeeper"]
        if not gks:
            row.update(
                gk_player_id=None,
                gk_odi=None,
                gk_odi_eligible=False,
                gk_odi_null_reason="no_gk_in_freeze_frame" if has_ff else "no_freeze_frame",
            )
        else:
            gk_pid = gks[0][0]
            odi, eligible, _count, reason = _odi_for(gk_pid)
            row.update(
                gk_player_id=gk_pid,
                gk_odi=odi,
                gk_odi_eligible=eligible,
                gk_odi_null_reason=reason,
            )

        row.update(
            data_version=DATA_VERSION,
            silver_schema_version=SCHEMA_VERSION,
            feature_version=FEATURE_VERSION,
            materialized_at=datetime.now(UTC).isoformat(),
        )
        rows_out.append(row)

    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("match_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("team_id", "INT64"),
        bigquery.SchemaField("period", "INT64"),
        bigquery.SchemaField("shot_timestamp_seconds", "FLOAT64"),
        bigquery.SchemaField("has_freeze_frame", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("nearest_defender_player_id", "INT64"),
        bigquery.SchemaField("nearest_defender_odi", "FLOAT64"),
        bigquery.SchemaField("nearest_defender_odi_eligible", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("nearest_defender_odi_involvement_count", "INT64"),
        bigquery.SchemaField("nearest_defender_odi_null_reason", "STRING"),
        bigquery.SchemaField("backline_defender_count", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("backline_odi_eligible_count", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("mean_backline_odi", "FLOAT64"),
        bigquery.SchemaField("mean_backline_odi_null_reason", "STRING"),
        bigquery.SchemaField("gk_player_id", "INT64"),
        bigquery.SchemaField("gk_odi", "FLOAT64"),
        bigquery.SchemaField("gk_odi_eligible", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("gk_odi_null_reason", "STRING"),
        bigquery.SchemaField("data_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("silver_schema_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("feature_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    job_config_load = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_json(rows_out, TABLE, job_config=job_config_load, location=LOCATION)
    job.result()
    print(json.dumps({"table": TABLE, "rows": len(rows_out), "feature_version": FEATURE_VERSION}))


if __name__ == "__main__":
    main()
