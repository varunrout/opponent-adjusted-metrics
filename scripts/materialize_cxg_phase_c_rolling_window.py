"""Phase C rolling-window defensive features (v3, BOTH tracks -- CxG event-wide AND CxG+).

Adds new columns to the EXISTING `oam_features.cxg_event_context_features` table
(ALTER TABLE ADD COLUMN IF NOT EXISTS + scoped UPDATE...FROM a staging table, keyed on
event_id) -- the same additive precedent as Phase A's
`materialize_cxg_phase_a_geometric_features.py`, deliberately NOT a `CREATE OR REPLACE
TABLE` rebuild of that JSON-blob-derived table.

Landed in `cxg_event_context_features` (not a `*_360_features` table) because these three
features are event-log-based and apply to EVERY shot in the governed 610-match universe
(15,737 shots), not just the 360-eligible 3,960-shot CxG+ subset -- confirmed live: that
table already covers exactly the same 15,737-row, 610-match population as
`cxg_shot_base_features`. CxG+ automatically inherits these columns as a subquery of the
same table (via `has_360_frame` downstream), so this script computes each feature ONCE per
shot rather than once per track.

  - defensive_action_rate_{15,30,45,60}m + defensive_action_rate_null_reason
  - territorial_dominance_last_15m
  - cross_match_defensive_rate + cross_match_defensive_rate_null_reason

See `phase_c_rolling_window.py`'s module docstring for the construction logic, half-life
justification, and cold-start conventions -- not repeated here.

COST DISCIPLINE: a single lean events query (period/minute/second/timestamp/event_type_name
/team_id/location_x/location_y only -- no join to passes/carries/etc, unlike the full E7-E12
Gold pipeline) fetches every event for the 610 split-assigned matches ONCE, pinning
data_version/silver_schema_version together throughout (this warehouse's `events` table
holds 3 schema-version copies of the same corpus; an unpinned join triples every count).
`oam_core.events` is confirmed unpartitioned/unclustered, so a windowed-SQL rewrite would
be an equally-expensive full scan with no cost advantage -- the established "fetch once,
process per-match in Python" pattern (matching `_momentum()`/Phase B's `COUNTS_QUERY`
precedent) satisfies cost discipline without reinventing the existing rolling-window logic.
Defending team_id is resolved via a single batched join to `oam_core.matches`
(home_team_id/away_team_id), not a per-match query.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from google.cloud import bigquery

from opponent_adjusted.analysis.defstyle.features import ACTION_TYPES
from opponent_adjusted.features.cxg.event_context import EventRecord, _ordered, event_clock_s
from opponent_adjusted.features.cxg.phase_c_rolling_window import (
    NULL_REASON_DEFENDING_TEAM_UNRESOLVED,
    NULL_REASON_FIRST_MATCH,
    cross_match_rolling_rate,
    defensive_action_rates,
    period_start_clock,
    territorial_dominance_extended,
)
from materialize_cxg_feature_family_tables import _create_training_view  # noqa: E402

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"

DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
TARGET_TABLE = "cxg_event_context_features"
STAGING_TABLE = "_phase_c_rolling_window_staging"

DEFENSIVE_ACTION_TYPES = frozenset(ACTION_TYPES)

NEW_COLUMNS = [
    ("defensive_action_rate_15m", "FLOAT64"),
    ("defensive_action_rate_30m", "FLOAT64"),
    ("defensive_action_rate_45m", "FLOAT64"),
    ("defensive_action_rate_60m", "FLOAT64"),
    ("defensive_action_rate_null_reason", "STRING"),
    ("territorial_dominance_last_15m", "FLOAT64"),
    ("cross_match_defensive_rate", "FLOAT64"),
    ("cross_match_defensive_rate_null_reason", "STRING"),
    ("phase_c_materialized_at", "STRING"),
]

# Lean projection -- only the columns Phase C's rolling-window logic actually needs, unlike
# the full E7-E12 Gold pipeline's multi-table join (no passes/carries/dribbles/receipts join
# required here). Pinned data_version/silver_schema_version, scoped to the 610-match
# governed universe via cxg_match_splits_v1 (Phase B's `COUNTS_QUERY` precedent).
EVENTS_QUERY = f"""
SELECT
  e.event_id, e.match_id, e.event_index, e.period, e.minute, e.second, e.timestamp,
  e.event_type_name, e.team_id, e.location_x, e.location_y
FROM `{PROJECT}.{CORE_DATASET}.events` e
JOIN `{PROJECT}.{ANALYSIS_DATASET}.cxg_match_splits_v1` s USING (match_id)
WHERE e.data_version = @data_version AND e.silver_schema_version = @schema_version
ORDER BY e.match_id, e.period, e.event_index
"""

MATCHES_QUERY = f"""
SELECT m.match_id, m.home_team_id, m.away_team_id, m.match_date, m.kick_off
FROM `{PROJECT}.{CORE_DATASET}.matches` m
JOIN `{PROJECT}.{ANALYSIS_DATASET}.cxg_match_splits_v1` s USING (match_id)
WHERE m.data_version = @data_version AND m.silver_schema_version = @schema_version
"""


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params() -> list:
    return [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]


def _fetch_events(client: bigquery.Client) -> dict[int, list[EventRecord]]:
    job_config = bigquery.QueryJobConfig(query_parameters=_params())
    matches: dict[int, list[EventRecord]] = defaultdict(list)
    for x in client.query(EVENTS_QUERY, job_config=job_config, location=LOCATION).result(page_size=20000):
        matches[x.match_id].append(
            EventRecord(
                event_id=x.event_id,
                match_id=x.match_id,
                event_index=x.event_index,
                period=x.period,
                minute=x.minute,
                second=x.second,
                timestamp=x.timestamp,
                event_type_name=x.event_type_name,
                outcome_name=None,
                team_id=x.team_id,
                possession_id=None,
                possession_team_id=None,
                location_x=x.location_x,
                location_y=x.location_y,
            )
        )
    return matches


def _fetch_matches(client: bigquery.Client) -> dict[int, tuple[int, int, str, str]]:
    job_config = bigquery.QueryJobConfig(query_parameters=_params())
    out: dict[int, tuple[int, int, str, str]] = {}
    for r in client.query(MATCHES_QUERY, job_config=job_config, location=LOCATION).result():
        out[r.match_id] = (r.home_team_id, r.away_team_id, r.match_date, r.kick_off)
    return out


def _defending_team_id(shot: EventRecord, match_row: tuple[int, int, str, str]) -> int | None:
    home_id, away_id, _date, _ko = match_row
    if shot.team_id == home_id:
        return away_id
    if shot.team_id == away_id:
        return home_id
    return None


def _team_match_defensive_rate(match_events: list[EventRecord], team_id: int) -> float | None:
    """Per-minute rate of `team_id`'s defensive actions across the WHOLE match (both
    periods) -- the per-(team, match) building block for 2c's cross-match rolling
    computation."""
    total_actions = sum(
        1
        for event in match_events
        if event.team_id == team_id and event.event_type_name in DEFENSIVE_ACTION_TYPES
    )
    total_minutes = 0.0
    for period in {event.period for event in match_events if event.period is not None}:
        period_events = [event for event in match_events if event.period == period]
        clocks = [event_clock_s(e) for e in period_events if event_clock_s(e) is not None]
        if clocks:
            total_minutes += (max(clocks) - min(clocks)) / 60.0
    if total_minutes <= 0:
        return None
    return total_actions / total_minutes


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()

    print("[phase_c] fetching events for the 610-match governed universe...")
    events_by_match = _fetch_events(client)
    print(f"[phase_c] fetched events for {len(events_by_match)} matches")

    print("[phase_c] fetching match home/away team ids...")
    match_rows = _fetch_matches(client)

    # --- 2c stage 1: per-(team_id, match_id) whole-match defensive rate, plus chronological
    # ordering per team (global by team_id + match_date, NOT scoped to competition/season --
    # per the locked decision that competitions 43/55 share real team_ids).
    team_match_rate: dict[tuple[int, int], float | None] = {}
    for match_id, match_events in events_by_match.items():
        row = match_rows.get(match_id)
        if row is None:
            continue
        home_id, away_id, _date, _ko = row
        for team_id in (home_id, away_id):
            team_match_rate[(team_id, match_id)] = _team_match_defensive_rate(match_events, team_id)

    matches_by_team: dict[int, list[int]] = defaultdict(list)
    for team_id, match_id in team_match_rate:
        matches_by_team[team_id].append(match_id)
    for team_id in matches_by_team:
        matches_by_team[team_id].sort(key=lambda mid: (match_rows[mid][2], match_rows[mid][3]))

    print(f"[phase_c] built whole-match defensive rate for {len(team_match_rate)} team-match pairs "
          f"across {len(matches_by_team)} teams")

    staging_rows: list[dict[str, object]] = []
    n_shots = 0
    n_first_match = 0
    for match_id, match_events in events_by_match.items():
        row = match_rows.get(match_id)
        ordered = _ordered(match_events)
        for index, shot in enumerate(ordered):
            if shot.event_type_name != "Shot":
                continue
            n_shots += 1
            prior_events = ordered[:index]
            shot_clock = event_clock_s(shot)
            defending_team_id = _defending_team_id(shot, row) if row is not None else None

            rate_cols = defensive_action_rates(
                prior_events, defending_team_id, DEFENSIVE_ACTION_TYPES, shot.period, shot_clock
            )
            field_tilt = territorial_dominance_extended(prior_events, shot.team_id, shot.period, shot_clock)

            cross_match_value, cross_match_reason = None, None
            if defending_team_id is not None and row is not None:
                team_matches = matches_by_team.get(defending_team_id, [])
                current_pos = team_matches.index(match_id) if match_id in team_matches else -1
                prior_match_ids = team_matches[:current_pos] if current_pos >= 0 else []
                prior_rates = [
                    team_match_rate[(defending_team_id, mid)]
                    for mid in reversed(prior_match_ids)
                    if team_match_rate.get((defending_team_id, mid)) is not None
                ]
                cross_match_value, cross_match_reason = cross_match_rolling_rate(prior_rates)
            else:
                cross_match_reason = NULL_REASON_DEFENDING_TEAM_UNRESOLVED
            if cross_match_reason == NULL_REASON_FIRST_MATCH:
                n_first_match += 1

            staging_rows.append(
                {
                    "event_id": shot.event_id,
                    **rate_cols,
                    "territorial_dominance_last_15m": field_tilt,
                    "cross_match_defensive_rate": cross_match_value,
                    "cross_match_defensive_rate_null_reason": cross_match_reason,
                    "phase_c_materialized_at": now,
                }
            )

    print(f"[phase_c] computed features for {n_shots} shots "
          f"(cross-match cold-start first matches: {n_first_match})")

    staging_ref = f"{PROJECT}.{FEATURE_DATASET}.{STAGING_TABLE}"
    staging_schema = [bigquery.SchemaField("event_id", "STRING", mode="REQUIRED")] + [
        bigquery.SchemaField(name, sql_type) for name, sql_type in NEW_COLUMNS
    ]
    client.delete_table(staging_ref, not_found_ok=True)
    client.create_table(bigquery.Table(staging_ref, schema=staging_schema))
    job_config = bigquery.LoadJobConfig(schema=staging_schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_json(staging_rows, staging_ref, job_config=job_config, location=LOCATION).result()

    print("[phase_c] ensuring target columns exist...")
    add_clauses = ", ".join(f"ADD COLUMN IF NOT EXISTS `{name}` {sql_type}" for name, sql_type in NEW_COLUMNS)
    client.query(f"ALTER TABLE `{PROJECT}.{FEATURE_DATASET}.{TARGET_TABLE}` {add_clauses}", location=LOCATION).result()

    print("[phase_c] updating target table from staging...")
    set_clause = ", ".join(f"tgt.`{name}` = src.`{name}`" for name, _ in NEW_COLUMNS)
    update_sql = f"""
    UPDATE `{PROJECT}.{FEATURE_DATASET}.{TARGET_TABLE}` AS tgt
    SET {set_clause}
    FROM `{staging_ref}` AS src
    WHERE tgt.event_id = src.event_id
    """
    client.query(update_sql, location=LOCATION).result()
    client.delete_table(staging_ref, not_found_ok=True)

    print("[phase_c] refreshing cxg_training_matrix_v1 view (existing, unmodified function)...")
    _create_training_view(client, f"{PROJECT}.{FEATURE_DATASET}")

    verify_sql = f"""
    SELECT
      COUNT(*) AS n,
      COUNTIF(defensive_action_rate_15m IS NOT NULL) AS n_rate_15m,
      COUNTIF(defensive_action_rate_null_reason IS NOT NULL) AS n_rate_null,
      COUNTIF(territorial_dominance_last_15m IS NOT NULL) AS n_field_tilt,
      COUNTIF(cross_match_defensive_rate IS NOT NULL) AS n_cross_match,
      COUNTIF(cross_match_defensive_rate_null_reason IS NOT NULL) AS n_cross_match_null
    FROM `{PROJECT}.{FEATURE_DATASET}.{TARGET_TABLE}`
    """
    summary = dict(list(client.query(verify_sql, location=LOCATION).result())[0].items())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
