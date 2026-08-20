"""Read-only real-data audit for CxG E13 and CxG+ F1-F15 (feature/cxg-e13-f1-f15).

Batches over the governed Silver corpus, checkpointing to a Windows temp
directory per the job's "temp audit output" requirement. BigQuery access is
read-only; no Silver/Gold table is mutated.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from google.cloud import bigquery

from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.event_context_e13 import E13_FEATURES, derive_e13_contexts
from opponent_adjusted.features.cxg.three_sixty_context import ALL_360_FEATURES, derive_360_contexts
from opponent_adjusted.features.cxg.three_sixty_frame import Frame, FramePlayer

PROJECT = "oam-varun-260819"
DATASET = "oam_core"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

TEMP_ROOT = Path(r"C:\Users\USER\AppData\Local\Temp\cxg_e13_f_audit")
CHECKPOINT_PATH = TEMP_ROOT / "checkpoint.json"
BATCH_RESULTS_PATH = TEMP_ROOT / "batch_results.ndjson"
LOG_PATH = TEMP_ROOT / "audit.log"
FINAL_RESULT_PATH = TEMP_ROOT / "final_result.json"

EVENTS_QUERY = f"""
SELECT
  e.event_id, e.match_id, e.event_index, e.period, e.minute, e.second, e.timestamp,
  e.event_type_name, e.team_id, e.possession_id, e.possession_team_id,
  e.location_x, e.location_y, e.player_id, e.play_pattern_name, e.duration,
  e.counterpress, e.under_pressure, d.card_name,
  s.outcome_name AS shot_outcome_name,
  COALESCE(p.end_x, c.end_x) AS end_x, COALESCE(p.end_y, c.end_y) AS end_y,
  COALESCE(p.outcome_name, dr.outcome_name, br.outcome_name) AS action_outcome_name
FROM `{PROJECT}.{DATASET}.events` e
LEFT JOIN `{PROJECT}.{DATASET}.shots` s
  ON e.event_id = s.event_id AND e.data_version = s.data_version AND e.silver_schema_version = s.silver_schema_version
LEFT JOIN `{PROJECT}.{DATASET}.passes` p
  ON e.event_id = p.event_id AND e.data_version = p.data_version AND e.silver_schema_version = p.silver_schema_version
LEFT JOIN `{PROJECT}.{DATASET}.carries` c
  ON e.event_id = c.event_id AND e.data_version = c.data_version AND e.silver_schema_version = c.silver_schema_version
LEFT JOIN `{PROJECT}.{DATASET}.dribbles` dr
  ON e.event_id = dr.event_id AND e.data_version = dr.data_version AND e.silver_schema_version = dr.silver_schema_version
LEFT JOIN `{PROJECT}.{DATASET}.ball_receipts` br
  ON e.event_id = br.event_id AND e.data_version = br.data_version AND e.silver_schema_version = br.silver_schema_version
LEFT JOIN `{PROJECT}.{DATASET}.disciplinary_events` d
  ON e.event_id = d.event_id AND e.data_version = d.data_version AND e.silver_schema_version = d.silver_schema_version
WHERE e.data_version = @data_version AND e.silver_schema_version = @schema_version
  AND e.match_id IN UNNEST(@match_ids)
ORDER BY e.match_id, e.period, e.event_index
"""

FRAMES_QUERY = f"""
SELECT
  f.match_id, f.event_uuid, f.visible_area,
  p.frame_player_ordinal, p.teammate, p.actor, p.keeper, p.x, p.y
FROM `{PROJECT}.{DATASET}.three_sixty_frames` f
JOIN `{PROJECT}.{DATASET}.three_sixty_players` p
  ON f.match_id = p.match_id AND f.event_uuid = p.event_uuid
  AND f.data_version = p.data_version AND f.silver_schema_version = p.silver_schema_version
WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
  AND f.match_id IN UNNEST(@match_ids)
ORDER BY f.match_id, f.event_uuid, p.frame_player_ordinal
"""


def _log(message: str) -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf8") as fh:
        fh.write(message + "\n")
    print(message, flush=True)


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _match_ids(client: bigquery.Client, only_with_360: bool) -> list[int]:
    table = "three_sixty_frames" if only_with_360 else "events"
    query = f"""
        SELECT DISTINCT match_id FROM `{PROJECT}.{DATASET}.{table}`
        WHERE data_version = @data_version AND silver_schema_version = @schema_version
        ORDER BY match_id
    """
    params = [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return [
        r.match_id for r in client.query(query, job_config=job_config, location=LOCATION).result()
    ]


def _fetch_events(client: bigquery.Client, match_ids: list[int]) -> dict[int, list[EventRecord]]:
    params = [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
        bigquery.ArrayQueryParameter("match_ids", "INT64", match_ids),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows: dict[int, list[EventRecord]] = defaultdict(list)
    for x in client.query(EVENTS_QUERY, job_config=job_config, location=LOCATION).result(
        page_size=20000
    ):
        outcome_name = x.shot_outcome_name
        rows[x.match_id].append(
            EventRecord(
                event_id=x.event_id,
                match_id=x.match_id,
                event_index=x.event_index,
                period=x.period,
                minute=x.minute,
                second=x.second,
                timestamp=x.timestamp,
                event_type_name=x.event_type_name,
                outcome_name=outcome_name,
                team_id=x.team_id,
                possession_id=x.possession_id,
                possession_team_id=x.possession_team_id,
                location_x=x.location_x,
                location_y=x.location_y,
                player_id=x.player_id,
                play_pattern_name=x.play_pattern_name,
                card_name=x.card_name,
                end_x=x.end_x,
                end_y=x.end_y,
                duration=x.duration,
                counterpress=x.counterpress,
                under_pressure=x.under_pressure,
                action_outcome_name=x.action_outcome_name,
            )
        )
    return rows


def _fetch_frames(client: bigquery.Client, match_ids: list[int]) -> dict[str, Frame]:
    params = [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
        bigquery.ArrayQueryParameter("match_ids", "INT64", match_ids),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    raw: dict[str, dict] = {}
    for x in client.query(FRAMES_QUERY, job_config=job_config, location=LOCATION).result(
        page_size=50000
    ):
        entry = raw.setdefault(
            x.event_uuid,
            {"match_id": x.match_id, "visible_area": tuple(x.visible_area or ()), "players": []},
        )
        entry["players"].append(
            FramePlayer(x.frame_player_ordinal, x.teammate, x.actor, x.keeper, x.x, x.y)
        )
    return {
        event_uuid: Frame(
            event_uuid, entry["match_id"], entry["visible_area"], tuple(entry["players"])
        )
        for event_uuid, entry in raw.items()
    }


def _numeric_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    quantiles = (
        statistics.quantiles(ordered, n=20, method="inclusive")
        if len(ordered) >= 2
        else ordered * 19
    )
    return {
        "min": ordered[0],
        "p05": quantiles[0] if len(ordered) >= 2 else ordered[0],
        "p25": quantiles[4] if len(ordered) >= 2 else ordered[0],
        "median": statistics.median(ordered),
        "p75": quantiles[14] if len(ordered) >= 2 else ordered[0],
        "p95": quantiles[18] if len(ordered) >= 2 else ordered[0],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-batches", type=int, default=None)
    args = parser.parse_args()

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    client = _client()

    all_match_ids = _match_ids(client, only_with_360=False)
    frame_match_ids = set(_match_ids(client, only_with_360=True))
    _log(f"total_matches={len(all_match_ids)} matches_with_360={len(frame_match_ids)}")

    e13_values: dict[str, list] = defaultdict(list)
    e13_null = Counter()
    e13_total = 0
    batch_vs_per_match_mismatches = 0

    f_values_all: dict[str, list] = defaultdict(list)
    f_null_all_shots = Counter()
    f_null_eligible_frames = Counter()
    total_shots = 0
    shots_with_frame = 0
    f10_eligible = Counter()

    batch_size = 5
    batches = [all_match_ids[i : i + batch_size] for i in range(0, len(all_match_ids), batch_size)]
    if args.first_batches:
        batches = batches[: args.first_batches]

    BATCH_RESULTS_PATH.write_text("", encoding="utf8")
    for i, batch in enumerate(batches):
        events_by_match = _fetch_events(client, batch)
        needs_frames = [m for m in batch if m in frame_match_ids]
        frames = _fetch_frames(client, needs_frames) if needs_frames else {}

        batch_events = [
            event for match_events in events_by_match.values() for event in match_events
        ]
        e13_batch = derive_e13_contexts(batch_events)
        f_batch = derive_360_contexts(batch_events, frames)

        # Batch-vs-per-match consistency check (section 51).
        for match_id, match_events in events_by_match.items():
            per_match_e13 = derive_e13_contexts(match_events)
            for event_id, ctx in per_match_e13.items():
                if e13_batch[event_id].values != ctx.values:
                    batch_vs_per_match_mismatches += 1

        for event_id, ctx in e13_batch.items():
            e13_total += 1
            for name in E13_FEATURES:
                value = ctx.values[name]
                if value is None:
                    e13_null[name] += 1
                else:
                    e13_values[name].append(value)

        for event_id, values in f_batch.items():
            total_shots += 1
            has_frame = event_id in frames
            if has_frame:
                shots_with_frame += 1
            for name in ALL_360_FEATURES:
                value = values[name]
                if value is None:
                    f_null_all_shots[name] += 1
                    if has_frame:
                        f_null_eligible_frames[name] += 1
                else:
                    f_values_all[name].append(value)
            if has_frame:
                for name in ("pre_shot_receiver_space", "shooter_space_previous_linked_event"):
                    if values[name] is not None:
                        f10_eligible[name] += 1

        with BATCH_RESULTS_PATH.open("a", encoding="utf8") as fh:
            fh.write(
                json.dumps(
                    {
                        "batch_start": i * batch_size,
                        "match_ids": batch,
                        "events": len(batch_events),
                        "shots": len(f_batch),
                        "shots_with_frame_running_total": shots_with_frame,
                    }
                )
                + "\n"
            )
        CHECKPOINT_PATH.write_text(
            json.dumps({"completed_batches": i + 1, "total_batches": len(batches)}), encoding="utf8"
        )
        _log(f"batch {i + 1}/{len(batches)} events={len(batch_events)} shots={len(f_batch)}")

    e13_summary = {}
    for name in E13_FEATURES:
        vals = e13_values[name]
        numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        categorical = [v for v in vals if isinstance(v, (str, bool))]
        e13_summary[name] = {
            "null": e13_null[name],
            "non_null": len(vals),
            "numeric_stats": _numeric_stats(numeric) if numeric else None,
            "category_counts": dict(Counter(categorical)) if categorical else None,
        }

    f_summary = {}
    for name in ALL_360_FEATURES:
        vals = f_values_all[name]
        numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        f_summary[name] = {
            "null_all_shots": f_null_all_shots[name],
            "null_eligible_frames": f_null_eligible_frames[name],
            "non_null": len(vals),
            "numeric_stats": _numeric_stats(numeric) if numeric else None,
            "nonfinite_count": sum(
                1 for v in numeric if v != v or v in (float("inf"), float("-inf"))
            ),
        }

    result = {
        "total_matches": len(all_match_ids),
        "matches_with_360": len(frame_match_ids),
        "batches_processed": len(batches),
        "e13_total_shots": e13_total,
        "e13_batch_vs_per_match_mismatches": batch_vs_per_match_mismatches,
        "e13_summary": e13_summary,
        "total_shots": total_shots,
        "shots_with_frame": shots_with_frame,
        "shot_360_coverage": shots_with_frame / total_shots if total_shots else None,
        "f10_eligible_counts": dict(f10_eligible),
        "f_summary": f_summary,
    }
    FINAL_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf8")
    _log(f"DONE -> {FINAL_RESULT_PATH}")


if __name__ == "__main__":
    main()
