"""Materialize the governed one-row-per-shot CxG/CxG+ Gold snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from google.cloud import bigquery

from audit_cxg_e13_f1_f15 import (
    DATA_VERSION,
    DATASET,
    LOCATION,
    PROJECT,
    SCHEMA_VERSION,
    _client,
    _fetch_events,
    _fetch_frames,
    _match_ids,
)
from opponent_adjusted.features.cxg.event_context import derive_event_contexts
from opponent_adjusted.features.cxg.event_context_e13 import derive_e13_contexts
from opponent_adjusted.features.cxg.event_context_extended import derive_extended_event_contexts
from opponent_adjusted.features.cxg.three_sixty_context import derive_360_contexts


FEATURE_VERSION = "cxg_gold_v1_e13_f1_f15"
TABLE = f"{PROJECT}.{DATASET}.cxg_shot_features"


def main() -> None:
    client = _client()
    match_ids = _match_ids(client, only_with_360=False)
    frame_match_ids = set(_match_ids(client, only_with_360=True))
    rows: list[dict] = []
    for start in range(0, len(match_ids), 5):
        batch = match_ids[start : start + 5]
        events_by_match = _fetch_events(client, batch)
        events = [event for match_events in events_by_match.values() for event in match_events]
        frames = _fetch_frames(client, [m for m in batch if m in frame_match_ids])
        base = derive_event_contexts(events)
        extended = derive_extended_event_contexts(events)
        e13 = derive_e13_contexts(events)
        f = derive_360_contexts(events, frames)
        for event in events:
            if event.event_type_name != "Shot":
                continue
            values = {}
            values.update(base[event.event_id].values)
            values.update(extended[event.event_id].values)
            values.update(e13[event.event_id].values)
            values.update(f[event.event_id])
            rows.append(
                {
                    "event_id": event.event_id,
                    "match_id": event.match_id,
                    "period": event.period,
                    "event_index": event.event_index,
                    "team_id": event.team_id,
                    "player_id": event.player_id,
                    "shot_x_sb": event.location_x,
                    "shot_y_sb": event.location_y,
                    "shot_outcome_name": event.outcome_name,
                    "is_goal": event.outcome_name == "Goal" if event.outcome_name else None,
                    "has_360_frame": event.event_id in frames,
                    "event_context_valid": base[event.event_id].possession_context_valid,
                    "feature_values_json": json.dumps(values, separators=(",", ":"), default=str),
                    "data_version": DATA_VERSION,
                    "silver_schema_version": SCHEMA_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "materialized_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("match_id", "INT64"),
        bigquery.SchemaField("period", "INT64"),
        bigquery.SchemaField("event_index", "INT64"),
        bigquery.SchemaField("team_id", "INT64"),
        bigquery.SchemaField("player_id", "INT64"),
        bigquery.SchemaField("shot_x_sb", "FLOAT64"),
        bigquery.SchemaField("shot_y_sb", "FLOAT64"),
        bigquery.SchemaField("shot_outcome_name", "STRING"),
        bigquery.SchemaField("is_goal", "BOOL"),
        bigquery.SchemaField("has_360_frame", "BOOL"),
        bigquery.SchemaField("event_context_valid", "BOOL"),
        bigquery.SchemaField("feature_values_json", "JSON"),
        bigquery.SchemaField("data_version", "STRING"),
        bigquery.SchemaField("silver_schema_version", "STRING"),
        bigquery.SchemaField("feature_version", "STRING"),
        bigquery.SchemaField("materialized_at", "TIMESTAMP"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_json(rows, TABLE, job_config=job_config, location=LOCATION)
    job.result()
    print(json.dumps({"table": TABLE, "rows": len(rows), "feature_version": FEATURE_VERSION}))


if __name__ == "__main__":
    main()
