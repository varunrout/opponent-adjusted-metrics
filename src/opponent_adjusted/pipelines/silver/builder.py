"""Build StatsBomb Silver v1 tables from immutable Bronze GCS JSON."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage  # type: ignore[import-untyped]

from opponent_adjusted.ingestion.contracts import load_subset_config
from opponent_adjusted.pipelines.silver.contracts import (
    CONTRACTS,
    table_arrow_schema,
    write_contract_json,
)
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
PARTITIONED_TABLES = {
    "events",
    "shots",
    "passes",
    "carries",
    "dribbles",
    "pressures",
    "ball_receipts",
    "disciplinary_events",
    "substitutions",
    "possessions",
    "three_sixty_frames",
    "three_sixty_players",
    "shot_freeze_frame_players",
}


@dataclass(frozen=True)
class SilverBuildConfig:
    bucket_name: str
    artifacts_bucket: str
    data_version: str
    source_ref: str
    silver_schema_version: str
    competitions_config_path: Path
    output_prefix: str
    project_id: str


@dataclass(frozen=True)
class SilverBuildResult:
    run_id: str
    output_prefix: str
    manifest_uri: str
    artifacts_manifest_uri: str
    row_counts: dict[str, int]
    object_counts: dict[str, int]
    bytes_by_table: dict[str, int]
    warnings: list[str]
    bronze_summary: dict[str, Any]
    qa_summary: dict[str, Any]


class _BufferedWriter:
    def __init__(self, schema: pa.Schema, path: Path, flush_rows: int = 5000) -> None:
        self.schema = schema
        self.path = path
        self.flush_rows = flush_rows
        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def append(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path.as_posix(), self.schema, compression="snappy")
        self._writer.write_table(table)
        self.row_count += len(self._buffer)
        self._buffer = []

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()


def _nested_id(obj: dict[str, Any], key: str) -> int | None:
    value = obj.get(key) or {}
    if isinstance(value, dict) and value.get("id") is not None:
        return int(value["id"])
    return None


def _nested_name(obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key) or {}
    if isinstance(value, dict) and value.get("name") is not None:
        return str(value["name"])
    return None


def _location_xy(event: dict[str, Any]) -> tuple[float | None, float | None]:
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        try:
            return float(loc[0]), float(loc[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _bool_or_none(obj: dict[str, Any], key: str) -> bool | None:
    if key in obj:
        return bool(obj.get(key))
    return None


def _table_path(root: Path, table: str, competition_id: int | None, season_id: int | None) -> Path:
    if table in PARTITIONED_TABLES and competition_id is not None and season_id is not None:
        return (
            root
            / table
            / f"competition_id={competition_id}"
            / f"season_id={season_id}"
            / "part-00000.parquet"
        )
    return root / table / "part-00000.parquet"


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def _load_json_blob(bucket: storage.Bucket, name: str) -> list | dict:
    blob = bucket.blob(name)
    payload = json.loads(blob.download_as_bytes().decode("utf-8"))
    if not isinstance(payload, (list, dict)):
        raise ValueError(f"Unexpected JSON root in {name}")
    return payload


def _upload_create_only(bucket: storage.Bucket, object_name: str, local_path: Path) -> str:
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_filename(local_path.as_posix(), if_generation_match=0)
        return "uploaded"
    except PreconditionFailed:
        blob.reload()
        local_md5 = _md5_b64(local_path)
        local_size = local_path.stat().st_size
        if blob.md5_hash and blob.md5_hash == local_md5 and blob.size == local_size:
            return "skipped_identical"
        raise RuntimeError(
            f"Existing object differs from local output: gs://{bucket.name}/{object_name}"
        )


def _bronze_verify(bucket: storage.Bucket, prefix: str) -> dict[str, Any]:
    required = [
        f"{prefix}/competitions.json",
        f"{prefix}/matches",
        f"{prefix}/events",
        f"{prefix}/three-sixty",
    ]
    found = {}
    for item in required:
        if item.endswith(".json"):
            found[item] = bucket.blob(item).exists()
        else:
            found[item] = any(bucket.list_blobs(prefix=f"{item}/", max_results=1))
    return {"prefix": f"gs://{bucket.name}/{prefix}/", "required": found}


def _manifest_for_local_root(
    run_id: str,
    config: SilverBuildConfig,
    local_root: Path,
    row_counts: dict[str, int],
    qa_summary: dict[str, Any],
    warnings: list[str],
    bronze_summary: dict[str, Any],
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    table_stats: dict[str, dict[str, Any]] = {}
    for table_name in CONTRACTS:
        table_dir = local_root / table_name
        parquet_files = list(table_dir.rglob("*.parquet")) if table_dir.exists() else []
        object_count = len(parquet_files)
        total_bytes = sum(p.stat().st_size for p in parquet_files)
        partitions = sorted(
            {
                str(p.parent.relative_to(local_root / table_name)).replace("\\", "/")
                for p in parquet_files
            }
        )
        schema_hash = hashlib.sha256(
            table_arrow_schema(table_name).to_string().encode("utf-8")
        ).hexdigest()
        table_stats[table_name] = {
            "row_count": row_counts.get(table_name, 0),
            "parquet_object_count": object_count,
            "total_bytes": total_bytes,
            "schema_fingerprint": schema_hash,
            "partitions": partitions,
            "qa_status": "pass",
        }

    return {
        "pipeline_name": "statsbomb_silver_build",
        "run_id": run_id,
        "git_commit_sha": os.getenv("GIT_COMMIT_SHA", "unknown"),
        "project_id": config.project_id,
        "source_ref": config.source_ref,
        "data_version": config.data_version,
        "silver_schema_version": config.silver_schema_version,
        "source_prefix": f"gs://{config.bucket_name}/raw/statsbomb/{config.data_version}/",
        "output_prefix": f"gs://{config.bucket_name}/{config.output_prefix}/",
        "start_time": started_at,
        "end_time": ended_at,
        "terminal_status": "success",
        "warnings": warnings,
        "errors": qa_summary.get("errors", []),
        "bronze_verification": bronze_summary,
        "quality_summary": qa_summary,
        "tables": table_stats,
    }


def build_statsbomb_silver(config: SilverBuildConfig) -> SilverBuildResult:
    started_at = datetime.now(UTC).isoformat()
    run_id = f"silver-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    client = storage.Client(project=config.project_id)
    bucket = client.bucket(config.bucket_name)
    artifacts_bucket = client.bucket(config.artifacts_bucket)

    source_prefix = f"raw/statsbomb/{config.data_version}"
    output_prefix = config.output_prefix.rstrip("/")

    bronze_summary = _bronze_verify(bucket, source_prefix)
    if not all(bronze_summary["required"].values()):
        raise RuntimeError(f"Bronze verification failed: {bronze_summary}")

    success_blob = bucket.blob(f"{output_prefix}/_SUCCESS")
    if success_blob.exists(client=client):
        raise RuntimeError(
            f"Silver output already published and immutable at gs://{config.bucket_name}/{output_prefix}/"
        )

    contract_path = write_contract_json()
    logger.info("Silver contract written at %s", contract_path)

    cfg = load_subset_config(config.competitions_config_path)
    competitions_filter = {
        (int(c["competition_id"]), int(c["season_id"])) for c in cfg.get("competitions", [])
    }

    competitions_payload = _load_json_blob(bucket, f"{source_prefix}/competitions.json")
    if not isinstance(competitions_payload, list):
        raise RuntimeError("Bronze competitions payload is not a list")

    local_root = Path(tempfile.mkdtemp(prefix=f"statsbomb_silver_{run_id}_"))
    writers: dict[tuple[str, int | None, int | None], _BufferedWriter] = {}
    row_counts: dict[str, int] = defaultdict(int)

    qa_errors: list[str] = []
    qa_warnings: list[str] = []

    event_ids_seen: set[str] = set()
    match_ids_seen: set[int] = set()
    shot_ids: set[str] = set()
    shot_ff_ids: set[str] = set()
    subtype_ids: dict[str, set[str]] = defaultdict(set)
    frame_keys: set[tuple[int, str]] = set()

    teams: dict[int, str] = {}
    players: dict[int, str] = {}

    event_type_counts: dict[str, int] = defaultdict(int)

    def get_writer(
        table: str, comp: int | None = None, season: int | None = None
    ) -> _BufferedWriter:
        key = (table, comp, season)
        if key not in writers:
            writers[key] = _BufferedWriter(
                schema=table_arrow_schema(table),
                path=_table_path(local_root, table, comp, season),
                flush_rows=5000,
            )
        return writers[key]

    def write_row(
        table: str, row: dict[str, Any], comp: int | None = None, season: int | None = None
    ) -> None:
        get_writer(table, comp, season).append(row)
        row_counts[table] += 1

    selected_competitions: list[dict[str, Any]] = []
    for comp in competitions_payload:
        if not isinstance(comp, dict):
            continue
        key = (int(comp.get("competition_id")), int(comp.get("season_id")))
        if key in competitions_filter:
            selected_competitions.append(comp)
            write_row(
                "competitions",
                {
                    "competition_id": key[0],
                    "season_id": key[1],
                    "competition_name": comp.get("competition_name"),
                    "competition_gender": comp.get("competition_gender"),
                    "country_name": comp.get("country_name"),
                    "season_name": comp.get("season_name"),
                    "match_updated": comp.get("match_updated"),
                    "match_available": comp.get("match_available"),
                    "match_updated_360": comp.get("match_updated_360"),
                    "match_available_360": comp.get("match_available_360"),
                    "data_version": config.data_version,
                    "silver_schema_version": config.silver_schema_version,
                },
            )

    for competition in selected_competitions:
        competition_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])

        matches_payload = _load_json_blob(
            bucket, f"{source_prefix}/matches/{competition_id}/{season_id}.json"
        )
        if not isinstance(matches_payload, list):
            qa_errors.append(f"matches payload not list for {competition_id}/{season_id}")
            continue

        for match in matches_payload:
            if not isinstance(match, dict):
                continue
            match_id = int(match["match_id"])
            if match_id in match_ids_seen:
                qa_errors.append(f"duplicate match_id {match_id}")
            match_ids_seen.add(match_id)

            home = match.get("home_team") or {}
            away = match.get("away_team") or {}
            home_team_id = (
                int(home["home_team_id"]) if home.get("home_team_id") is not None else None
            )
            away_team_id = (
                int(away["away_team_id"]) if away.get("away_team_id") is not None else None
            )
            home_team_name = home.get("home_team_name")
            away_team_name = away.get("away_team_name")

            write_row(
                "matches",
                {
                    "match_id": match_id,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "match_date": match.get("match_date"),
                    "kick_off": match.get("kick_off"),
                    "home_team_id": home_team_id,
                    "home_team_name": home_team_name,
                    "away_team_id": away_team_id,
                    "away_team_name": away_team_name,
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "competition_stage": _nested_name(match, "competition_stage"),
                    "stadium": _nested_name(match, "stadium"),
                    "referee": _nested_name(match, "referee"),
                    "match_status": match.get("match_status"),
                    "match_status_360": match.get("match_status_360"),
                    "last_updated": match.get("last_updated"),
                    "last_updated_360": match.get("last_updated_360"),
                    "data_version": config.data_version,
                    "silver_schema_version": config.silver_schema_version,
                },
            )

            if home_team_id is not None:
                teams[home_team_id] = (
                    str(home_team_name)
                    if home_team_name is not None
                    else teams.get(home_team_id, "")
                )
            if away_team_id is not None:
                teams[away_team_id] = (
                    str(away_team_name)
                    if away_team_name is not None
                    else teams.get(away_team_id, "")
                )

            events_payload = _load_json_blob(bucket, f"{source_prefix}/events/{match_id}.json")
            if not isinstance(events_payload, list):
                qa_errors.append(f"events payload not list for match {match_id}")
                continue

            possession_map: dict[int, dict[str, Any]] = {}

            for idx, event in enumerate(events_payload):
                if not isinstance(event, dict):
                    qa_errors.append(f"malformed event object in match {match_id} index {idx}")
                    continue
                event_id = event.get("id")
                if not event_id or not isinstance(event_id, str):
                    qa_errors.append(f"missing required event_id in match {match_id} index {idx}")
                    continue
                if event_id in event_ids_seen:
                    qa_errors.append(f"duplicate event_id {event_id}")
                event_ids_seen.add(event_id)

                period = event.get("period")
                minute = event.get("minute")
                second = event.get("second")
                timestamp = event.get("timestamp")
                event_type = event.get("type") or {}
                event_type_id = event_type.get("id")
                event_type_name = event_type.get("name")
                event_type_counts[str(event_type_name)] += 1

                team_id = _nested_id(event, "team")
                team_name = _nested_name(event, "team")
                player_id = _nested_id(event, "player")
                player_name = _nested_name(event, "player")
                position_id = _nested_id(event, "position")
                position_name = _nested_name(event, "position")
                possession_team_id = _nested_id(event, "possession_team")
                possession_team_name = _nested_name(event, "possession_team")
                play_pattern_id = _nested_id(event, "play_pattern")
                play_pattern_name = _nested_name(event, "play_pattern")
                location_x, location_y = _location_xy(event)

                if location_x is not None and not (0.0 <= location_x <= 120.0):
                    qa_warnings.append(
                        f"event coordinate x out of range event_id={event_id} x={location_x}"
                    )
                if location_y is not None and not (0.0 <= location_y <= 80.0):
                    qa_warnings.append(
                        f"event coordinate y out of range event_id={event_id} y={location_y}"
                    )

                if team_id is not None and team_name is not None:
                    teams[team_id] = str(team_name)
                if player_id is not None and player_name is not None:
                    players[player_id] = str(player_name)

                related = event.get("related_events")
                related_ids: list[str] = []
                if isinstance(related, list):
                    related_ids = [str(x) for x in related if isinstance(x, str)]

                event_row = {
                    "event_id": event_id,
                    "match_id": match_id,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "event_index": idx,
                    "period": int(period) if period is not None else None,
                    "minute": int(minute) if minute is not None else None,
                    "second": int(second) if second is not None else None,
                    "timestamp": timestamp,
                    "duration": (
                        float(event["duration"]) if event.get("duration") is not None else None
                    ),
                    "event_type_id": int(event_type_id) if event_type_id is not None else None,
                    "event_type_name": (
                        str(event_type_name) if event_type_name is not None else None
                    ),
                    "possession_id": (
                        int(event["possession"]) if event.get("possession") is not None else None
                    ),
                    "possession_team_id": possession_team_id,
                    "possession_team_name": possession_team_name,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position_id": position_id,
                    "position_name": position_name,
                    "play_pattern_id": play_pattern_id,
                    "play_pattern_name": play_pattern_name,
                    "under_pressure": _bool_or_none(event, "under_pressure"),
                    "counterpress": _bool_or_none(event, "counterpress"),
                    "off_camera": _bool_or_none(event, "off_camera"),
                    "out": _bool_or_none(event, "out"),
                    "location_x": location_x,
                    "location_y": location_y,
                    "related_event_ids": related_ids,
                    "data_version": config.data_version,
                    "silver_schema_version": config.silver_schema_version,
                }
                write_row("events", event_row, competition_id, season_id)

                possession_id = event_row["possession_id"]
                if possession_id is not None:
                    agg = possession_map.get(possession_id)
                    if agg is None:
                        agg = {
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "possession_id": possession_id,
                            "possession_team_id": possession_team_id,
                            "possession_team_name": possession_team_name,
                            "start_event_id": event_id,
                            "end_event_id": event_id,
                            "start_event_index": idx,
                            "end_event_index": idx,
                            "start_period": event_row["period"],
                            "end_period": event_row["period"],
                            "start_timestamp": timestamp,
                            "end_timestamp": timestamp,
                            "start_minute": event_row["minute"],
                            "start_second": event_row["second"],
                            "end_minute": event_row["minute"],
                            "end_second": event_row["second"],
                            "event_count": 1,
                            "start_event_type": event_row["event_type_name"],
                            "end_event_type": event_row["event_type_name"],
                            "start_play_pattern": play_pattern_name,
                            "end_play_pattern": play_pattern_name,
                        }
                        possession_map[possession_id] = agg
                    else:
                        agg["end_event_id"] = event_id
                        agg["end_event_index"] = idx
                        agg["end_period"] = event_row["period"]
                        agg["end_timestamp"] = timestamp
                        agg["end_minute"] = event_row["minute"]
                        agg["end_second"] = event_row["second"]
                        agg["end_event_type"] = event_row["event_type_name"]
                        agg["end_play_pattern"] = play_pattern_name
                        agg["event_count"] += 1

                if event_type_name == "Shot":
                    shot_ids.add(event_id)
                    shot = event.get("shot") or {}
                    end_loc = shot.get("end_location") or []
                    write_row(
                        "shots",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "location_x": location_x,
                            "location_y": location_y,
                            "end_x": (
                                float(end_loc[0])
                                if isinstance(end_loc, list) and len(end_loc) >= 1
                                else None
                            ),
                            "end_y": (
                                float(end_loc[1])
                                if isinstance(end_loc, list) and len(end_loc) >= 2
                                else None
                            ),
                            "end_z": (
                                float(end_loc[2])
                                if isinstance(end_loc, list) and len(end_loc) >= 3
                                else None
                            ),
                            "statsbomb_xg": (
                                float(shot["statsbomb_xg"])
                                if shot.get("statsbomb_xg") is not None
                                else None
                            ),
                            "outcome_id": _nested_id(shot, "outcome"),
                            "outcome_name": _nested_name(shot, "outcome"),
                            "body_part_id": _nested_id(shot, "body_part"),
                            "body_part_name": _nested_name(shot, "body_part"),
                            "technique_id": _nested_id(shot, "technique"),
                            "technique_name": _nested_name(shot, "technique"),
                            "shot_type_id": _nested_id(shot, "type"),
                            "shot_type_name": _nested_name(shot, "type"),
                            "first_time": _bool_or_none(shot, "first_time"),
                            "key_pass_id": shot.get("key_pass_id"),
                            "aerial_won": _bool_or_none(shot, "aerial_won"),
                            "follows_dribble": _bool_or_none(shot, "follows_dribble"),
                            "open_goal": _bool_or_none(shot, "open_goal"),
                            "one_on_one": _bool_or_none(shot, "one_on_one"),
                            "deflected": _bool_or_none(shot, "deflected"),
                            "saved_off_target": _bool_or_none(shot, "saved_off_target"),
                            "saved_to_post": _bool_or_none(shot, "saved_to_post"),
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["shots"].add(event_id)

                    freeze = shot.get("freeze_frame")
                    if isinstance(freeze, list):
                        for i, fp in enumerate(freeze):
                            if not isinstance(fp, dict):
                                continue
                            fplayer = fp.get("player") or {}
                            ppos = fp.get("position") or {}
                            ploc = fp.get("location") or []
                            fp_id = fplayer.get("id")
                            fp_name = fplayer.get("name")
                            if fp_id is not None and fp_name is not None:
                                players[int(fp_id)] = str(fp_name)
                            write_row(
                                "shot_freeze_frame_players",
                                {
                                    "shot_event_id": event_id,
                                    "match_id": match_id,
                                    "competition_id": competition_id,
                                    "season_id": season_id,
                                    "freeze_frame_player_ordinal": i,
                                    "player_id": int(fp_id) if fp_id is not None else None,
                                    "player_name": str(fp_name) if fp_name is not None else None,
                                    "position_id": (
                                        int(ppos["id"]) if ppos.get("id") is not None else None
                                    ),
                                    "position_name": (
                                        str(ppos.get("name"))
                                        if ppos.get("name") is not None
                                        else None
                                    ),
                                    "teammate": _bool_or_none(fp, "teammate"),
                                    "x": (
                                        float(ploc[0])
                                        if isinstance(ploc, list) and len(ploc) >= 1
                                        else None
                                    ),
                                    "y": (
                                        float(ploc[1])
                                        if isinstance(ploc, list) and len(ploc) >= 2
                                        else None
                                    ),
                                    "data_version": config.data_version,
                                    "silver_schema_version": config.silver_schema_version,
                                },
                                competition_id,
                                season_id,
                            )
                            shot_ff_ids.add(event_id)

                if event_type_name == "Pass":
                    p = event.get("pass") or {}
                    end_loc = p.get("end_location") or []
                    rec = p.get("recipient") or {}
                    rec_id = rec.get("id")
                    rec_name = rec.get("name")
                    if rec_id is not None and rec_name is not None:
                        players[int(rec_id)] = str(rec_name)
                    write_row(
                        "passes",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "recipient_id": int(rec_id) if rec_id is not None else None,
                            "recipient_name": str(rec_name) if rec_name is not None else None,
                            "length": float(p["length"]) if p.get("length") is not None else None,
                            "angle": float(p["angle"]) if p.get("angle") is not None else None,
                            "height_id": _nested_id(p, "height"),
                            "height_name": _nested_name(p, "height"),
                            "end_x": (
                                float(end_loc[0])
                                if isinstance(end_loc, list) and len(end_loc) >= 1
                                else None
                            ),
                            "end_y": (
                                float(end_loc[1])
                                if isinstance(end_loc, list) and len(end_loc) >= 2
                                else None
                            ),
                            "pass_type_id": _nested_id(p, "type"),
                            "pass_type_name": _nested_name(p, "type"),
                            "outcome_id": _nested_id(p, "outcome"),
                            "outcome_name": _nested_name(p, "outcome"),
                            "technique_id": _nested_id(p, "technique"),
                            "technique_name": _nested_name(p, "technique"),
                            "body_part_id": _nested_id(p, "body_part"),
                            "body_part_name": _nested_name(p, "body_part"),
                            "assisted_shot_id": p.get("assisted_shot_id"),
                            "shot_assist": _bool_or_none(p, "shot_assist"),
                            "goal_assist": _bool_or_none(p, "goal_assist"),
                            "cross": _bool_or_none(p, "cross"),
                            "cut_back": _bool_or_none(p, "cut_back"),
                            "switch": _bool_or_none(p, "switch"),
                            "through_ball": _bool_or_none(p, "through_ball"),
                            "inswinging": _bool_or_none(p, "inswinging"),
                            "outswinging": _bool_or_none(p, "outswinging"),
                            "straight": _bool_or_none(p, "straight"),
                            "deflected": _bool_or_none(p, "deflected"),
                            "miscommunication": _bool_or_none(p, "miscommunication"),
                            "no_touch": _bool_or_none(p, "no_touch"),
                            "aerial_won": _bool_or_none(p, "aerial_won"),
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["passes"].add(event_id)

                if event_type_name == "Carry":
                    c = event.get("carry") or {}
                    end_loc = c.get("end_location") or []
                    write_row(
                        "carries",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "end_x": (
                                float(end_loc[0])
                                if isinstance(end_loc, list) and len(end_loc) >= 1
                                else None
                            ),
                            "end_y": (
                                float(end_loc[1])
                                if isinstance(end_loc, list) and len(end_loc) >= 2
                                else None
                            ),
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["carries"].add(event_id)

                if event_type_name == "Dribble":
                    d = event.get("dribble") or {}
                    write_row(
                        "dribbles",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "outcome_id": _nested_id(d, "outcome"),
                            "outcome_name": _nested_name(d, "outcome"),
                            "nutmeg": _bool_or_none(d, "nutmeg"),
                            "overrun": _bool_or_none(d, "overrun"),
                            "no_touch": _bool_or_none(d, "no_touch"),
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["dribbles"].add(event_id)

                if event_type_name == "Pressure":
                    write_row(
                        "pressures",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "period": int(period) if period is not None else None,
                            "minute": int(minute) if minute is not None else None,
                            "second": int(second) if second is not None else None,
                            "timestamp": timestamp,
                            "duration": (
                                float(event["duration"])
                                if event.get("duration") is not None
                                else None
                            ),
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["pressures"].add(event_id)

                if event_type_name in {"Ball Receipt", "Ball Receipt*"}:
                    b = event.get("ball_receipt") or {}
                    write_row(
                        "ball_receipts",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "outcome_id": _nested_id(b, "outcome"),
                            "outcome_name": _nested_name(b, "outcome"),
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["ball_receipts"].add(event_id)

                if event_type_name in {"Bad Behaviour", "Foul Committed"}:
                    card_src = (
                        event.get("bad_behaviour")
                        if event_type_name == "Bad Behaviour"
                        else event.get("foul_committed")
                    )
                    card_src = card_src or {}
                    write_row(
                        "disciplinary_events",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "event_type": str(event_type_name),
                            "card_id": _nested_id(card_src, "card"),
                            "card_name": _nested_name(card_src, "card"),
                            "period": int(period) if period is not None else None,
                            "minute": int(minute) if minute is not None else None,
                            "second": int(second) if second is not None else None,
                            "timestamp": timestamp,
                            "event_index": idx,
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["disciplinary_events"].add(event_id)

                if event_type_name == "Substitution":
                    s = event.get("substitution") or {}
                    rep = s.get("replacement") or {}
                    rep_id = rep.get("id")
                    rep_name = rep.get("name")
                    if rep_id is not None and rep_name is not None:
                        players[int(rep_id)] = str(rep_name)
                    write_row(
                        "substitutions",
                        {
                            "event_id": event_id,
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": season_id,
                            "team_id": team_id,
                            "player_id": player_id,
                            "replacement_id": int(rep_id) if rep_id is not None else None,
                            "replacement_name": str(rep_name) if rep_name is not None else None,
                            "outcome_id": _nested_id(s, "outcome"),
                            "outcome_name": _nested_name(s, "outcome"),
                            "period": int(period) if period is not None else None,
                            "minute": int(minute) if minute is not None else None,
                            "second": int(second) if second is not None else None,
                            "timestamp": timestamp,
                            "event_index": idx,
                            "data_version": config.data_version,
                            "silver_schema_version": config.silver_schema_version,
                        },
                        competition_id,
                        season_id,
                    )
                    subtype_ids["substitutions"].add(event_id)

            for pos_id, agg in possession_map.items():
                write_row(
                    "possessions",
                    {
                        "match_id": match_id,
                        "competition_id": agg["competition_id"],
                        "season_id": agg["season_id"],
                        "possession_id": pos_id,
                        "possession_team_id": agg["possession_team_id"],
                        "possession_team_name": agg["possession_team_name"],
                        "start_event_id": agg["start_event_id"],
                        "end_event_id": agg["end_event_id"],
                        "start_event_index": agg["start_event_index"],
                        "end_event_index": agg["end_event_index"],
                        "start_period": agg["start_period"],
                        "end_period": agg["end_period"],
                        "start_timestamp": agg["start_timestamp"],
                        "end_timestamp": agg["end_timestamp"],
                        "start_minute": agg["start_minute"],
                        "start_second": agg["start_second"],
                        "end_minute": agg["end_minute"],
                        "end_second": agg["end_second"],
                        "event_count": agg["event_count"],
                        "start_event_type": agg["start_event_type"],
                        "end_event_type": agg["end_event_type"],
                        "start_play_pattern": agg["start_play_pattern"],
                        "end_play_pattern": agg["end_play_pattern"],
                        "data_version": config.data_version,
                        "silver_schema_version": config.silver_schema_version,
                    },
                    competition_id,
                    season_id,
                )

            ts_blob_name = f"{source_prefix}/three-sixty/{match_id}.json"
            ts_blob = bucket.blob(ts_blob_name)
            if ts_blob.exists(client=client):
                payload = json.loads(ts_blob.download_as_bytes().decode("utf-8"))
                if not isinstance(payload, list):
                    qa_errors.append(f"malformed three-sixty payload for match {match_id}")
                else:
                    for frame in payload:
                        if not isinstance(frame, dict):
                            qa_errors.append(f"malformed three-sixty frame for match {match_id}")
                            continue
                        event_uuid = frame.get("event_uuid")
                        if not isinstance(event_uuid, str) or not event_uuid:
                            qa_errors.append(
                                f"three-sixty frame missing event_uuid for match {match_id}"
                            )
                            continue
                        key = (match_id, event_uuid)
                        if key in frame_keys:
                            qa_errors.append(f"duplicate three-sixty frame identity {key}")
                        frame_keys.add(key)
                        visible_area = frame.get("visible_area")
                        if not isinstance(visible_area, list):
                            qa_errors.append(
                                f"invalid visible_area for match {match_id} event_uuid {event_uuid}"
                            )
                            continue
                        visible_area_f = [
                            float(v) for v in visible_area if isinstance(v, (int, float))
                        ]
                        freeze = frame.get("freeze_frame") or []
                        if not isinstance(freeze, list):
                            qa_errors.append(
                                f"invalid freeze_frame for match {match_id} event_uuid {event_uuid}"
                            )
                            continue
                        write_row(
                            "three_sixty_frames",
                            {
                                "match_id": match_id,
                                "event_uuid": event_uuid,
                                "competition_id": competition_id,
                                "season_id": season_id,
                                "visible_area": visible_area_f,
                                "frame_player_count": len(freeze),
                                "data_version": config.data_version,
                                "silver_schema_version": config.silver_schema_version,
                            },
                            competition_id,
                            season_id,
                        )

                        for ord_idx, fp in enumerate(freeze):
                            if not isinstance(fp, dict):
                                qa_errors.append(
                                    f"invalid freeze_frame player for match {match_id} event_uuid {event_uuid}"
                                )
                                continue
                            loc = fp.get("location") or []
                            x = (
                                float(loc[0])
                                if isinstance(loc, list)
                                and len(loc) >= 1
                                and isinstance(loc[0], (int, float))
                                else None
                            )
                            y = (
                                float(loc[1])
                                if isinstance(loc, list)
                                and len(loc) >= 2
                                and isinstance(loc[1], (int, float))
                                else None
                            )
                            if x is not None and not (0.0 <= x <= 120.0):
                                qa_warnings.append(
                                    f"three-sixty x out of range match={match_id} event_uuid={event_uuid}"
                                )
                            if y is not None and not (0.0 <= y <= 80.0):
                                qa_warnings.append(
                                    f"three-sixty y out of range match={match_id} event_uuid={event_uuid}"
                                )
                            write_row(
                                "three_sixty_players",
                                {
                                    "match_id": match_id,
                                    "event_uuid": event_uuid,
                                    "competition_id": competition_id,
                                    "season_id": season_id,
                                    "frame_player_ordinal": ord_idx,
                                    "teammate": _bool_or_none(fp, "teammate"),
                                    "actor": _bool_or_none(fp, "actor"),
                                    "keeper": _bool_or_none(fp, "keeper"),
                                    "x": x,
                                    "y": y,
                                    "data_version": config.data_version,
                                    "silver_schema_version": config.silver_schema_version,
                                },
                                competition_id,
                                season_id,
                            )

    for team_id, team_name in sorted(teams.items()):
        write_row(
            "teams",
            {
                "team_id": team_id,
                "team_name": team_name,
                "data_version": config.data_version,
                "silver_schema_version": config.silver_schema_version,
            },
        )

    for player_id, player_name in sorted(players.items()):
        write_row(
            "players",
            {
                "player_id": player_id,
                "player_name": player_name,
                "data_version": config.data_version,
                "silver_schema_version": config.silver_schema_version,
            },
        )

    for writer in writers.values():
        writer.close()

    # Hard QA checks
    if row_counts["events"] == 0:
        qa_errors.append("events table row_count is zero")
    if len(event_ids_seen) != row_counts["events"]:
        qa_errors.append("duplicate or missing events detected in events table")

    for name, ids in subtype_ids.items():
        if len(ids) != row_counts[name]:
            qa_errors.append(f"duplicate subtype event_id detected in {name}")
        if not ids.issubset(event_ids_seen):
            qa_errors.append(f"orphan subtype event_id found in {name}")

    if not shot_ff_ids.issubset(shot_ids):
        qa_errors.append("shot_freeze_frame_players contains shot_event_id not present in shots")

    qa_summary = {
        "errors": qa_errors,
        "warnings_count": len(qa_warnings),
        "warnings_sample": qa_warnings[:50],
        "row_counts": dict(row_counts),
        "event_type_counts": dict(event_type_counts),
        "three_sixty_event_coverage": {
            "frames": row_counts["three_sixty_frames"],
            "players": row_counts["three_sixty_players"],
        },
    }

    if qa_errors:
        raise RuntimeError(f"Silver QA hard failures: {qa_errors[:10]}")

    ended_at = datetime.now(UTC).isoformat()
    manifest = _manifest_for_local_root(
        run_id=run_id,
        config=config,
        local_root=local_root,
        row_counts=dict(row_counts),
        qa_summary=qa_summary,
        warnings=qa_warnings,
        bronze_summary=bronze_summary,
        started_at=started_at,
        ended_at=ended_at,
    )

    manifest_path = local_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    success_path = local_root / "_SUCCESS"
    success_path.write_text("", encoding="utf-8")

    upload_status = []
    files_to_upload = [p for p in sorted(local_root.rglob("*")) if p.is_file()]

    parquet_files = [p for p in files_to_upload if p.suffix == ".parquet"]
    manifest_files = [p for p in files_to_upload if p.name == "manifest.json"]
    success_files = [p for p in files_to_upload if p.name == "_SUCCESS"]

    # Enforce publication atomicity: data first, manifest next, completion marker last.
    ordered_uploads = [*parquet_files, *manifest_files, *success_files]

    for local_file in ordered_uploads:
        rel = local_file.relative_to(local_root).as_posix()
        status = _upload_create_only(bucket, f"{output_prefix}/{rel}", local_file)
        upload_status.append((rel, status))

    artifacts_manifest_object = f"manifests/{run_id}.json"
    _upload_create_only(artifacts_bucket, artifacts_manifest_object, manifest_path)

    qa_report_path = local_root / "qa_summary.json"
    qa_report_path.write_text(json.dumps(qa_summary, indent=2), encoding="utf-8")
    _upload_create_only(artifacts_bucket, f"reports/{run_id}/qa_summary.json", qa_report_path)

    object_counts: dict[str, int] = {}
    bytes_by_table: dict[str, int] = {}
    for table in CONTRACTS:
        table_dir = local_root / table
        files = list(table_dir.rglob("*.parquet")) if table_dir.exists() else []
        object_counts[table] = len(files)
        bytes_by_table[table] = sum(f.stat().st_size for f in files)

    return SilverBuildResult(
        run_id=run_id,
        output_prefix=f"gs://{config.bucket_name}/{output_prefix}/",
        manifest_uri=f"gs://{config.bucket_name}/{output_prefix}/manifest.json",
        artifacts_manifest_uri=f"gs://{config.artifacts_bucket}/{artifacts_manifest_object}",
        row_counts=dict(row_counts),
        object_counts=object_counts,
        bytes_by_table=bytes_by_table,
        warnings=qa_warnings,
        bronze_summary=bronze_summary,
        qa_summary=qa_summary,
    )
