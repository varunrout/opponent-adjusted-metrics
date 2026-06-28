#!/usr/bin/env python
"""Build baseline CxA action features from normalized event data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from opponent_adjusted.config import ensure_directories, settings
from opponent_adjusted.db.models import (
    BallReceiptEvent,
    CarryEvent,
    DribbleEvent,
    Event,
    Match,
    PassEvent,
    Shot,
)
from opponent_adjusted.db.feature_persistence import (
    DEFAULT_CXA_ACTION_FEATURE_VERSION,
    persist_cxa_action_features_to_database,
)
from opponent_adjusted.db.session import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ELIGIBLE_ACTION_TYPES = ("Pass", "Carry", "Dribble", "Ball Receipt")
FEATURE_STORE_DIR = settings.feature_store_path / "cxa"
ACTION_FEATURES_FILENAME = "action_features.parquet"
MAX_ACTIONS_TO_SHOT = 5
MAX_SECONDS_TO_SHOT = 15
SMOKE_MAX_MATCHES = 20
SQLITE_IN_CHUNK_SIZE = 900


def _elapsed(start: float) -> str:
    return f"{perf_counter() - start:.1f}s"


def _seconds(minute: pd.Series, second: pd.Series) -> pd.Series:
    return minute.fillna(0).astype(float) * 60 + second.fillna(0).astype(float)


def _distance_to_goal(x: pd.Series, y: pd.Series) -> pd.Series:
    return np.hypot(settings.goal_center_x - x.fillna(0), settings.goal_center_y - y.fillna(0))


def _angle_to_goal(x: pd.Series, y: pd.Series) -> pd.Series:
    dx = settings.goal_center_x - x.fillna(0)
    dy = (settings.goal_center_y - y.fillna(0)).abs()
    return np.arctan2(dy, dx.clip(lower=1e-9))


def _third_series(x: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [x.isna(), x < 40, x < 80],
            ["unknown", "defensive", "middle"],
            default="final",
        ),
        index=x.index,
    )


def _zone_series(x: pd.Series, y: pd.Series) -> pd.Series:
    thirds = _third_series(x)
    lanes = pd.Series(
        np.select(
            [x.isna() | y.isna(), y < 26.67, y < 53.33],
            ["unknown", "left", "central"],
            default="right",
        ),
        index=x.index,
    )
    return pd.Series(
        np.where((thirds == "unknown") | (lanes == "unknown"), "unknown", thirds + "_" + lanes),
        index=x.index,
    )


def _score_state() -> str:
    return "drawing"


def _limited_match_ids(session: Any, max_matches: int | None) -> list[int] | None:
    if max_matches is None:
        return None
    rows = (
        session.execute(
            select(Event.match_id).distinct().order_by(Event.match_id).limit(max_matches)
        )
        .scalars()
        .all()
    )
    return [int(row) for row in rows]


def _event_rows(
    session: Any,
    competition_id: int | None = None,
    max_matches: int | None = None,
) -> pd.DataFrame:
    match_ids = _limited_match_ids(session, max_matches)
    stmt = select(
        Event.id.label("event_id"),
        Event.raw_event_id,
        Event.match_id,
        Event.team_id,
        Event.player_id,
        Event.type.label("action_type"),
        Event.period,
        Event.minute,
        Event.second,
        Event.possession,
        Event.location_x.label("start_x"),
        Event.location_y.label("start_y"),
        Event.under_pressure,
        Event.outcome.label("event_outcome"),
    )
    if competition_id is not None:
        stmt = stmt.join(Match, Match.id == Event.match_id).where(
            Match.competition_id == competition_id
        )
    if match_ids is not None:
        stmt = stmt.where(Event.match_id.in_(match_ids))
    stmt = stmt.where(Event.type.in_([*ELIGIBLE_ACTION_TYPES, "Shot"]))
    stmt = stmt.order_by(Event.match_id, Event.period, Event.minute, Event.second, Event.id)
    return pd.DataFrame([dict(row) for row in session.execute(stmt).mappings().all()])


def _detail_frame(
    session: Any,
    model: Any,
    columns: tuple[Any, ...],
    event_ids: set[int] | None = None,
) -> pd.DataFrame:
    if event_ids is None:
        rows = session.execute(select(*columns)).mappings().all()
        return pd.DataFrame([dict(row) for row in rows])

    event_id_list = sorted(event_ids)
    frames: list[pd.DataFrame] = []
    for start in range(0, len(event_id_list), SQLITE_IN_CHUNK_SIZE):
        chunk = event_id_list[start : start + SQLITE_IN_CHUNK_SIZE]
        stmt = select(*columns).where(model.event_id.in_(chunk))
        rows = session.execute(stmt).mappings().all()
        if rows:
            frames.append(pd.DataFrame([dict(row) for row in rows]))
    if not frames:
        return pd.DataFrame(columns=[column.key for column in columns])
    return pd.concat(frames, ignore_index=True)


def _detail_maps(session: Any, event_ids: set[int] | None = None) -> dict[str, pd.DataFrame]:
    return {
        "Pass": _detail_frame(
            session,
            PassEvent,
            (
                PassEvent.event_id,
                PassEvent.length,
                PassEvent.angle,
                PassEvent.end_x,
                PassEvent.end_y,
                PassEvent.pass_height,
                PassEvent.pass_type,
                PassEvent.body_part,
                PassEvent.is_cross,
                PassEvent.is_through_ball,
            ),
            event_ids,
        ),
        "Carry": _detail_frame(
            session,
            CarryEvent,
            (
                CarryEvent.event_id,
                CarryEvent.end_x,
                CarryEvent.end_y,
                CarryEvent.length,
            ),
            event_ids,
        ),
        "Dribble": _detail_frame(session, DribbleEvent, (DribbleEvent.event_id,), event_ids),
        "Ball Receipt": _detail_frame(
            session,
            BallReceiptEvent,
            (BallReceiptEvent.event_id,),
            event_ids,
        ),
    }


def _shot_rows(session: Any, max_matches: int | None = None) -> pd.DataFrame:
    stmt = select(
        Shot.id.label("shot_id"),
        Shot.event_id,
        Shot.statsbomb_xg,
        Shot.outcome.label("created_shot_outcome"),
    )
    if max_matches is not None:
        match_ids = _limited_match_ids(session, max_matches)
        if match_ids is not None:
            stmt = stmt.where(Shot.match_id.in_(match_ids))
    return pd.DataFrame([dict(row) for row in session.execute(stmt).mappings().all()])


def _combined_details(detail_maps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for action_type, frame in detail_maps.items():
        if frame.empty:
            continue
        detail = frame.copy()
        detail["action_type"] = action_type
        frames.append(detail)
    if not frames:
        return pd.DataFrame(columns=["event_id", "action_type"])
    details = pd.concat(frames, ignore_index=True, sort=False)
    details = details.drop(columns=["start_x", "start_y"], errors="ignore")
    return details.drop_duplicates(["event_id", "action_type"])


def _target_frame(
    ordered: pd.DataFrame,
    shots: pd.DataFrame,
    max_actions_to_shot: int,
    max_seconds_to_shot: int,
    progress_every: int = 1000,
) -> pd.DataFrame:
    start = perf_counter()
    eligible_mask = ordered["action_type"].isin(ELIGIBLE_ACTION_TYPES)
    targets = ordered.loc[eligible_mask, ["event_id"]].copy()
    targets["created_shot_event_id"] = pd.NA
    targets["shot_created"] = 0

    if shots.empty:
        return targets

    group_cols = ["match_id", "team_id", "possession"]
    shot_event_ids = set(shots["event_id"].dropna().astype(int).tolist())
    shot_mask = (ordered["action_type"] == "Shot") & ordered["event_id"].isin(shot_event_ids)
    shot_groups = ordered.loc[shot_mask, group_cols].drop_duplicates()
    if shot_groups.empty:
        return targets

    scan_frame = ordered.loc[eligible_mask | shot_mask].merge(
        shot_groups,
        on=group_cols,
        how="inner",
    )
    grouped = scan_frame.groupby(group_cols, sort=False, dropna=False)
    group_count = len(grouped)
    logger.info(
        "CxA action features: scanning %d shot-containing match/team/possession groups",
        group_count,
    )

    processed = 0
    created_event_by_action: dict[int, int] = {}
    for _, group in grouped:
        processed += 1
        if processed % progress_every == 0:
            logger.info(
                "CxA action features: scanned %d/%d groups in %s",
                processed,
                group_count,
                _elapsed(start),
            )

        action_mask = group["action_type"].isin(ELIGIBLE_ACTION_TYPES).to_numpy()
        group_shot_mask = (group["action_type"] == "Shot") & group["event_id"].isin(shot_event_ids)
        if not action_mask.any() or not group_shot_mask.any():
            continue

        action_rows = group.loc[action_mask, ["event_id", "possession_index", "event_seconds"]]
        shot_rows = group.loc[group_shot_mask, ["event_id", "possession_index", "event_seconds"]]
        shot_positions = shot_rows["possession_index"].to_numpy()
        shot_seconds = shot_rows["event_seconds"].to_numpy()
        shot_events = shot_rows["event_id"].to_numpy()

        next_shot_index = np.searchsorted(
            shot_positions,
            action_rows["possession_index"].to_numpy() + 1,
        )
        valid = next_shot_index < len(shot_positions)
        if not valid.any():
            continue

        action_positions = action_rows["possession_index"].to_numpy()
        action_seconds = action_rows["event_seconds"].to_numpy()
        action_events = action_rows["event_id"].to_numpy()
        candidate_idx = next_shot_index[valid]
        valid_action_positions = action_positions[valid]
        valid_action_seconds = action_seconds[valid]
        window_ok = (
            shot_positions[candidate_idx] - valid_action_positions <= max_actions_to_shot
        ) & (shot_seconds[candidate_idx] - valid_action_seconds <= max_seconds_to_shot)
        if not window_ok.any():
            continue
        for action_event_id, shot_event_id in zip(
            action_events[valid][window_ok],
            shot_events[candidate_idx][window_ok],
            strict=False,
        ):
            created_event_by_action[int(action_event_id)] = int(shot_event_id)

    if created_event_by_action:
        targets["created_shot_event_id"] = targets["event_id"].map(created_event_by_action)
        targets["shot_created"] = targets["created_shot_event_id"].notna().astype(int)
    logger.info(
        "CxA action features: built target links for %d actions in %s",
        int(targets["shot_created"].sum()),
        _elapsed(start),
    )
    return targets


def _add_action_features(actions: pd.DataFrame) -> pd.DataFrame:
    actions = actions.copy()
    for column in ("end_x", "end_y"):
        if column not in actions.columns:
            actions[column] = np.nan
    actions["end_x"] = actions["end_x"].fillna(actions["start_x"])
    actions["end_y"] = actions["end_y"].fillna(actions["start_y"])

    dx = actions["end_x"].fillna(0) - actions["start_x"].fillna(0)
    dy = actions["end_y"].fillna(0) - actions["start_y"].fillna(0)
    if "length" not in actions.columns:
        actions["length"] = np.nan
    if "angle" not in actions.columns:
        actions["angle"] = np.nan
    actions["length"] = actions["length"].fillna(np.hypot(dx, dy)).fillna(0.0)
    actions["angle"] = actions["angle"].fillna(np.arctan2(dy, dx)).fillna(0.0)

    actions["start_x"] = actions["start_x"].fillna(0.0).astype(float)
    actions["start_y"] = actions["start_y"].fillna(0.0).astype(float)
    actions["end_x"] = actions["end_x"].fillna(0.0).astype(float)
    actions["end_y"] = actions["end_y"].fillna(0.0).astype(float)
    actions["x_progression"] = actions["end_x"] - actions["start_x"]
    actions["y_progression"] = actions["end_y"] - actions["start_y"]
    actions["distance_to_goal_before"] = _distance_to_goal(actions["start_x"], actions["start_y"])
    actions["distance_to_goal_after"] = _distance_to_goal(actions["end_x"], actions["end_y"])
    actions["angle_to_goal_before"] = _angle_to_goal(actions["start_x"], actions["start_y"])
    actions["angle_to_goal_after"] = _angle_to_goal(actions["end_x"], actions["end_y"])
    actions["is_pass"] = actions["action_type"].eq("Pass")
    actions["is_carry"] = actions["action_type"].eq("Carry")
    actions["is_dribble"] = actions["action_type"].eq("Dribble")
    is_cross = actions["is_cross"] if "is_cross" in actions.columns else False
    pass_type = (
        actions["pass_type"]
        if "pass_type" in actions.columns
        else pd.Series("", index=actions.index)
    )
    is_through_ball = actions["is_through_ball"] if "is_through_ball" in actions.columns else False
    actions["is_cross"] = is_cross
    actions["is_cutback"] = pass_type.fillna("").str.lower().eq("cut back")
    actions["is_through_ball"] = is_through_ball
    actions["is_progressive"] = actions["x_progression"] >= 10
    actions["start_third"] = _third_series(actions["start_x"])
    actions["end_third"] = _third_series(actions["end_x"])
    actions["enters_final_third"] = actions["start_third"].ne("final") & actions["end_third"].eq(
        "final"
    )
    actions["enters_penalty_area"] = actions["end_x"].ge(102) & actions["end_y"].between(18, 62)
    actions["enters_zone14"] = actions["end_x"].between(80, 102) & actions["end_y"].between(
        26.67, 53.33
    )
    actions["switches_play"] = actions["y_progression"].abs() >= 30
    actions["play_pattern"] = "unknown"
    actions["body_part"] = (
        actions["body_part"].fillna("unknown")
        if "body_part" in actions.columns
        else pd.Series("unknown", index=actions.index)
    )
    actions["pass_height"] = (
        actions["pass_height"].fillna("unknown")
        if "pass_height" in actions.columns
        else pd.Series("unknown", index=actions.index)
    )
    actions["start_zone"] = _zone_series(actions["start_x"], actions["start_y"])
    actions["end_zone"] = _zone_series(actions["end_x"], actions["end_y"])
    actions["score_state"] = _score_state()
    actions["carry_under_pressure"] = actions["under_pressure"] & actions["action_type"].eq("Carry")
    return actions


def build_action_features(
    events: pd.DataFrame,
    shots: pd.DataFrame,
    detail_maps: dict[str, pd.DataFrame] | None = None,
    *,
    max_actions_to_shot: int = MAX_ACTIONS_TO_SHOT,
    max_seconds_to_shot: int = MAX_SECONDS_TO_SHOT,
    max_actions: int | None = None,
) -> pd.DataFrame:
    """Build contract-aligned CxA action features from normalized events."""

    start = perf_counter()
    if events.empty:
        return pd.DataFrame()

    detail_maps = detail_maps or {}
    ordered = events.copy()
    ordered["event_seconds"] = _seconds(ordered["minute"], ordered["second"])
    ordered = ordered.sort_values(
        ["match_id", "possession", "period", "minute", "second", "event_id"]
    ).reset_index(drop=True)
    ordered["possession_index"] = ordered.groupby(
        ["match_id", "possession"], dropna=False
    ).cumcount()
    ordered["possession_start_seconds"] = ordered.groupby(["match_id", "possession"], dropna=False)[
        "event_seconds"
    ].transform("min")

    logger.info(
        "CxA action features: matches=%d possessions=%d events=%d",
        ordered["match_id"].nunique(),
        ordered[["match_id", "possession"]].drop_duplicates().shape[0],
        len(ordered),
    )
    targets = _target_frame(ordered, shots, max_actions_to_shot, max_seconds_to_shot)
    actions = ordered[ordered["action_type"].isin(ELIGIBLE_ACTION_TYPES)].copy()
    if max_actions is not None:
        actions = actions.head(max_actions).copy()
        targets = targets[targets["event_id"].isin(actions["event_id"])]
    logger.info("CxA action features: candidate actions=%d", len(actions))

    details = _combined_details(detail_maps)
    if not details.empty:
        actions = actions.merge(details, on=["event_id", "action_type"], how="left")
    actions = actions.merge(targets, on="event_id", how="left")
    if not shots.empty:
        shot_lookup = shots.rename(
            columns={
                "event_id": "created_shot_event_id",
                "shot_id": "created_shot_id",
                "statsbomb_xg": "created_shot_cxg",
            }
        )[["created_shot_event_id", "created_shot_id", "created_shot_cxg"]]
        actions = actions.merge(shot_lookup, on="created_shot_event_id", how="left")
    else:
        actions["created_shot_id"] = pd.NA
        actions["created_shot_cxg"] = 0.0

    actions["created_shot_cxg"] = actions["created_shot_cxg"].fillna(0.0)
    actions["created_shot_id"] = actions.get(
        "created_shot_id", pd.Series(pd.NA, index=actions.index)
    )
    actions = _add_action_features(actions)
    actions["created_shot_distance"] = np.nan
    actions["created_shot_angle"] = np.nan
    actions["action_id"] = "event-" + actions["event_id"].astype(int).astype(str)
    actions["sequence_id"] = (
        actions["match_id"].astype(int).astype(str)
        + "-"
        + actions["possession"].fillna(0).astype(int).astype(str)
    )
    actions["action_position"] = actions["possession_index"].astype(int)
    actions["sequence_length_so_far"] = actions["possession_index"].astype(int) + 1
    actions["seconds_since_possession_start"] = (
        actions["event_seconds"] - actions["possession_start_seconds"]
    )
    actions["opponent_def_rating_global"] = np.nan
    actions["opponent_zone_block_rate"] = np.nan
    actions["nearest_defensive_action_seconds"] = np.nan
    actions["teammate_receipt_pressure"] = np.nan
    actions["prior_action_type"] = "unknown"
    actions["prior_action_success"] = np.nan
    actions["set_piece_phase"] = "open_play"

    output_columns = [
        "action_id",
        "event_id",
        "sequence_id",
        "match_id",
        "possession",
        "team_id",
        "player_id",
        "shot_created",
        "created_shot_cxg",
        "created_shot_id",
        "created_shot_distance",
        "created_shot_angle",
        "action_type",
        "minute",
        "second",
        "action_position",
        "sequence_length_so_far",
        "seconds_since_possession_start",
        "under_pressure",
        "opponent_def_rating_global",
        "opponent_zone_block_rate",
        "nearest_defensive_action_seconds",
        "teammate_receipt_pressure",
        "prior_action_type",
        "prior_action_success",
        "carry_under_pressure",
        "set_piece_phase",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "length",
        "angle",
        "x_progression",
        "y_progression",
        "distance_to_goal_before",
        "distance_to_goal_after",
        "angle_to_goal_before",
        "angle_to_goal_after",
        "is_pass",
        "is_carry",
        "is_dribble",
        "is_cross",
        "is_cutback",
        "is_through_ball",
        "is_progressive",
        "enters_final_third",
        "enters_penalty_area",
        "enters_zone14",
        "switches_play",
        "play_pattern",
        "body_part",
        "pass_height",
        "start_zone",
        "end_zone",
        "start_third",
        "end_third",
        "score_state",
    ]
    result = actions.reindex(columns=output_columns)
    result["shot_created"] = result["shot_created"].fillna(0).astype(int)
    logger.info("CxA action features: built %d rows in %s", len(result), _elapsed(start))
    return result


def build_action_features_from_database(
    competition_id: int | None = None,
    max_matches: int | None = None,
    max_actions: int | None = None,
) -> pd.DataFrame:
    """Build CxA action features from the configured database."""

    start = perf_counter()
    logger.info("CxA pipeline: loading normalized events from database")
    with get_session() as session:
        events = _event_rows(session, competition_id=competition_id, max_matches=max_matches)
        logger.info("CxA pipeline: loaded %s eligible/shot event rows", len(events))
        logger.info(
            "CxA pipeline: loaded %s matches",
            events["match_id"].nunique() if not events.empty else 0,
        )
        logger.info(
            "CxA pipeline: loaded %s possessions/sequences",
            (
                events[["match_id", "possession"]].drop_duplicates().shape[0]
                if not events.empty
                else 0
            ),
        )
        shots = _shot_rows(session, max_matches=max_matches)
        logger.info("CxA pipeline: loaded %s shot detail rows", len(shots))
        event_ids = None
        if max_matches is not None and not events.empty:
            event_ids = set(events["event_id"].dropna().astype(int).tolist())
        details = _detail_maps(session, event_ids=event_ids)
        for action_type, detail_frame in details.items():
            logger.info("CxA pipeline: loaded %s %s detail rows", len(detail_frame), action_type)
    logger.info("CxA pipeline: loading complete in %s", _elapsed(start))
    logger.info("CxA pipeline: building action features")
    return build_action_features(events, shots, details, max_actions=max_actions)


def save_action_features(df: pd.DataFrame, output_dir: Path = FEATURE_STORE_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ACTION_FEATURES_FILENAME
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    logger.info("CxA pipeline: wrote %s action feature rows to %s", len(df), path)
    return path


def run_pipeline(
    competition_id: int | None = None,
    output_dir: Path = FEATURE_STORE_DIR,
    *,
    smoke: bool = False,
    max_matches: int | None = None,
    max_actions: int | None = None,
    persist_db: bool = False,
    feature_version: str = DEFAULT_CXA_ACTION_FEATURE_VERSION,
) -> dict[str, Path]:
    """Run the baseline CxA feature pipeline."""

    start = perf_counter()
    ensure_directories()
    if smoke:
        max_matches = max_matches or SMOKE_MAX_MATCHES
        if output_dir == FEATURE_STORE_DIR:
            output_dir = FEATURE_STORE_DIR / "smoke"
    logger.info("CxA pipeline: starting baseline feature build")
    logger.info(
        "CxA pipeline: competition_id=%s output_dir=%s smoke=%s max_matches=%s max_actions=%s",
        competition_id,
        output_dir,
        smoke,
        max_matches,
        max_actions,
    )
    features = build_action_features_from_database(
        competition_id=competition_id,
        max_matches=max_matches,
        max_actions=max_actions,
    )
    logger.info("CxA pipeline: built %s action feature rows", len(features))
    if not features.empty:
        logger.info(
            "CxA pipeline: action rows by type: %s",
            features["action_type"].value_counts(dropna=False).to_dict(),
        )
        logger.info(
            "CxA pipeline: shot-created rows=%s positive target sum=%.6f",
            int(features["shot_created"].sum()),
            float(features["created_shot_cxg"].sum()),
        )
    output_path = save_action_features(features, output_dir)
    db_persistence = {"enabled": False, "deleted": 0, "inserted": 0}
    if persist_db and not smoke:
        db_persistence = {
            "enabled": True,
            **persist_cxa_action_features_to_database(features, version_tag=feature_version),
        }
    elif persist_db and smoke:
        logger.info("CxA pipeline: skipping DB persistence for smoke feature build")
    metadata_path = output_dir / "pipeline_metadata.json"
    metadata = {
        "pipeline": "cxa_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": competition_id,
        "smoke": smoke,
        "max_matches": max_matches,
        "max_actions": max_actions,
        "target": "created_shot_cxg",
        "shot_creation_indicator": "shot_created",
        "attribution": {
            "same_team_only": True,
            "same_possession_preferred": True,
            "max_actions_to_shot": MAX_ACTIONS_TO_SHOT,
            "max_seconds_to_shot": MAX_SECONDS_TO_SHOT,
        },
        "files": {"action_features": str(output_path)},
        "row_counts": {"action_features": int(len(features))},
        "db_persistence": db_persistence,
        "elapsed_seconds": round(perf_counter() - start, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("CxA pipeline: wrote metadata to %s", metadata_path)
    logger.info("CxA pipeline: complete in %s", _elapsed(start))
    return {"action_features": output_path, "metadata": metadata_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline CxA action features")
    parser.add_argument("--competition-id", "-c", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=FEATURE_STORE_DIR)
    parser.add_argument("--smoke", action="store_true", help="Run a quick subset build")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument(
        "--no-db-persist",
        action="store_true",
        help="Write feature-store files only and skip action_features DB persistence",
    )
    parser.add_argument("--feature-version", default=DEFAULT_CXA_ACTION_FEATURE_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = run_pipeline(
            competition_id=args.competition_id,
            output_dir=args.output_dir,
            smoke=args.smoke,
            max_matches=args.max_matches,
            max_actions=args.max_actions,
            persist_db=not args.no_db_persist,
            feature_version=args.feature_version,
        )
    except Exception as exc:
        logger.error("CxA pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
