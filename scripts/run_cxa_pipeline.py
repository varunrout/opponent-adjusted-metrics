#!/usr/bin/env python
"""Build baseline CxA action features from normalized event data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
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


def _third(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    if x < 40:
        return "defensive"
    if x < 80:
        return "middle"
    return "final"


def _zone(x: float | None, y: float | None) -> str:
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return "unknown"
    third = _third(float(x))
    if y < 26.67:
        lane = "left"
    elif y < 53.33:
        lane = "central"
    else:
        lane = "right"
    return f"{third}_{lane}"


def _distance_to_goal(x: float | None, y: float | None) -> float:
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return 0.0
    return float(np.hypot(settings.goal_center_x - float(x), settings.goal_center_y - float(y)))


def _angle_to_goal(x: float | None, y: float | None) -> float:
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return 0.0
    dx = settings.goal_center_x - float(x)
    dy = abs(settings.goal_center_y - float(y))
    return float(np.arctan2(dy, max(dx, 1e-9)))


def _seconds(minute: int | None, second: int | None) -> float:
    return float((minute or 0) * 60 + (second or 0))


def _score_state() -> str:
    return "drawing"


def _event_rows(session, competition_id: int | None = None) -> pd.DataFrame:
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
    stmt = stmt.where(Event.type.in_([*ELIGIBLE_ACTION_TYPES, "Shot"]))
    stmt = stmt.order_by(Event.match_id, Event.period, Event.minute, Event.second, Event.id)
    return pd.DataFrame([dict(row) for row in session.execute(stmt).mappings().all()])


def _detail_frame(session, model: Any) -> pd.DataFrame:
    objects = session.execute(select(model)).scalars().all()
    if not objects:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {column.name: getattr(obj, column.name) for column in model.__table__.columns}
            for obj in objects
        ]
    )


def _detail_maps(session) -> dict[str, pd.DataFrame]:
    details = {
        "Pass": _detail_frame(session, PassEvent),
        "Carry": _detail_frame(session, CarryEvent),
        "Dribble": _detail_frame(session, DribbleEvent),
        "Ball Receipt": _detail_frame(session, BallReceiptEvent),
    }
    return details


def _shot_rows(session) -> pd.DataFrame:
    stmt = select(
        Shot.id.label("shot_id"),
        Shot.event_id,
        Shot.statsbomb_xg,
        Shot.outcome.label("created_shot_outcome"),
    )
    return pd.DataFrame([dict(row) for row in session.execute(stmt).mappings().all()])


def _as_detail_lookup(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if df.empty or "event_id" not in df.columns:
        return {}
    return df.set_index("event_id").to_dict(orient="index")


def _enrich_action(row: pd.Series, detail: dict[str, Any]) -> dict[str, Any]:
    action_type = str(row["action_type"])
    start_x = row.get("start_x")
    start_y = row.get("start_y")
    end_x = detail.get("end_x", start_x)
    end_y = detail.get("end_y", start_y)
    if pd.isna(end_x):
        end_x = start_x
    if pd.isna(end_y):
        end_y = start_y

    length = detail.get("length")
    if length is None or pd.isna(length):
        length = float(
            np.hypot(
                float(end_x or 0) - float(start_x or 0), float(end_y or 0) - float(start_y or 0)
            )
        )
    angle = detail.get("angle")
    if angle is None or pd.isna(angle):
        angle = float(
            np.arctan2(
                float(end_y or 0) - float(start_y or 0), float(end_x or 0) - float(start_x or 0)
            )
        )

    x_progression = float(end_x or 0) - float(start_x or 0)
    y_progression = float(end_y or 0) - float(start_y or 0)
    return {
        "start_x": float(start_x or 0),
        "start_y": float(start_y or 0),
        "end_x": float(end_x or 0),
        "end_y": float(end_y or 0),
        "length": float(length or 0),
        "angle": float(angle or 0),
        "x_progression": x_progression,
        "y_progression": y_progression,
        "distance_to_goal_before": _distance_to_goal(start_x, start_y),
        "distance_to_goal_after": _distance_to_goal(end_x, end_y),
        "angle_to_goal_before": _angle_to_goal(start_x, start_y),
        "angle_to_goal_after": _angle_to_goal(end_x, end_y),
        "is_pass": action_type == "Pass",
        "is_carry": action_type == "Carry",
        "is_dribble": action_type == "Dribble",
        "is_cross": bool(detail.get("is_cross", False)),
        "is_cutback": str(detail.get("pass_type", "")).lower() == "cut back",
        "is_through_ball": bool(detail.get("is_through_ball", False)),
        "is_progressive": x_progression >= 10,
        "enters_final_third": _third(start_x) != "final" and _third(end_x) == "final",
        "enters_penalty_area": float(end_x or 0) >= 102 and 18 <= float(end_y or 0) <= 62,
        "enters_zone14": 80 <= float(end_x or 0) <= 102 and 26.67 <= float(end_y or 0) <= 53.33,
        "switches_play": abs(y_progression) >= 30,
        "play_pattern": "unknown",
        "body_part": detail.get("body_part") or "unknown",
        "pass_height": detail.get("pass_height") or "unknown",
        "start_zone": _zone(start_x, start_y),
        "end_zone": _zone(end_x, end_y),
        "start_third": _third(start_x),
        "end_third": _third(end_x),
        "score_state": _score_state(),
    }


def build_action_features(
    events: pd.DataFrame,
    shots: pd.DataFrame,
    detail_maps: dict[str, pd.DataFrame] | None = None,
    *,
    max_actions_to_shot: int = MAX_ACTIONS_TO_SHOT,
    max_seconds_to_shot: int = MAX_SECONDS_TO_SHOT,
) -> pd.DataFrame:
    """Build contract-aligned CxA action features from normalized events."""

    if events.empty:
        return pd.DataFrame()

    detail_maps = detail_maps or {}
    lookups = {name: _as_detail_lookup(frame) for name, frame in detail_maps.items()}
    shots_by_event = shots.set_index("event_id").to_dict(orient="index") if not shots.empty else {}

    ordered = events.copy()
    ordered["event_seconds"] = ordered.apply(
        lambda row: _seconds(row["minute"], row["second"]), axis=1
    )
    ordered = ordered.sort_values(
        ["match_id", "possession", "period", "minute", "second", "event_id"]
    )
    ordered["possession_index"] = ordered.groupby(["match_id", "possession"]).cumcount()

    action_rows: list[dict[str, Any]] = []
    eligible = ordered[ordered["action_type"].isin(ELIGIBLE_ACTION_TYPES)].copy()
    for _, action in eligible.iterrows():
        same_possession = ordered[
            (ordered["match_id"] == action["match_id"])
            & (ordered["team_id"] == action["team_id"])
            & (ordered["possession"] == action["possession"])
            & (ordered["possession_index"] > action["possession_index"])
            & ((ordered["possession_index"] - action["possession_index"]) <= max_actions_to_shot)
            & ((ordered["event_seconds"] - action["event_seconds"]) <= max_seconds_to_shot)
            & (ordered["action_type"] == "Shot")
            & (ordered["event_id"].isin(shots_by_event))
        ]
        created_shot = same_possession.head(1)
        created_shot_id = None
        created_shot_cxg = 0.0
        created_shot_distance = np.nan
        created_shot_angle = np.nan
        if not created_shot.empty:
            shot_event_id = int(created_shot.iloc[0]["event_id"])
            shot_info = shots_by_event.get(shot_event_id, {})
            created_shot_id = shot_info.get("shot_id")
            created_shot_cxg = float(shot_info.get("statsbomb_xg") or 0.0)
            created_shot_distance = _distance_to_goal(
                created_shot.iloc[0].get("start_x"), created_shot.iloc[0].get("start_y")
            )
            created_shot_angle = _angle_to_goal(
                created_shot.iloc[0].get("start_x"), created_shot.iloc[0].get("start_y")
            )

        detail = lookups.get(str(action["action_type"]), {}).get(int(action["event_id"]), {})
        feature_row = {
            "action_id": f"event-{int(action['event_id'])}",
            "event_id": int(action["event_id"]),
            "sequence_id": f"{int(action['match_id'])}-{int(action['possession'] or 0)}",
            "match_id": int(action["match_id"]),
            "possession": int(action["possession"] or 0),
            "team_id": int(action["team_id"]),
            "player_id": action["player_id"],
            "shot_created": int(not created_shot.empty),
            "created_shot_cxg": created_shot_cxg,
            "created_shot_id": created_shot_id,
            "created_shot_distance": created_shot_distance,
            "created_shot_angle": created_shot_angle,
            "action_type": str(action["action_type"]),
            "minute": int(action["minute"] or 0),
            "second": int(action["second"] or 0),
            "action_position": int(action["possession_index"]),
            "sequence_length_so_far": int(action["possession_index"] + 1),
            "seconds_since_possession_start": float(
                action["event_seconds"]
                - ordered[
                    (ordered["match_id"] == action["match_id"])
                    & (ordered["possession"] == action["possession"])
                ]["event_seconds"].min()
            ),
            "under_pressure": bool(action["under_pressure"]),
            "opponent_def_rating_global": np.nan,
            "opponent_zone_block_rate": np.nan,
            "nearest_defensive_action_seconds": np.nan,
            "teammate_receipt_pressure": np.nan,
            "prior_action_type": "unknown",
            "prior_action_success": np.nan,
            "carry_under_pressure": bool(
                action["under_pressure"] and action["action_type"] == "Carry"
            ),
            "set_piece_phase": "open_play",
        }
        feature_row.update(_enrich_action(action, detail))
        action_rows.append(feature_row)

    return pd.DataFrame(action_rows)


def build_action_features_from_database(competition_id: int | None = None) -> pd.DataFrame:
    """Build CxA action features from the configured database."""

    with get_session() as session:
        events = _event_rows(session, competition_id=competition_id)
        shots = _shot_rows(session)
        details = _detail_maps(session)
    return build_action_features(events, shots, details)


def save_action_features(df: pd.DataFrame, output_dir: Path = FEATURE_STORE_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ACTION_FEATURES_FILENAME
    df.to_parquet(path, index=False)
    return path


def run_pipeline(
    competition_id: int | None = None,
    output_dir: Path = FEATURE_STORE_DIR,
) -> dict[str, Path]:
    """Run the baseline CxA feature pipeline."""

    ensure_directories()
    features = build_action_features_from_database(competition_id=competition_id)
    output_path = save_action_features(features, output_dir)
    metadata_path = output_dir / "pipeline_metadata.json"
    metadata = {
        "pipeline": "cxa_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": competition_id,
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
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved CxA action features: %s rows -> %s", len(features), output_path)
    return {"action_features": output_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline CxA action feature pipeline")
    parser.add_argument("--competition-id", "-c", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=FEATURE_STORE_DIR)
    args = parser.parse_args()

    try:
        outputs = run_pipeline(competition_id=args.competition_id, output_dir=args.output_dir)
    except Exception as exc:
        logger.error("CxA pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
