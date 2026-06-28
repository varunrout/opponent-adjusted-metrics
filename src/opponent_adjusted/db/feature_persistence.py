"""Persistence helpers for engineered feature tables."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from opponent_adjusted.db import session as db_session
from opponent_adjusted.db.models import ActionFeature
from opponent_adjusted.db.session import session_scope

logger = logging.getLogger(__name__)

BULK_CHUNK_SIZE = 50_000
DEFAULT_CXA_ACTION_FEATURE_VERSION = "cxa_action_features_v1"


def ensure_feature_tables() -> None:
    """Create feature tables when a local DB predates the Alembic migration."""

    ActionFeature.__table__.create(bind=db_session.engine, checkfirst=True)


def _finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _str_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _bulk_insert(session: Any, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), BULK_CHUNK_SIZE):
        session.bulk_insert_mappings(ActionFeature, rows[start : start + BULK_CHUNK_SIZE])


def _action_feature_rows(
    frame: pd.DataFrame,
    *,
    feature_family: str,
    version_tag: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        event_id = _str_or_none(record.get("event_id"))
        action_id = _str_or_none(record.get("action_id")) or (
            f"event-{event_id}" if event_id is not None else None
        )
        if action_id is None:
            continue
        rows.append(
            {
                "feature_family": feature_family,
                "version_tag": version_tag,
                "action_id": action_id,
                "event_id": event_id,
                "match_id": _int_or_none(record.get("match_id")),
                "team_id": _int_or_none(record.get("team_id")),
                "player_id": _int_or_none(record.get("player_id")),
                "possession_id": _str_or_none(record.get("possession_id")),
                "possession_number": _int_or_none(record.get("possession")),
                "sequence_id": _str_or_none(record.get("sequence_id")),
                "action_type": _str_or_none(record.get("action_type")),
                "start_x": _finite_float(record.get("start_x")),
                "start_y": _finite_float(record.get("start_y")),
                "end_x": _finite_float(record.get("end_x")),
                "end_y": _finite_float(record.get("end_y")),
                "length": _finite_float(record.get("length")),
                "angle": _finite_float(record.get("angle")),
                "x_progression": _finite_float(record.get("x_progression")),
                "y_progression": _finite_float(record.get("y_progression")),
                "distance_to_goal_before": _finite_float(record.get("distance_to_goal_before")),
                "distance_to_goal_after": _finite_float(record.get("distance_to_goal_after")),
                "angle_to_goal_before": _finite_float(record.get("angle_to_goal_before")),
                "angle_to_goal_after": _finite_float(record.get("angle_to_goal_after")),
                "start_zone": _str_or_none(record.get("start_zone")),
                "end_zone": _str_or_none(record.get("end_zone")),
                "is_progressive": _bool_or_none(record.get("is_progressive")),
                "enters_final_third": _bool_or_none(record.get("enters_final_third")),
                "enters_penalty_area": _bool_or_none(record.get("enters_penalty_area")),
                "target_shot_created": _bool_or_none(record.get("shot_created")),
                "target_created_shot_cxg": _finite_float(record.get("created_shot_cxg")),
                "target_created_shot_id": _str_or_none(record.get("created_shot_id")),
            }
        )
    return rows


def persist_cxa_action_features_to_database(
    features: pd.DataFrame,
    *,
    version_tag: str = DEFAULT_CXA_ACTION_FEATURE_VERSION,
    feature_family: str = "cxa",
) -> dict[str, int]:
    """Persist CxA action features idempotently to the configured database."""

    ensure_feature_tables()
    rows = _action_feature_rows(
        features,
        feature_family=feature_family,
        version_tag=version_tag,
    )
    with session_scope() as session:
        deleted = int(
            session.query(ActionFeature)
            .filter_by(feature_family=feature_family, version_tag=version_tag)
            .delete(synchronize_session=False)
        )
        _bulk_insert(session, rows)
        logger.info(
            "CxA action feature DB persistence: deleted=%d inserted=%d version=%s",
            deleted,
            len(rows),
            version_tag,
        )
    return {"deleted": deleted, "inserted": len(rows)}
