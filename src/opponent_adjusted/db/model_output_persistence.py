"""Persistence helpers for file-backed modeling outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opponent_adjusted.db import session as db_session
from opponent_adjusted.db.models import (
    ActionPrediction,
    ActionThreatPrediction,
    AggregatesPlayer,
    AggregatesSequence,
    AggregatesTeam,
    EvaluationMetric,
    ModelRegistry,
    ShotPrediction,
)
from opponent_adjusted.db.session import session_scope

logger = logging.getLogger(__name__)

BULK_CHUNK_SIZE = 50_000


def ensure_model_output_tables() -> None:
    """Create new modeling output tables when a local DB predates the migration."""

    ActionPrediction.__table__.create(bind=db_session.engine, checkfirst=True)
    ActionThreatPrediction.__table__.create(bind=db_session.engine, checkfirst=True)
    AggregatesSequence.__table__.create(bind=db_session.engine, checkfirst=True)


def _version_tag(model_version: str) -> str:
    return model_version[:20]


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


def _str_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _bulk_insert(session: Any, model: Any, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), BULK_CHUNK_SIZE):
        session.bulk_insert_mappings(model, rows[start : start + BULK_CHUNK_SIZE])


def _metric_rows(metrics: dict[str, Any], model_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, value in metrics.items():
        if metric_name == "folds":
            continue
        metric_value = _finite_float(value)
        if metric_value is not None:
            rows.append(
                {
                    "model_id": model_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
            )

    for fold in metrics.get("folds", []):
        fold_number = fold.get("fold")
        for metric_name, value in fold.items():
            if metric_name == "fold":
                continue
            metric_value = _finite_float(value)
            if metric_value is not None:
                rows.append(
                    {
                        "model_id": model_id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "slice_name": f"fold_{fold_number}",
                    }
                )
    return rows


def _delete_existing_model_outputs(session: Any, model_name: str, model_version: str) -> None:
    existing = (
        session.query(ModelRegistry)
        .filter_by(model_name=model_name, version=model_version)
        .one_or_none()
    )
    if existing is None:
        return

    for model in (
        ShotPrediction,
        ActionPrediction,
        ActionThreatPrediction,
        AggregatesPlayer,
        AggregatesTeam,
        AggregatesSequence,
        EvaluationMetric,
    ):
        session.query(model).filter_by(model_id=existing.id).delete(synchronize_session=False)
    session.delete(existing)
    session.flush()


def _create_registry(
    session: Any,
    *,
    model_name: str,
    model_version: str,
    algorithm: str,
    artifact_path: Path | str,
    metrics: dict[str, Any],
    hyperparams: dict[str, Any] | None = None,
) -> ModelRegistry:
    registry = ModelRegistry(
        model_name=model_name,
        version=model_version,
        algorithm=algorithm,
        hyperparams=hyperparams,
        trained_on_version_tag=_version_tag(model_version),
        artifact_path=str(artifact_path),
        calibration_metrics=metrics,
    )
    session.add(registry)
    session.flush()
    return registry


def _aggregate_player_rows(
    frame: pd.DataFrame,
    *,
    model_id: int,
    version_tag: str,
    value_column: str,
    count_column: str,
) -> list[dict[str, Any]]:
    if frame.empty or "player_id" not in frame.columns:
        return []
    working = frame.copy()
    working["_entity_id"] = working["player_id"].map(_int_or_none)
    working["_total_value"] = working[value_column].fillna(0.0).astype(float)
    working["_count"] = working[count_column].fillna(0).astype(int)
    working = working.dropna(subset=["_entity_id"])
    if working.empty:
        return []
    collapsed = (
        working.groupby("_entity_id", dropna=False)
        .agg(
            shots_count=("_count", "sum"),
            summed_cxg=("_total_value", "sum"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for record in collapsed.to_dict("records"):
        player_id = int(record["_entity_id"])
        total_value = float(record["summed_cxg"])
        shots_count = int(record["shots_count"])
        rows.append(
            {
                "player_id": player_id,
                "model_id": model_id,
                "version_tag": version_tag,
                "shots_count": shots_count,
                "summed_cxg": total_value,
                "summed_neutral_cxg": total_value,
                "summed_oppadj_diff": 0.0,
                "avg_oppadj_diff": total_value / shots_count if shots_count else None,
            }
        )
    return rows


def _aggregate_team_rows(
    frame: pd.DataFrame,
    *,
    model_id: int,
    version_tag: str,
    value_column: str,
    count_column: str,
) -> list[dict[str, Any]]:
    if frame.empty or "team_id" not in frame.columns:
        return []
    working = frame.copy()
    working["_entity_id"] = working["team_id"].map(_int_or_none)
    working["_total_value"] = working[value_column].fillna(0.0).astype(float)
    working["_count"] = working[count_column].fillna(0).astype(int)
    working = working.dropna(subset=["_entity_id"])
    if working.empty:
        return []
    collapsed = (
        working.groupby("_entity_id", dropna=False)
        .agg(
            shots_count=("_count", "sum"),
            summed_cxg=("_total_value", "sum"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for record in collapsed.to_dict("records"):
        team_id = int(record["_entity_id"])
        total_value = float(record["summed_cxg"])
        shots_count = int(record["shots_count"])
        rows.append(
            {
                "team_id": team_id,
                "model_id": model_id,
                "version_tag": version_tag,
                "shots_count": shots_count,
                "summed_cxg": total_value,
                "summed_neutral_cxg": total_value,
                "summed_oppadj_diff": 0.0,
                "avg_oppadj_diff": total_value / shots_count if shots_count else None,
            }
        )
    return rows


def _sequence_rows(
    frame: pd.DataFrame,
    *,
    model_id: int,
    model_name: str,
    model_version: str,
    value_column: str,
    count_column: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    working = frame.copy()
    working["_sequence_id"] = working.apply(
        lambda row: _str_or_none(row.get("sequence_id") or row.get("possession_id")),
        axis=1,
    )
    working["_possession_id"] = working.apply(
        lambda row: _str_or_none(row.get("possession") or row.get("possession_id")),
        axis=1,
    )
    working["_match_id"] = working["match_id"].map(_int_or_none) if "match_id" in working else None
    working["_team_id"] = working["team_id"].map(_int_or_none) if "team_id" in working else None
    working["_total_value"] = working[value_column].fillna(0.0).astype(float)
    working["_action_count"] = working[count_column].fillna(0).astype(int)
    collapsed = (
        working.groupby(["_match_id", "_sequence_id"], dropna=False)
        .agg(
            team_id=("_team_id", "first"),
            possession_id=("_possession_id", "first"),
            total_value=("_total_value", "sum"),
            action_count=("_action_count", "sum"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for record in collapsed.to_dict("records"):
        rows.append(
            {
                "model_id": model_id,
                "model_family": model_name,
                "model_name": model_name,
                "model_version": model_version,
                "match_id": _int_or_none(record.get("_match_id")),
                "team_id": _int_or_none(record.get("team_id")),
                "possession_id": _str_or_none(record.get("possession_id")),
                "sequence_id": _str_or_none(record.get("_sequence_id")),
                "total_value": float(record.get("total_value") or 0.0),
                "action_count": int(record.get("action_count") or 0),
            }
        )
    return rows


def persist_cxa_outputs_to_database(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    scored: pd.DataFrame,
    player_aggregates: pd.DataFrame,
    team_aggregates: pd.DataFrame,
    sequence_aggregates: pd.DataFrame,
) -> dict[str, int]:
    """Persist CxA file outputs into model output tables."""

    ensure_model_output_tables()
    model_name = str(metadata.get("model_name") or "cxa")
    model_version = str(metadata["model_version"])
    version_tag = _version_tag(model_version)
    with session_scope() as session:
        _delete_existing_model_outputs(session, model_name, model_version)
        registry = _create_registry(
            session,
            model_name=model_name,
            model_version=model_version,
            algorithm=str(metadata.get("model_type") or "baseline_action_classifier"),
            artifact_path=metadata.get("artifact_path") or "",
            metrics=metrics,
            hyperparams=metadata.get("features"),
        )

        action_rows = [
            {
                "model_id": registry.id,
                "model_version": model_version,
                "action_id": _str_or_none(record.get("action_id")),
                "event_id": _str_or_none(record.get("event_id")),
                "match_id": _int_or_none(record.get("match_id")),
                "team_id": _int_or_none(record.get("team_id")),
                "player_id": _int_or_none(record.get("player_id")),
                "possession_id": _str_or_none(record.get("possession")),
                "sequence_id": _str_or_none(record.get("sequence_id")),
                "action_type": _str_or_none(record.get("action_type")),
                "predicted_cxa": _finite_float(record.get("predicted_cxa")),
                "predicted_value": _finite_float(record.get("cxa_value")),
                "target_value": _finite_float(record.get("created_shot_cxg")),
            }
            for record in scored.to_dict("records")
        ]
        player_rows = _aggregate_player_rows(
            player_aggregates,
            model_id=registry.id,
            version_tag=version_tag,
            value_column="total_cxa",
            count_column="action_count",
        )
        team_rows = _aggregate_team_rows(
            team_aggregates,
            model_id=registry.id,
            version_tag=version_tag,
            value_column="total_cxa",
            count_column="action_count",
        )
        sequence_rows = _sequence_rows(
            sequence_aggregates,
            model_id=registry.id,
            model_name=model_name,
            model_version=model_version,
            value_column="total_cxa",
            count_column="action_count",
        )
        metric_rows = _metric_rows(metrics, registry.id)

        _bulk_insert(session, ActionPrediction, action_rows)
        _bulk_insert(session, AggregatesPlayer, player_rows)
        _bulk_insert(session, AggregatesTeam, team_rows)
        _bulk_insert(session, AggregatesSequence, sequence_rows)
        _bulk_insert(session, EvaluationMetric, metric_rows)

        logger.info("CxA DB persistence: inserted %d action_predictions", len(action_rows))
        logger.info("CxA DB persistence: inserted %d player aggregates", len(player_rows))
        logger.info("CxA DB persistence: inserted %d team aggregates", len(team_rows))
        logger.info("CxA DB persistence: inserted %d sequence aggregates", len(sequence_rows))
        logger.info("CxA DB persistence: inserted %d evaluation metrics", len(metric_rows))
        return {
            "model_registry": 1,
            "action_predictions": len(action_rows),
            "aggregates_player": len(player_rows),
            "aggregates_team": len(team_rows),
            "aggregates_sequence": len(sequence_rows),
            "evaluation_metrics": len(metric_rows),
        }


def persist_cxt_outputs_to_database(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    player_aggregates: pd.DataFrame,
    team_aggregates: pd.DataFrame,
    sequence_aggregates: pd.DataFrame,
) -> dict[str, int]:
    """Persist CxT file outputs into model output tables."""

    ensure_model_output_tables()
    model_name = str(metadata.get("model_name") or "cxt")
    model_version = str(metadata["model_version"])
    version_tag = _version_tag(model_version)
    with session_scope() as session:
        _delete_existing_model_outputs(session, model_name, model_version)
        registry = _create_registry(
            session,
            model_name=model_name,
            model_version=model_version,
            algorithm=str(metadata.get("model_type") or "baseline_grid_threat"),
            artifact_path=metadata.get("artifact_path") or "",
            metrics=metrics,
            hyperparams=metadata.get("features"),
        )

        threat_rows = [
            {
                "model_id": registry.id,
                "model_version": model_version,
                "action_id": _str_or_none(record.get("action_id")),
                "event_id": _str_or_none(record.get("event_id")),
                "match_id": _int_or_none(record.get("match_id")),
                "team_id": _int_or_none(record.get("team_id")),
                "player_id": _int_or_none(record.get("player_id")),
                "possession_id": _str_or_none(record.get("possession_id")),
                "sequence_id": _str_or_none(
                    record.get("sequence_id") or record.get("possession_id")
                ),
                "action_type": _str_or_none(record.get("action_type")),
                "start_zone": _str_or_none(record.get("start_zone")),
                "end_zone": _str_or_none(record.get("end_zone")),
                "cxt_value": _finite_float(record.get("cxt_value")),
                "predicted_threat": _finite_float(record.get("end_threat")),
                "threat_delta": _finite_float(record.get("cxt_value")),
            }
            for record in predictions.to_dict("records")
        ]
        player_rows = _aggregate_player_rows(
            player_aggregates,
            model_id=registry.id,
            version_tag=version_tag,
            value_column="total_cxt",
            count_column="actions",
        )
        team_rows = _aggregate_team_rows(
            team_aggregates,
            model_id=registry.id,
            version_tag=version_tag,
            value_column="total_cxt",
            count_column="actions",
        )
        sequence_rows = _sequence_rows(
            sequence_aggregates,
            model_id=registry.id,
            model_name=model_name,
            model_version=model_version,
            value_column="total_cxt",
            count_column="action_count",
        )
        metric_rows = _metric_rows(metrics, registry.id)

        _bulk_insert(session, ActionThreatPrediction, threat_rows)
        _bulk_insert(session, AggregatesPlayer, player_rows)
        _bulk_insert(session, AggregatesTeam, team_rows)
        _bulk_insert(session, AggregatesSequence, sequence_rows)
        _bulk_insert(session, EvaluationMetric, metric_rows)

        logger.info("CxT DB persistence: inserted %d action_threat_predictions", len(threat_rows))
        logger.info("CxT DB persistence: inserted %d player aggregates", len(player_rows))
        logger.info("CxT DB persistence: inserted %d team aggregates", len(team_rows))
        logger.info("CxT DB persistence: inserted %d sequence aggregates", len(sequence_rows))
        logger.info("CxT DB persistence: inserted %d evaluation metrics", len(metric_rows))
        return {
            "model_registry": 1,
            "action_threat_predictions": len(threat_rows),
            "aggregates_player": len(player_rows),
            "aggregates_team": len(team_rows),
            "aggregates_sequence": len(sequence_rows),
            "evaluation_metrics": len(metric_rows),
        }
