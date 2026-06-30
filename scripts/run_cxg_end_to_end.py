#!/usr/bin/env python
"""Run CxG training, evaluation, scoring, export, and reporting end to end."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from opponent_adjusted.config import settings
from opponent_adjusted.db.models import (
    AggregatesPlayer,
    AggregatesTeam,
    Event,
    EvaluationMetric,
    ModelRegistry,
    Player,
    RawEvent,
    Shot,
    ShotPrediction,
    Team,
)
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.utils.logging import get_logger
from opponent_adjusted.modeling.cxg.contextual_model import (
    _build_pipeline,
    _filter_features,
    _prepare_frame,
)

DEFAULT_MODELING_ROOT = Path("outputs") / "modeling" / "cxg"
DEFAULT_OUTPUT_DIR = DEFAULT_MODELING_ROOT / "baseline"
LEGACY_OUTPUT_DIR = DEFAULT_MODELING_ROOT
MODEL_VERSION_PREFIX = "cxg_contextual"
logger = get_logger(__name__)


@dataclass(frozen=True)
class CxGRunOutputs:
    """Paths emitted by the end-to-end CxG run."""

    model_path: Path
    metadata_path: Path
    metrics_path: Path
    scored_predictions_path: Path
    player_aggregates_path: Path
    team_aggregates_path: Path
    model_card_path: Path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular input: {path}")


def discover_feature_input(feature_store_dir: Path | None = None) -> Path:
    """Find the richest available CxG feature output."""

    cxg_feature_store = feature_store_dir or settings.feature_store_path / "cxg"
    candidates = [
        cxg_feature_store / "shot_features.parquet",
        cxg_feature_store / "shot_features.csv",
        cxg_feature_store / "shots.parquet",
        cxg_feature_store / "shots.csv",
        DEFAULT_OUTPUT_DIR / "cxg_dataset_enriched.parquet",
        DEFAULT_OUTPUT_DIR / "cxg_dataset_filtered.parquet",
        DEFAULT_OUTPUT_DIR / "cxg_dataset.parquet",
        LEGACY_OUTPUT_DIR / "cxg_dataset_enriched.parquet",
        LEGACY_OUTPUT_DIR / "cxg_dataset_filtered.parquet",
        LEGACY_OUTPUT_DIR / "cxg_dataset.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No CxG feature input found. Run scripts/run_cxg_pipeline.py first or pass --input."
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "shot_x" in df.columns and "location_x" not in df.columns:
        rename_map["shot_x"] = "location_x"
    if "shot_y" in df.columns and "location_y" not in df.columns:
        rename_map["shot_y"] = "location_y"
    df = df.rename(columns=rename_map)

    if "is_goal" not in df.columns and "outcome" in df.columns:
        df["is_goal"] = (df["outcome"] == "Goal").astype(int)
    if "match_id" not in df.columns:
        df["match_id"] = np.arange(len(df)) // 20
    if "shot_id" not in df.columns:
        df["shot_id"] = np.arange(len(df))
    if "score_diff_at_shot" not in df.columns:
        df["score_diff_at_shot"] = 0
    if "minute_bucket_label" not in df.columns and "minute" in df.columns:
        df["minute_bucket_label"] = pd.cut(
            df["minute"].fillna(0),
            bins=[-1, 15, 30, 45, 60, 75, 200],
            labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        ).astype(str)
    if "score_state" not in df.columns:
        df["score_state"] = np.where(
            df["score_diff_at_shot"] > 0,
            "leading",
            np.where(df["score_diff_at_shot"] < 0, "trailing", "drawing"),
        )
    if "simple_state" not in df.columns:
        df["simple_state"] = df["score_state"]
    if "is_leading" not in df.columns:
        df["is_leading"] = df["score_diff_at_shot"] > 0
    if "is_trailing" not in df.columns:
        df["is_trailing"] = df["score_diff_at_shot"] < 0
    if "is_drawing" not in df.columns:
        df["is_drawing"] = df["score_diff_at_shot"] == 0
    defaults: dict[str, Any] = {
        "time_gap_seconds": 0.0,
        "possession_match": 0.0,
        "chain_label": "unknown",
        "pass_style": "unknown",
        "assist_category": "unknown",
        "pressure_state": "unknown",
        "set_piece_category": "open_play",
        "set_piece_phase": "none",
        "def_label": "average",
        "opponent_def_rating_global": 0.0,
        "opponent_def_zone_rating": 0.0,
        "opponent_zone_block_rate": 0.0,
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value
    for prior in (
        "finishing_bias_logit",
        "finishing_bias_multiplier",
        "concession_bias_logit",
        "concession_bias_multiplier",
        "set_piece_logit",
        "set_piece_multiplier",
        "set_piece_modeled_prob",
        "assist_quality_logit",
        "assist_quality_multiplier",
        "assist_quality_modeled_prob",
        "pressure_logit",
        "pressure_multiplier",
        "pressure_modeled_prob",
        "def_trigger_logit",
        "def_trigger_multiplier",
        "def_trigger_modeled_prob",
    ):
        if prior not in df.columns:
            df[prior] = 0.0
    return _prepare_frame(df)


def load_cxg_features(input_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load and normalize CxG feature data for modeling."""

    resolved = input_path or discover_feature_input()
    df = _normalise_columns(_read_table(resolved))
    required = {"is_goal", "match_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CxG input is missing required columns: {sorted(missing)}")
    return df, resolved


def _evaluation_splits(df: pd.DataFrame) -> int:
    class_counts = df["is_goal"].astype(int).value_counts()
    group_count = df["match_id"].nunique()
    return int(max(2, min(5, group_count, class_counts.min())))


def train_and_evaluate(
    df: pd.DataFrame,
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, list[str]]]:
    """Fit a deterministic sklearn CxG model and produce scored predictions."""

    df = df.dropna(subset=["is_goal", "match_id"]).copy()
    numeric, binary, categorical = _filter_features(df)
    feature_cols = numeric + binary + categorical
    if not feature_cols:
        raise ValueError("No supported CxG model features were found")
    if df["is_goal"].nunique() < 2:
        raise ValueError("CxG training data must contain both goals and non-goals")

    y = df["is_goal"].astype(int).to_numpy()
    n_splits = _evaluation_splits(df)
    groups = df["match_id"].to_numpy()
    if df["match_id"].nunique() >= n_splits:
        splitter = GroupKFold(n_splits=n_splits).split(df[feature_cols], y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            df[feature_cols], y
        )

    scored_parts = []
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(splitter, start=1):
        model = _build_pipeline(numeric, binary, categorical)
        model.fit(df.iloc[train_idx][feature_cols], y[train_idx])
        probs = model.predict_proba(df.iloc[test_idx][feature_cols])[:, 1]
        y_test = y[test_idx]
        metric_row: dict[str, Any] = {
            "fold": fold,
            "brier": float(brier_score_loss(y_test, probs)),
            "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
        }
        if len(np.unique(y_test)) == 2:
            metric_row["auc"] = float(roc_auc_score(y_test, probs))
        fold_metrics.append(metric_row)
        part = df.iloc[test_idx].copy()
        part["cxg_raw"] = probs
        scored_parts.append(part)

    final_model = _build_pipeline(numeric, binary, categorical)
    final_model.fit(df[feature_cols], y)
    neutral = df.copy()
    neutral["score_diff_at_shot"] = 0
    neutral["minute"] = 55
    neutral["minute_bucket_label"] = "46-60"
    neutral["is_leading"] = False
    neutral["is_trailing"] = False
    neutral["is_drawing"] = True
    neutral["score_state"] = "drawing"
    neutral["simple_state"] = "drawing"
    neutral["opponent_def_rating_global"] = 0.0
    neutral["opponent_def_zone_rating"] = 0.0
    neutral["opponent_zone_block_rate"] = 0.0

    scored = pd.concat(scored_parts).sort_index()
    scored["cxg_neutral"] = final_model.predict_proba(neutral.loc[scored.index, feature_cols])[:, 1]
    scored["cxg_opp_adjusted_diff"] = scored["cxg_raw"] - scored["cxg_neutral"]
    scored["cxg_opp_adjusted_ratio"] = scored["cxg_raw"] / scored["cxg_neutral"].replace(0, np.nan)

    metrics = {
        "brier_mean": float(np.mean([m["brier"] for m in fold_metrics])),
        "log_loss_mean": float(np.mean([m["log_loss"] for m in fold_metrics])),
        "auc_mean": float(np.nanmean([m.get("auc", np.nan) for m in fold_metrics])),
        "folds": fold_metrics,
        "n_rows": int(len(df)),
        "n_splits": n_splits,
    }
    features = {"numeric": numeric, "binary": binary, "categorical": categorical}
    return final_model, metrics, scored, features


def _aggregate(scored: pd.DataFrame, entity_id: str, entity_name: str) -> pd.DataFrame:
    if entity_id not in scored.columns:
        return pd.DataFrame()
    name_col = entity_name if entity_name in scored.columns else entity_id
    group_columns = list(dict.fromkeys([entity_id, name_col]))
    return (
        scored.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            shots_count=("shot_id", "count"),
            goals=("is_goal", "sum"),
            summed_cxg=("cxg_raw", "sum"),
            summed_neutral_cxg=("cxg_neutral", "sum"),
            summed_oppadj_diff=("cxg_opp_adjusted_diff", "sum"),
            avg_oppadj_diff=("cxg_opp_adjusted_diff", "mean"),
        )
        .sort_values(["summed_cxg", "shots_count"], ascending=False)
    )


def _db_identifier_lookup(shot_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not shot_ids:
        return {}

    with session_scope() as session:
        rows = (
            session.query(
                Shot.id,
                Shot.player_id,
                Shot.team_id,
                Shot.opponent_team_id,
                Event.team_id,
                RawEvent.raw_json,
            )
            .join(Event, Shot.event_id == Event.id)
            .join(RawEvent, Event.raw_event_id == RawEvent.id)
            .filter(Shot.id.in_(shot_ids))
            .all()
        )
        player_lookup = dict(session.query(Player.statsbomb_player_id, Player.id).all())
        player_name_lookup = dict(session.query(Player.id, Player.name).all())
        team_name_lookup = dict(session.query(Team.id, Team.name).all())

    lookup: dict[int, dict[str, Any]] = {}
    for (
        shot_id,
        shot_player_id,
        shot_team_id,
        shot_opponent_team_id,
        event_team_id,
        raw_json,
    ) in rows:
        raw_json = raw_json or {}
        raw_player = raw_json.get("player") or {}
        statsbomb_player_id = raw_player.get("id")
        player_id = shot_player_id or (
            player_lookup.get(statsbomb_player_id) if statsbomb_player_id is not None else None
        )
        team_id = shot_team_id or event_team_id
        lookup[int(shot_id)] = {
            "player_id": player_id,
            "player_name": (
                player_name_lookup.get(player_id)
                if player_id is not None
                else raw_player.get("name")
            ),
            "team_id": team_id,
            "team_name": team_name_lookup.get(team_id),
            "opponent_team_id": shot_opponent_team_id,
        }
    return lookup


def _enrich_scored_identifiers_from_database(scored: pd.DataFrame) -> pd.DataFrame:
    if "shot_id" not in scored.columns:
        return scored

    enriched = scored.copy()
    shot_ids = [int(value) for value in enriched["shot_id"].dropna().unique().tolist()]
    lookup = _db_identifier_lookup(shot_ids)
    if not lookup:
        return enriched

    for column in ("player_id", "player_name", "team_id", "team_name", "opponent_team_id"):
        values = enriched["shot_id"].map(lambda shot_id: lookup.get(int(shot_id), {}).get(column))
        if column not in enriched.columns:
            enriched[column] = values
        else:
            enriched[column] = enriched[column].where(enriched[column].notna(), values)
    return enriched


def _write_model_card(path: Path, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# CxG End-to-End Model Card",
                "",
                f"- Model version: `{metadata['model_version']}`",
                f"- Created at: `{metadata['created_at']}`",
                f"- Training rows: {metadata['trained_rows']}",
                f"- Artifact: `{metadata['artifact_path']}`",
                "",
                "## Intended use",
                "Reproducible contextual expected-goals scoring for project fixtures and StatsBomb-derived feature tables.",
                "",
                "## Evaluation",
                f"- Mean Brier score: {metrics['brier_mean']:.4f}",
                f"- Mean log loss: {metrics['log_loss_mean']:.4f}",
                f"- Mean ROC AUC: {metrics['auc_mean']:.4f}",
                "",
                "## Outputs",
                "The run exports raw CxG, neutral CxG, opponent-adjusted deltas, and player/team aggregates.",
                "",
                "## Limitations",
                "This is a pragmatic sklearn baseline using available event-derived context; it is not a production betting model and does not use tracking data.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _version_tag(model_version: str) -> str:
    """Fit model versions into legacy DB version_tag columns."""

    return model_version[:20]


def _finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _existing_ids(session: Any, model: Any, ids: pd.Series) -> set[int]:
    clean_ids = sorted({int(value) for value in ids.dropna().tolist()})
    if not clean_ids:
        return set()
    rows = session.query(model.id).filter(model.id.in_(clean_ids)).all()
    return {int(row[0]) for row in rows}


def _delete_existing_cxg_model_rows(session: Any, model_name: str, model_version: str) -> None:
    existing = (
        session.query(ModelRegistry)
        .filter_by(model_name=model_name, version=model_version)
        .one_or_none()
    )
    if existing is None:
        return

    session.query(ShotPrediction).filter_by(model_id=existing.id).delete(synchronize_session=False)
    session.query(AggregatesPlayer).filter_by(model_id=existing.id).delete(
        synchronize_session=False
    )
    session.query(AggregatesTeam).filter_by(model_id=existing.id).delete(synchronize_session=False)
    session.query(EvaluationMetric).filter_by(model_id=existing.id).delete(
        synchronize_session=False
    )
    session.delete(existing)
    session.flush()


def _metric_rows(metrics: dict[str, Any], model_id: int) -> list[EvaluationMetric]:
    rows: list[EvaluationMetric] = []
    for metric_name, value in metrics.items():
        if metric_name == "folds":
            continue
        metric_value = _finite_float(value)
        if metric_value is not None:
            rows.append(
                EvaluationMetric(
                    model_id=model_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                )
            )

    for fold in metrics.get("folds", []):
        fold_number = fold.get("fold")
        for metric_name, value in fold.items():
            if metric_name == "fold":
                continue
            metric_value = _finite_float(value)
            if metric_value is not None:
                rows.append(
                    EvaluationMetric(
                        model_id=model_id,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        slice_name=f"fold_{fold_number}",
                    )
                )
    return rows


def _prediction_rows(
    session: Any,
    scored: pd.DataFrame,
    model_id: int,
    version_tag: str,
) -> list[ShotPrediction]:
    if "shot_id" not in scored.columns:
        return []

    valid_shot_ids = _existing_ids(session, Shot, scored["shot_id"])
    rows: list[ShotPrediction] = []
    for record in scored.to_dict("records"):
        shot_id = record.get("shot_id")
        if pd.isna(shot_id) or int(shot_id) not in valid_shot_ids:
            continue
        rows.append(
            ShotPrediction(
                shot_id=int(shot_id),
                model_id=model_id,
                version_tag=version_tag,
                is_neutralized=False,
                raw_probability=float(record["cxg_raw"]),
                neutral_probability=_finite_float(record.get("cxg_neutral")),
                opponent_adjusted_diff=_finite_float(record.get("cxg_opp_adjusted_diff")),
                opponent_adjusted_ratio=_finite_float(record.get("cxg_opp_adjusted_ratio")),
            )
        )
    return rows


def _player_aggregate_rows(
    session: Any,
    player_aggregates: pd.DataFrame,
    model_id: int,
    version_tag: str,
) -> list[AggregatesPlayer]:
    if player_aggregates.empty or "player_id" not in player_aggregates.columns:
        return []

    valid_player_ids = _existing_ids(session, Player, player_aggregates["player_id"])
    rows: list[AggregatesPlayer] = []
    for record in player_aggregates.to_dict("records"):
        player_id = record.get("player_id")
        if pd.isna(player_id) or int(player_id) not in valid_player_ids:
            continue
        rows.append(
            AggregatesPlayer(
                player_id=int(player_id),
                model_id=model_id,
                version_tag=version_tag,
                shots_count=int(record.get("shots_count") or 0),
                summed_cxg=float(record.get("summed_cxg") or 0.0),
                summed_neutral_cxg=float(record.get("summed_neutral_cxg") or 0.0),
                summed_oppadj_diff=float(record.get("summed_oppadj_diff") or 0.0),
                avg_oppadj_diff=_finite_float(record.get("avg_oppadj_diff")),
            )
        )
    return rows


def _team_aggregate_rows(
    session: Any,
    team_aggregates: pd.DataFrame,
    model_id: int,
    version_tag: str,
) -> list[AggregatesTeam]:
    if team_aggregates.empty or "team_id" not in team_aggregates.columns:
        return []

    valid_team_ids = _existing_ids(session, Team, team_aggregates["team_id"])
    rows: list[AggregatesTeam] = []
    for record in team_aggregates.to_dict("records"):
        team_id = record.get("team_id")
        if pd.isna(team_id) or int(team_id) not in valid_team_ids:
            continue
        rows.append(
            AggregatesTeam(
                team_id=int(team_id),
                model_id=model_id,
                version_tag=version_tag,
                shots_count=int(record.get("shots_count") or 0),
                summed_cxg=float(record.get("summed_cxg") or 0.0),
                summed_neutral_cxg=float(record.get("summed_neutral_cxg") or 0.0),
                summed_oppadj_diff=float(record.get("summed_oppadj_diff") or 0.0),
                avg_oppadj_diff=_finite_float(record.get("avg_oppadj_diff")),
            )
        )
    return rows


def persist_cxg_outputs_to_database(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    scored: pd.DataFrame,
    player_aggregates: pd.DataFrame,
    team_aggregates: pd.DataFrame,
) -> dict[str, int]:
    """Persist file-backed CxG outputs into the existing modeling tables."""

    model_name = str(metadata.get("model_name") or "cxg")
    model_version = str(metadata["model_version"])
    version_tag = _version_tag(model_version)

    with session_scope() as session:
        _delete_existing_cxg_model_rows(session, model_name, model_version)

        registry = ModelRegistry(
            model_name=model_name,
            version=model_version,
            algorithm=str(metadata.get("model_type") or "contextual_logistic"),
            hyperparams=metadata.get("features"),
            trained_on_version_tag=version_tag,
            artifact_path=str(metadata["artifact_path"]),
            calibration_metrics=metrics,
        )
        session.add(registry)
        session.flush()
        logger.info(
            "CxG DB persistence: wrote model_registry id=%s version=%s",
            registry.id,
            model_version,
        )

        prediction_rows = _prediction_rows(session, scored, registry.id, version_tag)
        player_rows = _player_aggregate_rows(session, player_aggregates, registry.id, version_tag)
        team_rows = _team_aggregate_rows(session, team_aggregates, registry.id, version_tag)
        metric_rows = _metric_rows(metrics, registry.id)

        session.bulk_save_objects(prediction_rows)
        session.bulk_save_objects(player_rows)
        session.bulk_save_objects(team_rows)
        session.bulk_save_objects(metric_rows)

        logger.info("CxG DB persistence: inserted %d shot_predictions", len(prediction_rows))
        logger.info("CxG DB persistence: inserted %d player aggregates", len(player_rows))
        logger.info("CxG DB persistence: inserted %d team aggregates", len(team_rows))
        logger.info("CxG DB persistence: inserted %d evaluation metrics", len(metric_rows))

        return {
            "model_registry": 1,
            "shot_predictions": len(prediction_rows),
            "aggregates_player": len(player_rows),
            "aggregates_team": len(team_rows),
            "evaluation_metrics": len(metric_rows),
        }


def run_end_to_end(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_version: str | None = None,
    persist_db: bool = False,
) -> CxGRunOutputs:
    """Run the complete CxG modeling workflow and return emitted paths."""

    df, resolved_input = load_cxg_features(input_path)
    model, metrics, scored, features = train_and_evaluate(df)
    if persist_db:
        scored = _enrich_scored_identifiers_from_database(scored)
    created_at = datetime.now(timezone.utc).isoformat()
    version = model_version or f"{MODEL_VERSION_PREFIX}_{created_at[:10].replace('-', '')}"

    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    aggregates_dir = output_dir / "aggregates"
    for directory in (models_dir, reports_dir, predictions_dir, aggregates_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "contextual_model.joblib"
    metadata_path = model_path.with_suffix(".json")
    metrics_path = reports_dir / "metrics.json"
    scored_path = predictions_dir / "shot_predictions.parquet"
    player_path = aggregates_dir / "player_cxg.parquet"
    team_path = aggregates_dir / "team_cxg.parquet"
    card_path = reports_dir / "model_card.md"

    joblib.dump(model, model_path)
    scored.to_parquet(scored_path, index=False)
    player_aggregates = _aggregate(scored, "player_id", "player_name")
    team_aggregates = _aggregate(scored, "team_id", "team_name")
    player_aggregates.to_parquet(player_path, index=False)
    team_aggregates.to_parquet(team_path, index=False)

    metadata = {
        "model_name": "cxg",
        "model_version": version,
        "version": version,
        "model_type": "contextual_logistic",
        "target": "is_goal",
        "prediction_columns": {
            "cxg_raw": "Predicted goal probability in observed shot context.",
            "cxg_neutral": "Predicted goal probability after applying neutral context defaults.",
            "cxg_opp_adjusted_diff": "Observed-context CxG minus neutral-context CxG.",
            "cxg_opp_adjusted_ratio": "Observed-context CxG divided by neutral-context CxG.",
        },
        "artifact_path": str(model_path),
        "metadata_path": str(metadata_path),
        "created_at": created_at,
        "generated_at": created_at,
        "trained_at": created_at,
        "training_input_path": str(resolved_input),
        "trained_rows": int(len(df)),
        "features": features,
        "metrics": metrics,
        "outputs": {
            "scored_predictions": str(scored_path),
            "player_aggregates": str(player_path),
            "team_aggregates": str(team_path),
            "model_card": str(card_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_model_card(card_path, metadata, metrics)
    if persist_db:
        persist_cxg_outputs_to_database(
            metadata=metadata,
            metrics=metrics,
            scored=scored,
            player_aggregates=player_aggregates,
            team_aggregates=team_aggregates,
        )

    return CxGRunOutputs(
        model_path, metadata_path, metrics_path, scored_path, player_path, team_path, card_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CxG end-to-end training/evaluation/export")
    parser.add_argument(
        "--input", type=Path, default=None, help="Optional shot feature parquet/csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-version", default=None)
    parser.add_argument(
        "--no-db-persist",
        action="store_true",
        help="Write CxG files only and skip DB persistence.",
    )
    args = parser.parse_args()

    outputs = run_end_to_end(
        args.input,
        args.output_dir,
        args.model_version,
        persist_db=not args.no_db_persist,
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
