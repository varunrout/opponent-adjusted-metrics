"""CxG inference helpers for the FastAPI service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from opponent_adjusted.api.schemas import ShotPredictionRequest
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import ModelRegistry, OpponentDefProfile
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.features.cxg.geometry import calculate_all_geometry_features
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


class CxGModelNotAvailable(RuntimeError):
    """Raised when no usable CxG model artefact can be found."""


@dataclass(frozen=True)
class LoadedCxGModel:
    """Loaded CxG model and metadata."""

    model: Any
    metadata: dict[str, Any]
    model_path: Path
    version: str


DEFAULT_CONTEXTUAL_FEATURES = {
    "numeric": [
        "shot_distance",
        "shot_angle",
        "statsbomb_xg",
        "score_diff_at_shot",
        "minute",
        "time_gap_seconds",
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
    ],
    "binary": ["is_leading", "is_trailing", "is_drawing", "possession_match"],
    "categorical": [
        "chain_label",
        "pass_style",
        "score_state",
        "simple_state",
        "minute_bucket_label",
        "assist_category",
        "pressure_state",
        "set_piece_category",
        "set_piece_phase",
        "def_label",
    ],
}


def _safe_metadata_path(model_path: Path) -> Path | None:
    metadata_path = model_path.with_suffix(".json")
    return metadata_path if metadata_path.exists() else None


def _load_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = _safe_metadata_path(model_path)
    if not metadata_path:
        return {"features": DEFAULT_CONTEXTUAL_FEATURES, "model_type": "unknown"}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _candidate_model_paths(registry_path: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if registry_path:
        path = Path(registry_path)
        candidates.append(path if path.is_absolute() else Path.cwd() / path)

    candidates.extend(
        [
            settings.model_artifacts_path / "cxg" / "models" / "contextual_model.joblib",
            Path("outputs") / "modeling" / "cxg" / "models" / "contextual_model.joblib",
            Path("outputs")
            / "modeling"
            / "cxg"
            / "models"
            / "contextual_model_neutral_priors_refresh.joblib",
        ]
    )
    return candidates


def load_latest_cxg_model() -> LoadedCxGModel:
    """Load the latest available CxG model artefact.

    The function first checks the model registry. If no registry row exists, it
    falls back to common output locations used by the CxG training scripts.
    """

    registry_version = "unregistered"
    registry_artifact_path: str | None = None
    try:
        with session_scope() as session:
            model_row = (
                session.query(ModelRegistry)
                .filter(ModelRegistry.model_name == "cxg")
                .order_by(ModelRegistry.id.desc())
                .first()
            )
            if model_row:
                registry_version = model_row.version
                registry_artifact_path = model_row.artifact_path
    except Exception as exc:  # pragma: no cover - inference can still use local artefacts
        logger.warning("Could not read model registry: %s", exc)

    for model_path in _candidate_model_paths(registry_artifact_path):
        if not model_path.exists():
            continue
        try:
            model = joblib.load(model_path)
            metadata = _load_metadata(model_path)
            return LoadedCxGModel(
                model=model,
                metadata=metadata,
                model_path=model_path,
                version=registry_version,
            )
        except Exception as exc:
            logger.warning("Could not load CxG model from %s: %s", model_path, exc)

    raise CxGModelNotAvailable("No trained CxG model artefact was found")


def _minute_bucket(minute: int) -> str:
    if minute <= 15:
        return "0-15"
    if minute <= 30:
        return "16-30"
    if minute <= 45:
        return "31-45"
    if minute <= 60:
        return "46-60"
    if minute <= 75:
        return "61-75"
    return "76+"


def _score_state(score_diff: int) -> str:
    if score_diff > 0:
        return "leading"
    if score_diff < 0:
        return "trailing"
    return "drawing"


def _latest_opponent_profile(opponent_team_id: int) -> dict[str, float]:
    try:
        with session_scope() as session:
            profile = (
                session.query(OpponentDefProfile)
                .filter(OpponentDefProfile.team_id == opponent_team_id)
                .order_by(OpponentDefProfile.id.desc())
                .first()
            )
            if not profile:
                return {}
            return {
                "opponent_def_rating_global": float(profile.global_rating or 0.0),
                "opponent_def_zone_rating": float(profile.zone_rating or 0.0),
                "opponent_zone_block_rate": float(profile.block_rate or 0.0),
            }
    except Exception as exc:  # pragma: no cover - optional DB enrichment
        logger.warning("Could not load opponent profile for team_id=%s: %s", opponent_team_id, exc)
        return {}


def _base_feature_row(request: ShotPredictionRequest) -> dict[str, Any]:
    geometry = calculate_all_geometry_features(request.location_x, request.location_y)
    state = _score_state(request.score_diff)
    opponent_profile = _latest_opponent_profile(request.opponent_team_id)

    row: dict[str, Any] = {
        **geometry,
        **opponent_profile,
        "statsbomb_xg": None,
        "score_diff_at_shot": request.score_diff,
        "minute": request.minute,
        "minute_bucket_label": _minute_bucket(request.minute),
        "body_part": request.body_part,
        "technique": request.technique,
        "shot_type": request.shot_type,
        "first_time": request.first_time,
        "under_pressure": request.under_pressure,
        "is_leading": request.score_diff > 0,
        "is_trailing": request.score_diff < 0,
        "is_drawing": request.score_diff == 0,
        "score_state": state,
        "simple_state": state,
        "pressure_state": "under_pressure" if request.under_pressure else "no_pressure",
        "possession_match": float(request.possession_length or 0),
        "possession_duration": request.possession_duration,
        "possession_sequence_length": request.possession_length,
        "time_gap_seconds": 0.0,
        "chain_label": "api_request",
        "pass_style": "unknown",
        "assist_category": "unknown",
        "set_piece_category": (
            "open_play" if request.shot_type.lower() == "open play" else request.shot_type
        ),
        "set_piece_phase": "none",
        "def_label": "average",
        "is_blocked": False,
        "opponent_def_rating_global": opponent_profile.get("opponent_def_rating_global", 0.0),
        "opponent_def_zone_rating": opponent_profile.get("opponent_def_zone_rating", 0.0),
        "opponent_zone_block_rate": opponent_profile.get("opponent_zone_block_rate", 0.0),
    }

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
        row.setdefault(prior, 0.0)

    return row


def _model_feature_columns(metadata: dict[str, Any]) -> list[str]:
    feature_groups = metadata.get("features") or DEFAULT_CONTEXTUAL_FEATURES
    columns: list[str] = []
    for key in ("numeric", "binary", "categorical"):
        columns.extend(feature_groups.get(key, []))
    return columns


def _frame_for_model(row: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    columns = _model_feature_columns(metadata)
    if not columns:
        columns = list(row)
    return pd.DataFrame([{column: row.get(column) for column in columns}])


def _predict_probability(model: Any, frame: pd.DataFrame) -> float:
    probabilities = model.predict_proba(frame)
    if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
        return float(probabilities[:, 1][0])
    return float(probabilities.ravel()[0])


def predict_cxg(request: ShotPredictionRequest) -> dict[str, Any]:
    loaded = load_latest_cxg_model()

    raw_row = _base_feature_row(request)
    neutral_row = {
        **raw_row,
        "score_diff_at_shot": 0,
        "minute": 55,
        "minute_bucket_label": "46-60",
        "under_pressure": False,
        "pressure_state": "no_pressure",
        "is_leading": False,
        "is_trailing": False,
        "is_drawing": True,
        "score_state": "drawing",
        "simple_state": "drawing",
        "opponent_def_rating_global": 0.0,
        "opponent_def_zone_rating": 0.0,
        "opponent_zone_block_rate": 0.0,
    }

    raw_probability = _predict_probability(
        loaded.model,
        _frame_for_model(raw_row, loaded.metadata),
    )
    neutral_probability = _predict_probability(
        loaded.model,
        _frame_for_model(neutral_row, loaded.metadata),
    )
    opponent_adjusted_diff = raw_probability - neutral_probability
    opponent_adjusted_ratio = raw_probability / neutral_probability if neutral_probability else 0.0

    return {
        "raw_probability": raw_probability,
        "neutral_probability": neutral_probability,
        "opponent_adjusted_diff": opponent_adjusted_diff,
        "opponent_adjusted_ratio": opponent_adjusted_ratio,
        "model_version": loaded.version,
        "model_path": str(loaded.model_path),
    }
