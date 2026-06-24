"""Model artifact loading and feature preparation for API inference."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from opponent_adjusted.api.schemas import ShotPredictionRequest
from opponent_adjusted.config import settings


class ModelArtifactError(RuntimeError):
    """Raised when an inference artifact cannot be loaded or used."""


def _default_cxg_artifact_path() -> Path:
    """Return the default CxG model artifact path.

    The path can be overridden with `CXG_MODEL_ARTIFACT`. The default matches the
    documented modelling output convention used elsewhere in the repository.
    """

    configured = os.getenv("CXG_MODEL_ARTIFACT")
    if configured:
        return Path(configured)
    return settings.model_artifacts_path / "cxg" / "models" / "contextual_model.joblib"


def _metadata_path(model_path: Path) -> Path:
    """Return the sidecar metadata path for a model artifact."""

    configured = os.getenv("CXG_MODEL_METADATA")
    if configured:
        return Path(configured)
    return model_path.with_suffix(".json")


@lru_cache(maxsize=1)
def load_cxg_artifact() -> tuple[Any, dict[str, Any], Path]:
    """Load the configured CxG artifact and optional metadata.

    Metadata is optional because older artifacts may not have a sidecar file yet.
    When metadata exists, the API uses its feature contract to order columns.
    """

    model_path = _default_cxg_artifact_path()
    if not model_path.exists():
        raise ModelArtifactError(
            f"CxG model artifact not found at {model_path}. "
            "Set CXG_MODEL_ARTIFACT or run the CxG training pipeline first."
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - depends on artifact format
        raise ModelArtifactError(f"Could not load CxG model artifact: {exc}") from exc

    metadata: dict[str, Any] = {}
    meta_path = _metadata_path(model_path)
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelArtifactError(f"CxG metadata JSON is invalid: {exc}") from exc

    return model, metadata, model_path


def build_cxg_feature_frame(request: ShotPredictionRequest, metadata: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row feature frame for CxG inference.

    The API starts from request-level fields and then honours a metadata feature
    contract when one is available. Missing metadata-backed features are filled
    with conservative neutral defaults so the endpoint can work with both older
    and newer artifacts while still making missing fields explicit.
    """

    row: dict[str, Any] = {
        "location_x": request.location_x,
        "location_y": request.location_y,
        "body_part": request.body_part,
        "technique": request.technique,
        "shot_type": request.shot_type,
        "first_time": request.first_time,
        "minute": request.minute,
        "score_diff": request.score_diff,
        "score_diff_at_shot": request.score_diff,
        "under_pressure": request.under_pressure,
        "opponent_team_id": request.opponent_team_id,
        "possession_duration": request.possession_duration or 0.0,
        "possession_length": request.possession_length or 0,
    }

    # Common geometry features used by several model variants.
    dx = settings.goal_center_x - request.location_x
    dy = settings.goal_center_y - request.location_y
    distance = (dx**2 + dy**2) ** 0.5
    row.update(
        {
            "distance_to_goal": distance,
            "centrality": abs(request.location_y - settings.goal_center_y),
        }
    )

    feature_spec = metadata.get("features", {})
    ordered_features: list[str] = []
    for key in ("numeric", "binary", "categorical"):
        values = feature_spec.get(key, [])
        if isinstance(values, list):
            ordered_features.extend(str(v) for v in values)

    if not ordered_features:
        ordered_features = list(row)

    neutral_context = settings.neutralization_context
    for feature in ordered_features:
        if feature not in row:
            row[feature] = neutral_context.get(feature, 0)

    return pd.DataFrame([{feature: row[feature] for feature in ordered_features}])


def predict_raw_probability(model: Any, features: pd.DataFrame) -> float:
    """Return the positive-class probability from a loaded model."""

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        return float(proba[0][1] if getattr(proba, "ndim", 1) == 2 else proba[0])

    if hasattr(model, "predict"):
        prediction = model.predict(features)
        return float(prediction[0])

    raise ModelArtifactError("CxG model artifact does not expose predict_proba or predict.")


def neutral_probability_from_metadata(raw_probability: float, metadata: dict[str, Any]) -> float:
    """Return a neutral probability fallback until full neutral scoring is wired.

    Newer CxG artifacts should eventually expose a dedicated neutralisation path.
    Until then, metadata can provide `neutral_probability_reference`; otherwise
    the API returns the raw probability as the neutral baseline and makes the
    opponent adjustment zero. This is safer than inventing an adjustment.
    """

    reference = metadata.get("neutral_probability_reference")
    if reference is None:
        return raw_probability
    return float(reference)
