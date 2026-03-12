"""Score a prepared CxG dataset with saved models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from opponent_adjusted.modeling.modeling_utilities import get_modeling_output_dir

MODELS_DIR = get_modeling_output_dir("cxg", subdir="models")


@dataclass
class ModelConfig:
    label: str
    model_path: Path
    metadata_path: Path


def load_model_config(label: str) -> ModelConfig:
    model_path = MODELS_DIR / f"{label}.joblib"
    metadata_path = MODELS_DIR / f"{label}.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing model artifacts for label '{label}'")
    return ModelConfig(label=label, model_path=model_path, metadata_path=metadata_path)


def _load_feature_columns(metadata_path: Path) -> list[str]:
    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    features = metadata.get("features", {})
    if isinstance(features, dict):
        cols: list[str] = []
        for key in ("numeric", "binary", "categorical"):
            cols.extend(features.get(key, []))
        return cols
    return list(features)


def _prepare_features(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = df[list(columns)].copy()
    bool_cols = frame.select_dtypes(include=["bool", "boolean"]).columns
    if len(bool_cols) > 0:
        frame[bool_cols] = frame[bool_cols].astype(float)
    return frame


def score_dataset(
    dataset_path: Path,
    output_path: Path,
    model_label: str = "contextual_model",
) -> None:
    config = load_model_config(model_label)
    model = joblib.load(config.model_path)
    feature_columns = _load_feature_columns(config.metadata_path)

    df = pd.read_parquet(dataset_path)
    feature_frame = _prepare_features(df, feature_columns)
    preds = model.predict_proba(feature_frame)[:, 1]

    scored = df.copy()
    scored["cxg_prediction"] = preds
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        scored.to_parquet(output_path, index=False)
    else:
        scored.to_csv(output_path, index=False)
    print(f"Saved scored dataset to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Score a CxG dataset with a saved model")
    parser.add_argument("dataset", type=Path, help="Path to enriched dataset (parquet or csv)")
    parser.add_argument("output", type=Path, help="Path to save scored dataset")
    parser.add_argument(
        "--model-label",
        default="contextual_model",
        help="Model artifact label (e.g., contextual_model_filtered)",
    )
    args = parser.parse_args()
    score_dataset(args.dataset, args.output, args.model_label)
