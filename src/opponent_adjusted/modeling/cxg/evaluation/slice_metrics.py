"""Compute slice-based evaluation metrics for CxG models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from opponent_adjusted.modeling.cxg.contextual_model import DATASET_FILE_MAP
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_dataset_from_versions,
)

MODELS_DIR = get_modeling_output_dir("cxg", subdir="models")
EVAL_DIR = get_modeling_output_dir("cxg", subdir="evaluation")
MIN_SLICE_SIZE = 100
N_BINS_ECE = 10


@dataclass
class ModelSpec:
    label: str
    display_name: str
    kind: str  # "provider" | "joblib"
    dataset_key: str
    model_filename: str | None = None
    metadata_filename: str | None = None

    @property
    def model_path(self) -> Path | None:
        if not self.model_filename:
            return None
        return MODELS_DIR / self.model_filename

    @property
    def metadata_path(self) -> Path | None:
        if not self.metadata_filename:
            return None
        return MODELS_DIR / self.metadata_filename


@dataclass
class SliceSpec:
    name: str
    description: str
    filter_func: Callable[[pd.DataFrame], pd.Series]


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        label="provider_xg",
        display_name="StatsBomb Provider xG",
        kind="provider",
        dataset_key="enriched",
    ),
    ModelSpec(
        label="baseline_geometry",
        display_name="Baseline Geometry",
        kind="joblib",
        dataset_key="filtered",
        model_filename="baseline_geometry_model.joblib",
        metadata_filename="baseline_geometry_model.json",
    ),
    ModelSpec(
        label="contextual_filtered",
        display_name="Contextual (Filtered)",
        kind="joblib",
        dataset_key="filtered",
        model_filename="contextual_model_filtered.joblib",
        metadata_filename="contextual_model_filtered.json",
    ),
    ModelSpec(
        label="contextual_enriched",
        display_name="Contextual (Enriched Priors)",
        kind="joblib",
        dataset_key="enriched",
        model_filename="contextual_model.joblib",
        metadata_filename="contextual_model.json",
    ),
    ModelSpec(
        label="contextual_raw",
        display_name="Contextual (Raw)",
        kind="joblib",
        dataset_key="raw",
        model_filename="contextual_model_raw.joblib",
        metadata_filename="contextual_model_raw.json",
    ),
]


def _slice_all(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _slice_under_pressure(df: pd.DataFrame) -> pd.Series:
    state = df["pressure_state"].astype(str).str.lower()
    return state.eq("under pressure")


def _slice_not_under_pressure(df: pd.DataFrame) -> pd.Series:
    state = df["pressure_state"].astype(str).str.lower()
    return state.eq("not under pressure")


def _slice_leading(df: pd.DataFrame) -> pd.Series:
    return df["score_diff_at_shot"].fillna(0) > 0


def _slice_trailing(df: pd.DataFrame) -> pd.Series:
    return df["score_diff_at_shot"].fillna(0) < 0


def _slice_drawing(df: pd.DataFrame) -> pd.Series:
    return df["score_diff_at_shot"].fillna(0) == 0


def _slice_first_half(df: pd.DataFrame) -> pd.Series:
    return df["minute"].fillna(0) < 46


def _slice_second_half(df: pd.DataFrame) -> pd.Series:
    return df["minute"].fillna(0) >= 46


def _slice_close_range(df: pd.DataFrame) -> pd.Series:
    return df["shot_distance"].fillna(np.inf) <= 12.0


def _slice_long_range(df: pd.DataFrame) -> pd.Series:
    return df["shot_distance"].fillna(0) >= 20.0


def _slice_set_piece(df: pd.DataFrame) -> pd.Series:
    return df["set_piece_category"].astype(str).str.lower() != "open play"


def _slice_open_play(df: pd.DataFrame) -> pd.Series:
    return ~_slice_set_piece(df)


SLICE_SPECS: list[SliceSpec] = [
    SliceSpec("overall", "All shots", _slice_all),
    SliceSpec("under_pressure", "Shots under pressure", _slice_under_pressure),
    SliceSpec("not_under_pressure", "Shots not under pressure", _slice_not_under_pressure),
    SliceSpec("leading", "Team leading", _slice_leading),
    SliceSpec("trailing", "Team trailing", _slice_trailing),
    SliceSpec("drawing", "Score level", _slice_drawing),
    SliceSpec("first_half", "Minutes 0-45", _slice_first_half),
    SliceSpec("second_half", "Minutes 46+", _slice_second_half),
    SliceSpec("close_range", "Shot distance ≤ 12m", _slice_close_range),
    SliceSpec("long_range", "Shot distance ≥ 20m", _slice_long_range),
    SliceSpec("set_piece", "Set piece phases", _slice_set_piece),
    SliceSpec("open_play", "Open play shots", _slice_open_play),
]


@lru_cache(maxsize=None)
def _load_dataset(dataset_key: str) -> pd.DataFrame:
    if dataset_key not in DATASET_FILE_MAP:
        raise ValueError(f"Unknown dataset version '{dataset_key}'")
    return load_dataset_from_versions("cxg", DATASET_FILE_MAP[dataset_key]).copy()


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _collect_feature_columns(metadata: dict) -> list[str]:
    features = metadata.get("features", [])
    if isinstance(features, dict):
        cols: list[str] = []
        for key in ("numeric", "binary", "categorical"):
            cols.extend(features.get(key, []))
        return cols
    return list(features)


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df


def _predict_provider(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "statsbomb_xg" not in df.columns:
        raise KeyError("statsbomb_xg column missing for provider evaluation")
    df = df.dropna(subset=["is_goal", "statsbomb_xg"]).copy()
    preds = df["statsbomb_xg"].astype(float).clip(0.0, 1.0).to_numpy()
    return df, preds


def _predict_joblib(df: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, np.ndarray]:
    if not spec.model_path or not spec.metadata_path:
        raise ValueError(f"Model paths missing for spec {spec.label}")
    metadata = _load_metadata(spec.metadata_path)
    feature_columns = _collect_feature_columns(metadata)
    df = df.dropna(subset=["is_goal"]).copy()
    df = _ensure_required_columns(df, feature_columns)
    feature_frame = df[feature_columns].copy()

    bool_cols = feature_frame.select_dtypes(include=["bool", "boolean"]).columns
    if len(bool_cols) > 0:
        feature_frame[bool_cols] = feature_frame[bool_cols].astype(float)

    if metadata.get("model_type") == "baseline_geometry":
        feature_frame = feature_frame.dropna(subset=feature_columns)
        df = df.loc[feature_frame.index]

    import joblib  # local import to avoid mandatory dependency if unused

    model = joblib.load(spec.model_path)
    preds = model.predict_proba(feature_frame)[:, 1]
    return df, preds


def _ece_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, N_BINS_ECE + 1)
    bin_ids = np.digitize(y_pred, bins) - 1
    total = len(y_true)
    ece = 0.0
    for b in range(N_BINS_ECE):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        avg_pred = float(y_pred[mask].mean())
        avg_true = float(y_true[mask].mean())
        weight = mask.sum() / total
        ece += weight * abs(avg_true - avg_pred)
    return float(ece)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "count": int(len(y_true)),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred)),
        "auc": float("nan"),
        "ece": float(_ece_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred))
    return metrics


def _evaluate_spec(spec: ModelSpec) -> pd.DataFrame:
    df = _load_dataset(spec.dataset_key)
    if spec.kind == "provider":
        eval_df, preds = _predict_provider(df)
    else:
        eval_df, preds = _predict_joblib(df, spec)

    eval_df = eval_df.copy()
    eval_df["prediction"] = preds
    eval_df["is_goal"] = eval_df["is_goal"].astype(int)

    records: list[dict] = []
    for slice_spec in SLICE_SPECS:
        mask = slice_spec.filter_func(eval_df)
        subset = eval_df[mask]
        if len(subset) < MIN_SLICE_SIZE:
            continue
        y_true = subset["is_goal"].to_numpy()
        y_pred = subset["prediction"].to_numpy()
        metrics = _compute_metrics(y_true, y_pred)
        metrics.update(
            {
                "model": spec.label,
                "model_display": spec.display_name,
                "slice": slice_spec.name,
                "slice_description": slice_spec.description,
                "dataset_key": spec.dataset_key,
            }
        )
        records.append(metrics)
    return pd.DataFrame(records)


def main() -> None:
    all_frames: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        try:
            frame = _evaluate_spec(spec)
        except FileNotFoundError as exc:  # pragma: no cover - runtime guard
            print(f"Skipping {spec.label}: {exc}")
            continue
        all_frames.append(frame)

    if not all_frames:
        raise RuntimeError("No evaluation results generated")

    result_df = pd.concat(all_frames, ignore_index=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = EVAL_DIR / "slice_metrics.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"Saved slice metrics to {output_csv}")

    summary = (
        result_df[result_df["slice"] == "overall"][
            ["model", "model_display", "brier", "log_loss", "auc", "ece", "count"]
        ]
        .sort_values("brier")
        .to_dict(orient="records")
    )
    with (EVAL_DIR / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Overall summary written to {EVAL_DIR / 'summary.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
