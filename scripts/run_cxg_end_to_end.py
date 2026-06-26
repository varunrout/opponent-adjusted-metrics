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
from opponent_adjusted.modeling.cxg.contextual_model import (
    _build_pipeline,
    _filter_features,
    _prepare_frame,
)

DEFAULT_OUTPUT_DIR = Path("outputs") / "modeling" / "cxg"
MODEL_VERSION_PREFIX = "cxg_contextual"


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
    return (
        scored.groupby([entity_id, name_col], dropna=False)
        .agg(
            shots_count=("shot_id", "count"),
            goals=("is_goal", "sum"),
            summed_cxg=("cxg_raw", "sum"),
            summed_neutral_cxg=("cxg_neutral", "sum"),
            summed_oppadj_diff=("cxg_opp_adjusted_diff", "sum"),
            avg_oppadj_diff=("cxg_opp_adjusted_diff", "mean"),
        )
        .reset_index()
        .sort_values(["summed_cxg", "shots_count"], ascending=False)
    )


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


def run_end_to_end(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_version: str | None = None,
) -> CxGRunOutputs:
    """Run the complete CxG modeling workflow and return emitted paths."""

    df, resolved_input = load_cxg_features(input_path)
    model, metrics, scored, features = train_and_evaluate(df)
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
    _aggregate(scored, "player_id", "player_name").to_parquet(player_path, index=False)
    _aggregate(scored, "team_id", "team_name").to_parquet(team_path, index=False)

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
    args = parser.parse_args()

    outputs = run_end_to_end(args.input, args.output_dir, args.model_version)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
