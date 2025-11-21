"""Train a set-piece phase uplift model with shrinkage."""

from __future__ import annotations

import json
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)

BUCKET_COL = "set_piece_bucket"
BUCKET_FEATURE = "set_piece_encoder"
NUMERIC_FEATURES = ["statsbomb_xg"]
GROUP_COLS = [
    "set_piece_category",
    "set_piece_phase",
    "score_state",
    "minute_bucket_label",
]
MIN_SHOTS = 15


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["set_piece_category"].notna()].copy()
    if subset.empty:
        raise ValueError("No set-piece rows available for training.")
    for col in GROUP_COLS:
        subset[col] = subset[col].astype("object").fillna("Unknown")
    subset[BUCKET_COL] = subset[GROUP_COLS].agg("|".join, axis=1)
    subset = subset.dropna(subset=["statsbomb_xg", BUCKET_COL])
    return subset


def _build_pipeline() -> Pipeline:
    bucket_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers: List[tuple[str, object, List[str]]] = []
    if NUMERIC_FEATURES:
        transformers.append(("numeric", "passthrough", NUMERIC_FEATURES))
    transformers.append((BUCKET_FEATURE, bucket_encoder, [BUCKET_COL]))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("clf", model)])


def _indices_from_slice(idx: slice | np.ndarray) -> np.ndarray:
    if isinstance(idx, slice):
        start = 0 if idx.start is None else idx.start
        stop = start if idx.stop is None else idx.stop
        return np.arange(start, stop)
    return np.asarray(idx)


def _extract_bucket_map(pipeline: Pipeline) -> Dict[str, float]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    encoder: OneHotEncoder = preprocessor.named_transformers_[BUCKET_FEATURE]
    categories = encoder.categories_[0]
    idx = _indices_from_slice(preprocessor.output_indices_[BUCKET_FEATURE])
    coefs = clf.coef_[0][idx]
    return {str(cat): float(coef) for cat, coef in zip(categories, coefs, strict=False)}


def _summarize(df: pd.DataFrame, bucket_map: Dict[str, float]) -> pd.DataFrame:
    agg = (
        df.groupby(GROUP_COLS, observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
            modeled_prob=("modeled_prob", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["shots"] >= MIN_SHOTS]
    agg[BUCKET_COL] = agg[GROUP_COLS].agg("|".join, axis=1)
    agg["goal_rate"] = agg["goals"] / agg["shots"].clip(lower=1)
    agg["xg_per_shot"] = agg["xg_total"] / agg["shots"].clip(lower=1)
    agg["lift_vs_xg"] = agg["goal_rate"] - agg["xg_per_shot"]
    agg["lift_vs_model"] = agg["goal_rate"] - agg["modeled_prob"]
    agg["bucket_logit"] = agg[BUCKET_COL].map(bucket_map).fillna(0.0)
    agg["odds_multiplier"] = np.exp(agg["bucket_logit"])
    cols = GROUP_COLS + [
        "shots",
        "goals",
        "xg_total",
        "goal_rate",
        "xg_per_shot",
        "modeled_prob",
        "lift_vs_xg",
        "lift_vs_model",
        "bucket_logit",
        "odds_multiplier",
    ]
    return agg[cols].sort_values("bucket_logit", ascending=False)


def main() -> None:
    df = load_cxg_modeling_dataset()
    subset = _prepare(df)
    features = NUMERIC_FEATURES + [BUCKET_COL]
    X = subset[features]
    y = subset["is_goal"].astype(int)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)
    modeled_prob = pipeline.predict_proba(X)[:, 1]
    subset = subset.assign(modeled_prob=modeled_prob)

    metrics = {
        "samples": int(len(subset)),
        "log_loss": float(log_loss(y, modeled_prob)),
        "brier": float(brier_score_loss(y, modeled_prob)),
        "goal_rate": float(y.mean()),
        "mean_prediction": float(modeled_prob.mean()),
    }

    bucket_map = _extract_bucket_map(pipeline)
    summary = _summarize(subset, bucket_map)

    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    model_path = out_dir / "set_piece_phase_model.joblib"
    metrics_path = out_dir / "set_piece_phase_model_metrics.json"
    summary_path = out_dir / "set_piece_phase_modeled.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    summary.to_csv(summary_path, index=False)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved modeled summary to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
