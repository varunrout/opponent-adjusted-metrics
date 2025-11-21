"""Train a finishing/concession bias sub-model and emit modeled priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from opponent_adjusted.db.models import Team
from opponent_adjusted.db.session import SessionLocal
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)

CAT_FEATURE = "team_encoder"
OPP_FEATURE = "opp_encoder"


def _team_lookup() -> dict[int, str]:
    with SessionLocal() as session:
        rows = session.query(Team.id, Team.name).all()
    return {row.id: row.name for row in rows}


def _indices_from_slice(idx: slice | np.ndarray) -> np.ndarray:
    if isinstance(idx, slice):
        start = 0 if idx.start is None else idx.start
        stop = start if idx.stop is None else idx.stop
        return np.arange(start, stop)
    return np.asarray(idx)


def _build_pipeline() -> Pipeline:
    team_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    opp_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("geom", "passthrough", ["statsbomb_xg"]),
            (CAT_FEATURE, team_encoder, ["team_id"]),
            (OPP_FEATURE, opp_encoder, ["opponent_team_id"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("clf", model)])


def _extract_bias_map(
    pipeline: Pipeline,
    field: str,
) -> Dict[int, float]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    encoder: OneHotEncoder = preprocessor.named_transformers_[field]
    categories = encoder.categories_[0]
    idx = _indices_from_slice(preprocessor.output_indices_[field])
    coefs = clf.coef_[0][idx]
    return {int(cat): float(coef) for cat, coef in zip(categories, coefs, strict=False)}


def _summarize(
    df: pd.DataFrame,
    group_col: str,
    label: str,
    bias_map: dict[int, float],
    lookup: dict[int, str],
) -> pd.DataFrame:
    agg = (
        df.groupby(group_col, observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
        )
        .reset_index()
    )
    agg[label] = agg[group_col].map(lookup)
    agg["goal_rate"] = agg["goals"] / agg["shots"].clip(lower=1)
    agg["xg_per_shot"] = agg["xg_total"] / agg["shots"].clip(lower=1)
    agg["lift_vs_xg"] = agg["goal_rate"] - agg["xg_per_shot"]
    agg["bias_logit"] = agg[group_col].map(bias_map).fillna(0.0)
    agg["odds_multiplier"] = np.exp(agg["bias_logit"])
    cols = [
        label,
        group_col,
        "shots",
        "goals",
        "xg_total",
        "goal_rate",
        "xg_per_shot",
        "lift_vs_xg",
        "bias_logit",
        "odds_multiplier",
    ]
    return agg[cols].sort_values("bias_logit", ascending=False)


def main() -> None:
    df = load_cxg_modeling_dataset()
    features = ["statsbomb_xg", "team_id", "opponent_team_id"]
    X = df[features]
    y = df["is_goal"].astype(int)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "samples": int(len(df)),
        "log_loss": float(log_loss(y, preds)),
        "brier": float(brier_score_loss(y, preds)),
        "goal_rate": float(y.mean()),
        "mean_prediction": float(preds.mean()),
    }

    team_bias = _extract_bias_map(pipeline, CAT_FEATURE)
    opp_bias = _extract_bias_map(pipeline, OPP_FEATURE)
    lookup = _team_lookup()

    team_summary = _summarize(df, "team_id", "team_name", team_bias, lookup)
    opp_summary = _summarize(df, "opponent_team_id", "opponent_name", opp_bias, lookup)

    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "finishing_bias_model.joblib"
    metrics_path = out_dir / "finishing_bias_model_metrics.json"
    team_path = out_dir / "finishing_bias_modeled.csv"
    opp_path = out_dir / "concession_bias_modeled.csv"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    team_summary.to_csv(team_path, index=False)
    opp_summary.to_csv(opp_path, index=False)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved team priors to {team_path}")
    print(f"Saved opponent priors to {opp_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
