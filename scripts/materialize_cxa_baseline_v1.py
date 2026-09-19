"""CxA P_create model ladder, rungs 1-2: dumb baseline and XY baseline (model name
"v1"), for BOTH tracks (event-only, CxA+) -- mirrors CxG's ladder shape
(oam_ml.cxg_baseline_v1_metrics: model in {dumb_baseline, v1}) and table-naming
convention, split per-track into `oam_ml.cxa_event_baseline_v1_*` /
`oam_ml.cxa_plus_baseline_v1_*` (CxG names its tracks via a `track` column on a
single shared table; CxA instead prefixes the table name per track, matching how the
feature/split tables already prefix event vs plus -- kept consistent with the rest of
this project's naming rather than CxG's single-table convention).

Dumb baseline: the track's TRAIN split creation rate, predicted as a constant for
every row. Reported as log_loss only (no roc_auc -- a constant predictor has no rank
information, so AUC is undefined/degenerate, not just "bad").

XY baseline ("v1"): logistic regression on `start_x`, `start_y` ONLY, per
docs/cxa_split_policy_and_parallel_plan.md's baseline spec. This is deliberately NOT
the locked feature set (see materialize_cxa_candidate_v1.py for that) -- it exists so
the candidate models below have a comparison point.

Both fit on TRAIN only, evaluated on TRAIN and VALIDATION. Test is never queried (this
script's load helper only ever requests train/validation splits).

Writes (WRITE_TRUNCATE, full rebuild each run -- these are pure derived model-output
tables, same class as CxG's baseline tables):
  oam_ml.cxa_event_baseline_v1_metrics / _predictions
  oam_ml.cxa_plus_baseline_v1_metrics / _predictions
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cxa_modeling_common import (  # noqa: E402
    ML_DATASET,
    PROJECT,
    compute_metrics,
    encode_xy,
    load_track,
    write_table,
)

MATERIALIZED_AT = datetime.now(UTC).isoformat()


def fit_xy_model(train_df: pd.DataFrame) -> Logit:
    X = add_constant(encode_xy(train_df), has_constant="add")
    y = train_df["y_create"].astype(int)
    model = Logit(y, X).fit(disp=0)
    return model


def predict_xy(model: Logit, df: pd.DataFrame) -> np.ndarray:
    X = add_constant(encode_xy(df), has_constant="add")
    X = X[model.params.index]
    return model.predict(X).to_numpy()


def run_track(client: bigquery.Client, track: str) -> None:
    print(f"=== {track} ===")
    df = load_track(client, track)
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]

    dumb_rate = train["y_create"].mean()
    print(f"train n={len(train)}, validation n={len(validation)}, dumb baseline rate={dumb_rate:.5f}")

    xy_model = fit_xy_model(train)
    train_xy_pred = predict_xy(xy_model, train)
    val_xy_pred = predict_xy(xy_model, validation)

    metrics_rows = []
    predictions_rows = []
    for split_name, split_df, xy_pred in (
        ("train", train, train_xy_pred),
        ("validation", validation, val_xy_pred),
    ):
        y_true = split_df["y_create"].to_numpy()
        dumb_pred = np.full(len(split_df), dumb_rate)

        dumb_metrics = compute_metrics(y_true, dumb_pred, with_auc=False)
        xy_metrics = compute_metrics(y_true, xy_pred, with_auc=True)

        for model_name, m in (("dumb_baseline", dumb_metrics), ("v1", xy_metrics)):
            metrics_rows.append(
                {
                    "track": track,
                    "split": split_name,
                    "model": model_name,
                    "n": m["n"],
                    "log_loss": m["log_loss"],
                    "brier_score": m["brier_score"] if model_name == "v1" else None,
                    "roc_auc": m["roc_auc"],
                    "materialized_at": MATERIALIZED_AT,
                }
            )

        predictions_rows.append(
            pd.DataFrame(
                {
                    "track": track,
                    "pass_event_id": split_df["pass_event_id"].to_numpy(),
                    "split": split_name,
                    "dumb_baseline_prob": dumb_pred,
                    "v1_predicted_prob": xy_pred,
                    "y_create": y_true,
                    "materialized_at": MATERIALIZED_AT,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(predictions_rows, ignore_index=True)

    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_baseline_v1_metrics", metrics_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_baseline_v1_predictions", predictions_df)

    print(metrics_df.to_string(index=False))
    print()


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    for track in ("event", "plus"):
        run_track(client, track)


if __name__ == "__main__":
    main()
