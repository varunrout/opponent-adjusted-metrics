"""CxA P_create model ladder, rung 3: candidate models on the full LOCKED feature set,
for BOTH tracks -- two separate model families, deliberately NOT named "v1"/"v3" (CxG's
own names) or anything implying a frozen choice:

  - "v_candidate_logistic": statsmodels Logit (same fitting approach as CxG's v3, for
    a directly comparable lineage and so a coefficients table with std_error/p_value
    can be produced, mirroring oam_ml.cxg_event_v3_coefficients).
  - "v_candidate_tree": LightGBM gradient boosting (handles the EDA-documented
    zero-inflated `reception_opponents_within_5m` and the correlated locked-feature
    pairs -- is_cross/is_cut_back, pass_type=Corner/play_pattern=From Corner, and
    CxA+'s three reception_* 360 features -- without needing them de-correlated or
    manually transformed, per the EDA doc's own recommendation). Hyperparameters below
    are reasonable, UNTUNED defaults for a first candidate, not a frozen choice --
    tuning is explicitly out of scope for this task (split policy step 9, later).

Feature set and encoding fixes: see _cxa_modeling_common.py's module docstring
(pass_technique_name=Straight pooled into reference, start_x clipped to [0,120],
CxA+'s 154 missing-360-geometry rows flagged + imputed rather than dropped,
pass_body_part_name=No Touch excluded from the CxA+ candidate set).

Both families fit on TRAIN only, evaluated on TRAIN and VALIDATION. Test is never
queried. Does not freeze a feature set or hyperparameters -- that is split policy step
9, a separate, later, reviewed decision.

Writes (WRITE_TRUNCATE):
  oam_ml.cxa_event_candidate_v1_metrics / _predictions / _coefficients / _calibration
  oam_ml.cxa_plus_candidate_v1_metrics / _predictions / _coefficients / _calibration
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import bigquery
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cxa_modeling_common import (  # noqa: E402
    ML_DATASET,
    PROJECT,
    calibration_table,
    compute_metrics,
    encode_event_candidate,
    encode_plus_candidate,
    load_track,
    write_table,
)

MATERIALIZED_AT = datetime.now(UTC).isoformat()

# Untuned, reasonable defaults for a first candidate -- deliberately conservative
# (shallow trees, modest estimator count, an L2-ish min_child_samples floor) given the
# ~2% positive rate; not a tuned or frozen hyperparameter set.
LGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


def encode_track(track: str, df: pd.DataFrame, train_median_dist: float | None) -> pd.DataFrame:
    if track == "event":
        return encode_event_candidate(df)
    return encode_plus_candidate(df, train_median_dist)


def fit_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Logit:
    X = add_constant(X_train, has_constant="add")
    return Logit(y_train, X).fit(disp=0, maxiter=200)


def predict_logistic(model: Logit, X: pd.DataFrame) -> np.ndarray:
    X = add_constant(X, has_constant="add")
    X = X[model.params.index]
    return model.predict(X).to_numpy()


def coefficients_frame(track: str, model: Logit) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track": track,
            "feature": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
            "materialized_at": MATERIALIZED_AT,
        }
    )


def run_track(client: bigquery.Client, track: str) -> None:
    print(f"=== {track} candidate models ===")
    df = load_track(client, track)
    train_raw = df[df["split"] == "train"]
    validation_raw = df[df["split"] == "validation"]

    train_median_dist = None
    if track == "plus":
        train_median_dist = float(train_raw["reception_nearest_opponent_distance_m"].median())
        print(f"train median reception_nearest_opponent_distance_m (imputation value): {train_median_dist:.3f}")

    X_train = encode_track(track, train_raw, train_median_dist)
    X_val = encode_track(track, validation_raw, train_median_dist)
    y_train = train_raw["y_create"].astype(int)
    y_val = validation_raw["y_create"].astype(int)

    print(f"train n={len(X_train)}, validation n={len(X_val)}, feature columns={list(X_train.columns)}")

    # --- Logistic candidate ---
    logit_model = fit_logistic(X_train, y_train)
    train_logit_pred = predict_logistic(logit_model, X_train)
    val_logit_pred = predict_logistic(logit_model, X_val)
    coeffs_df = coefficients_frame(track, logit_model)

    # --- Tree candidate ---
    tree_model = lgb.LGBMClassifier(**LGB_PARAMS)
    tree_model.fit(X_train, y_train)
    train_tree_pred = tree_model.predict_proba(X_train)[:, 1]
    val_tree_pred = tree_model.predict_proba(X_val)[:, 1]

    metrics_rows = []
    predictions_frames = []
    calibration_rows = []

    for split_name, y_true, logit_pred, tree_pred, split_df in (
        ("train", y_train.to_numpy().astype(bool), train_logit_pred, train_tree_pred, train_raw),
        ("validation", y_val.to_numpy().astype(bool), val_logit_pred, val_tree_pred, validation_raw),
    ):
        for model_name, pred in (
            ("v_candidate_logistic", logit_pred),
            ("v_candidate_tree", tree_pred),
        ):
            m = compute_metrics(y_true, pred, with_auc=True)
            metrics_rows.append(
                {
                    "track": track,
                    "split": split_name,
                    "model": model_name,
                    "n": m["n"],
                    "log_loss": m["log_loss"],
                    "brier_score": m["brier_score"],
                    "roc_auc": m["roc_auc"],
                    "materialized_at": MATERIALIZED_AT,
                }
            )
            cal = calibration_table(y_true, pred)
            cal["track"] = track
            cal["split"] = split_name
            cal["model"] = model_name
            cal["materialized_at"] = MATERIALIZED_AT
            calibration_rows.append(cal)

        predictions_frames.append(
            pd.DataFrame(
                {
                    "track": track,
                    "pass_event_id": split_df["pass_event_id"].to_numpy(),
                    "split": split_name,
                    "v_candidate_logistic_predicted_prob": logit_pred,
                    "v_candidate_tree_predicted_prob": tree_pred,
                    "y_create": y_true,
                    "materialized_at": MATERIALIZED_AT,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(predictions_frames, ignore_index=True)
    calibration_df = pd.concat(calibration_rows, ignore_index=True)

    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_candidate_v1_metrics", metrics_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_candidate_v1_predictions", predictions_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_candidate_v1_coefficients", coeffs_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_candidate_v1_calibration", calibration_df)

    print(metrics_df.to_string(index=False))
    print()


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    for track in ("event", "plus"):
        run_track(client, track)


if __name__ == "__main__":
    main()
