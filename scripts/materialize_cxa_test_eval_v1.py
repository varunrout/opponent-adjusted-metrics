"""CxA P_create Step 10: the ONE-TIME sealed test-split evaluation, both tracks.

This is the only script in this project's history permitted to read `split='test'`
from `oam_analysis.cxa_match_splits_v1` (`_cxa_modeling_common.load_track` defaults to
train+validation only everywhere else; this script explicitly opts in by passing
`splits=("train", "validation", "test")`, done once, for this one run). Test is
scored exactly once per track -- no iteration, no comparing alternatives on test, no
re-tuning based on what test shows.

Refit decision (stated per the task, not left implicit): each track's frozen model is
**refit on train+validation combined**, using the exact feature list and exact
hyperparameters read directly from `oam_ml.cxa_{track}_frozen_v1_config` (not
hardcoded here -- avoids any risk of silently drifting from what was actually frozen).
This is standard practice once hyperparameters are chosen: validation's only remaining
job after model/hyperparameter selection is to serve as more training signal for the
final fit, since it can no longer leak into a decision that has already been made.
This script does not also evaluate the validation-fit model on test for comparison --
doing so and picking whichever looked better would be exactly the test-leakage-through-
model-selection this step is designed to avoid.

The full ladder is scored on test for a direct read against the baseline/candidate
doc's validation numbers: `dumb_baseline` (train+validation creation rate, constant),
`v1` (XY-only logistic, refit on train+validation for the same reason as the frozen
model), and `frozen_tree` (the refit frozen LightGBM candidate).

Writes (WRITE_TRUNCATE, one run only, `split='test'` stated explicitly on every row):
  oam_ml.cxa_event_test_v1_metrics / _calibration / _predictions
  oam_ml.cxa_plus_test_v1_metrics / _calibration / _predictions

Does not touch, modify, or re-freeze `oam_ml.cxa_{track}_frozen_v1_config` -- read
only.
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
    encode_xy,
    load_track,
    write_table,
)

MATERIALIZED_AT = datetime.now(UTC).isoformat()
ALL_SPLITS = ("train", "validation", "test")


def read_frozen_config(client: bigquery.Client, track: str) -> dict:
    sql = f"SELECT * FROM `{PROJECT}.{ML_DATASET}.cxa_{track}_frozen_v1_config`"
    rows = list(client.query(sql).result())
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly 1 frozen config row for {track}, found {len(rows)}")
    return dict(rows[0].items())


def fit_xy(train_df: pd.DataFrame) -> Logit:
    X = add_constant(encode_xy(train_df), has_constant="add")
    y = train_df["y_create"].astype(int)
    return Logit(y, X).fit(disp=0)


def predict_xy(model: Logit, df: pd.DataFrame) -> np.ndarray:
    X = add_constant(encode_xy(df), has_constant="add")
    X = X[model.params.index]
    return model.predict(X).to_numpy()


def run_track(client: bigquery.Client, track: str) -> None:
    print(f"=== {track}: sealed test evaluation (ONE TIME) ===")
    frozen = read_frozen_config(client, track)
    print(f"  frozen config: {frozen['chosen_config_label']!r} "
          f"(n_estimators={frozen['n_estimators']}, max_depth={frozen['max_depth']}, "
          f"num_leaves={frozen['num_leaves']}, learning_rate={frozen['learning_rate']}, "
          f"min_child_samples={frozen['min_child_samples']}, subsample={frozen['subsample']}, "
          f"colsample_bytree={frozen['colsample_bytree']})")

    df = load_track(client, track, splits=ALL_SPLITS)
    fit_pool = df[df["split"].isin(("train", "validation"))]
    test_df = df[df["split"] == "test"]
    print(f"  fit pool (train+validation) n={len(fit_pool)}, test n={len(test_df)}")

    # --- encode ---
    if track == "event":
        X_fit = encode_event_candidate(fit_pool)
        X_test = encode_event_candidate(test_df)
    else:
        median_dist = float(fit_pool["reception_nearest_opponent_distance_m"].median())
        print(f"  fit-pool median reception_nearest_opponent_distance_m (imputation value): {median_dist:.3f}")
        X_fit = encode_plus_candidate(fit_pool, median_dist)
        X_test = encode_plus_candidate(test_df, median_dist)

    y_fit = fit_pool["y_create"].astype(int)
    y_test = test_df["y_create"].astype(bool)

    # --- dumb baseline: fit-pool creation rate ---
    dumb_rate = float(y_fit.mean())
    dumb_pred = np.full(len(test_df), dumb_rate)

    # --- XY baseline (v1), refit on train+validation ---
    xy_model = fit_xy(fit_pool)
    v1_pred = predict_xy(xy_model, test_df)

    # --- frozen tree, refit on train+validation with the EXACT frozen hyperparameters ---
    lgb_params = dict(
        n_estimators=int(frozen["n_estimators"]),
        max_depth=int(frozen["max_depth"]),
        num_leaves=int(frozen["num_leaves"]),
        learning_rate=float(frozen["learning_rate"]),
        min_child_samples=int(frozen["min_child_samples"]),
        subsample=float(frozen["subsample"]),
        colsample_bytree=float(frozen["colsample_bytree"]),
        random_state=42,
        verbosity=-1,
    )
    tree_model = lgb.LGBMClassifier(**lgb_params)
    tree_model.fit(X_fit, y_fit)
    tree_pred = tree_model.predict_proba(X_test)[:, 1]

    # --- ONE score on test per model, no iteration ---
    metrics_rows = []
    for model_name, pred, with_auc in (
        ("dumb_baseline", dumb_pred, False),
        ("v1", v1_pred, True),
        ("frozen_tree", tree_pred, True),
    ):
        m = compute_metrics(y_test.to_numpy(), pred, with_auc=with_auc)
        metrics_rows.append(
            {
                "track": track,
                "split": "test",
                "model": model_name,
                "n": m["n"],
                "log_loss": m["log_loss"],
                "brier_score": m["brier_score"] if model_name != "dumb_baseline" else None,
                "roc_auc": m["roc_auc"],
                "materialized_at": MATERIALIZED_AT,
            }
        )
        print(f"  {model_name}: log_loss={m['log_loss']:.5f} "
              f"brier={m['brier_score']:.5f} roc_auc={m['roc_auc']}")

    metrics_df = pd.DataFrame(metrics_rows)

    calibration_rows = []
    for model_name, pred in (("v1", v1_pred), ("frozen_tree", tree_pred)):
        cal = calibration_table(y_test.to_numpy(), pred)
        cal["track"] = track
        cal["split"] = "test"
        cal["model"] = model_name
        cal["materialized_at"] = MATERIALIZED_AT
        calibration_rows.append(cal)
    calibration_df = pd.concat(calibration_rows, ignore_index=True)

    predictions_df = pd.DataFrame(
        {
            "track": track,
            "pass_event_id": test_df["pass_event_id"].to_numpy(),
            "split": "test",
            "dumb_baseline_prob": dumb_pred,
            "v1_predicted_prob": v1_pred,
            "frozen_tree_predicted_prob": tree_pred,
            "y_create": y_test.to_numpy(),
            "materialized_at": MATERIALIZED_AT,
        }
    )

    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_test_v1_metrics", metrics_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_test_v1_calibration", calibration_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_test_v1_predictions", predictions_df)
    print()


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    for track in ("event", "plus"):
        run_track(client, track)


if __name__ == "__main__":
    main()
