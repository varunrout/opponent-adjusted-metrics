"""CxA P_convert model ladder: dumb baseline -> XY baseline (v1) -> candidate models
(logistic + tree), for BOTH tracks. Train + validation only -- test stays sealed,
never queried by this script.

Writes no BigQuery tables (unlike P_create's equivalent scripts, which also persisted
to oam_ml -- out of this task's explicit scope, which asks only for the write-up doc).
Writes JSON results to audit_outputs/cxconvert_analysis/baseline_and_candidate/ for the
doc to be written from.

Encoding fixes: see _cxconvert_modeling_common.py's module docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import bigquery
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

from _cxconvert_modeling_common import (
    PROJECT,
    calibration_table,
    compute_metrics,
    encode_event_candidate,
    encode_plus_candidate,
    encode_xy,
    load_track,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxconvert_analysis" / "baseline_and_candidate"

# Untuned, reasonable first-pass defaults, mirroring P_create's own candidate config.
LGB_PARAMS_EVENT = dict(
    n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
)
# CxA+'s train split (1,991 rows, 181 positives) is ~4x smaller than event-only's
# (7,847 rows, 738 positives) -- min_child_samples is raised upward (50 -> 100, double
# event-only's value) so a leaf must cover a comparable SHARE of the smaller
# population, not a comparable absolute count -- 100/1,991 rows is proportionally a
# tighter constraint than 50/7,847, which is the direction this track's smaller size
# warrants (a first attempt at 30, i.e. a DECREASE, was tried and found to badly
# overfit -- train log_loss 0.165 vs validation 0.353, worse than the dumb baseline --
# corrected here; see the doc's diagnostics section for the full empirical comparison).
# n_estimators/max_depth/num_leaves are kept at P_create's own mirrored values per the
# task's explicit scope (only min_child_samples was asked to be reconsidered) -- even
# at this corrected value, meaningful train/validation overfitting remains and is
# reported honestly, not tuned away by also changing depth/estimator count.
LGB_PARAMS_PLUS = dict(
    n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
)


def fit_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Logit:
    X = add_constant(X_train, has_constant="add")
    return Logit(y_train, X).fit(disp=0, maxiter=200)


def predict_logistic(model: Logit, X: pd.DataFrame) -> np.ndarray:
    X = add_constant(X, has_constant="add")
    X = X[model.params.index]
    return model.predict(X).to_numpy()


def coefficients_frame(model: Logit) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "p_value": model.pvalues.values,
    })


def separation_check(X_train: pd.DataFrame, y_train: pd.Series, bool_cols: list[str]) -> list[dict]:
    """For every boolean/dummy column, count positives in the TRUE and FALSE groups
    on train. A zero-positive group in either direction is the precondition for
    quasi-complete separation -- checked explicitly, not inferred from std_error alone."""
    rows = []
    for col in bool_cols:
        if col not in X_train.columns:
            continue
        true_mask = X_train[col] == 1
        rows.append({
            "feature": col,
            "n_true": int(true_mask.sum()),
            "n_true_positive": int(y_train[true_mask].sum()),
            "n_false": int((~true_mask).sum()),
            "n_false_positive": int(y_train[~true_mask].sum()),
        })
    return rows


def run_track(client: bigquery.Client, track: str) -> dict:
    print(f"=== {track} ===")
    df = load_track(client, track)
    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    y_train = train_raw["y_goal"].astype(int)
    y_val = val_raw["y_goal"].astype(int)

    result: dict = {"track": track, "train_n": len(train_raw), "val_n": len(val_raw),
                     "train_positive": int(y_train.sum()), "val_positive": int(y_val.sum())}

    # --- 1. Dumb baseline ---
    train_rate = float(y_train.mean())
    dumb_train_pred = np.full(len(y_train), train_rate)
    dumb_val_pred = np.full(len(y_val), train_rate)
    result["dumb_baseline"] = {
        "train_rate": train_rate,
        "train": compute_metrics(y_train.to_numpy().astype(bool), dumb_train_pred, with_auc=False),
        "validation": compute_metrics(y_val.to_numpy().astype(bool), dumb_val_pred, with_auc=False),
    }

    # --- 2. XY baseline (v1) ---
    X_train_xy = encode_xy(train_raw)
    X_val_xy = encode_xy(val_raw)
    xy_model = fit_logistic(X_train_xy, y_train)
    xy_train_pred = predict_logistic(xy_model, X_train_xy)
    xy_val_pred = predict_logistic(xy_model, X_val_xy)
    result["xy_baseline"] = {
        "train": compute_metrics(y_train.to_numpy().astype(bool), xy_train_pred),
        "validation": compute_metrics(y_val.to_numpy().astype(bool), xy_val_pred),
    }

    # --- 3. Candidate models ---
    train_median_gk = float(train_raw["shot_gk_distance_m"].median())
    n_null_gk_train = int(train_raw["shot_gk_distance_m"].isna().sum())
    n_null_gk_val = int(val_raw["shot_gk_distance_m"].isna().sum())
    result["gk_distance_imputation"] = {
        "train_median": train_median_gk, "n_null_train": n_null_gk_train, "n_null_val": n_null_gk_val,
    }

    encode_fn = encode_event_candidate if track == "event" else encode_plus_candidate
    lgb_params = LGB_PARAMS_EVENT if track == "event" else LGB_PARAMS_PLUS
    result["lgb_params"] = lgb_params

    # Decided from TRAIN's null count only, not a per-track hardcoded assumption --
    # see _gk_distance_columns' docstring for why a constant-zero flag singularizes
    # the logistic design matrix.
    gk_flag_informative = n_null_gk_train > 0
    result["gk_distance_imputation"]["flag_included_in_design_matrix"] = gk_flag_informative

    X_train_logit = encode_fn(train_raw, train_median_gk, for_tree=False, gk_flag_informative=gk_flag_informative)
    X_val_logit = encode_fn(val_raw, train_median_gk, for_tree=False, gk_flag_informative=gk_flag_informative)
    X_train_tree = encode_fn(train_raw, train_median_gk, for_tree=True, gk_flag_informative=gk_flag_informative)
    X_val_tree = encode_fn(val_raw, train_median_gk, for_tree=True, gk_flag_informative=gk_flag_informative)

    print(f"  logistic columns: {list(X_train_logit.columns)}")
    print(f"  tree columns: {list(X_train_tree.columns)}")

    logit_model = fit_logistic(X_train_logit, y_train)
    train_logit_pred = predict_logistic(logit_model, X_train_logit)
    val_logit_pred = predict_logistic(logit_model, X_val_logit)
    coeffs_df = coefficients_frame(logit_model)

    tree_model = lgb.LGBMClassifier(**lgb_params)
    tree_model.fit(X_train_tree, y_train)
    train_tree_pred = tree_model.predict_proba(X_train_tree)[:, 1]
    val_tree_pred = tree_model.predict_proba(X_val_tree)[:, 1]

    result["candidate"] = {
        "logistic": {
            "train": compute_metrics(y_train.to_numpy().astype(bool), train_logit_pred),
            "validation": compute_metrics(y_val.to_numpy().astype(bool), val_logit_pred),
            "coefficients": coeffs_df.to_dict(orient="records"),
        },
        "tree": {
            "train": compute_metrics(y_train.to_numpy().astype(bool), train_tree_pred),
            "validation": compute_metrics(y_val.to_numpy().astype(bool), val_tree_pred),
            "feature_importance": dict(zip(X_train_tree.columns, [int(v) for v in tree_model.feature_importances_])),
        },
    }

    # --- Calibration (validation) ---
    result["calibration"] = {
        "logistic": calibration_table(y_val.to_numpy().astype(bool), val_logit_pred).to_dict(orient="records"),
        "tree": calibration_table(y_val.to_numpy().astype(bool), val_tree_pred).to_dict(orient="records"),
    }

    # --- Separation diagnostics ---
    bool_cols = [c for c in X_train_logit.columns if X_train_logit[c].isin([0, 1]).all()]
    result["separation_check"] = separation_check(X_train_logit, y_train, bool_cols)

    return result


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for track in ("event", "plus"):
        result = run_track(client, track)
        out_path = OUTPUT_DIR / f"{track}_result.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
