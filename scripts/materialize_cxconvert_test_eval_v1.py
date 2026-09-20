"""CxA P_convert: the ONE-TIME sealed test-split evaluation, both tracks.

This is the only script in this project's P_convert history permitted to read
`split='test'` from the training matrices (`_cxconvert_modeling_common.load_track`
defaults to train+validation only everywhere else; this script explicitly opts in by
passing `splits=("train", "validation", "test")`, done once, for this one run). Test
is scored exactly once per track -- no iteration, no comparing alternatives on test, no
re-tuning based on what test shows.

Refit decision (stated per the task, not left implicit): each track's frozen model is
refit on train+validation COMBINED, using the exact feature list and exact
hyperparameters read directly from `oam_ml.cxconvert_{track}_frozen_v1_config` (not
hardcoded here, to avoid any risk of drift from what was actually frozen). This is
standard practice once hyperparameters/family are chosen: validation's only remaining
job is to serve as more training signal for the final fit. This script does not also
evaluate the validation-fit model on test for comparison -- doing so and picking
whichever looked better would be exactly the test-leakage-through-model-selection this
step exists to avoid.

Per-track model family is READ from the frozen config, not assumed to be the same for
both tracks (unlike P_create, where both tracks froze tree -- P_convert froze tree for
event-only and logistic for CxA+, per docs/analysis/cxa_p_convert_freeze_v1.md).

The full ladder is scored on test for a direct read against the baseline/candidate
doc's validation numbers: `dumb_baseline` (train+validation goal rate, constant), `v1`
(XY-only logistic on shot_x_sb/shot_y_sb, refit on train+validation), and
`frozen_candidate` (the refit frozen model -- tree for event, logistic for CxA+).

Writes (WRITE_TRUNCATE, one run only, `split='test'` stated explicitly on every row):
  oam_ml.cxconvert_event_test_v1_metrics / _calibration / _predictions
  oam_ml.cxconvert_plus_test_v1_metrics / _calibration / _predictions

Does not touch, modify, or re-freeze `oam_ml.cxconvert_{track}_frozen_v1_config` --
read only.
"""

from __future__ import annotations

from datetime import UTC, datetime

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

ML_DATASET = "oam_ml"
MATERIALIZED_AT = datetime.now(UTC).isoformat()
ALL_SPLITS = ("train", "validation", "test")


def read_frozen_config(client: bigquery.Client, track: str) -> dict:
    sql = f"SELECT * FROM `{PROJECT}.{ML_DATASET}.cxconvert_{track}_frozen_v1_config`"
    rows = list(client.query(sql).result())
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly 1 frozen config row for {track}, found {len(rows)}")
    return dict(rows[0].items())


def fit_xy(train_df: pd.DataFrame) -> Logit:
    X = add_constant(encode_xy(train_df), has_constant="add")
    y = train_df["y_goal"].astype(int)
    return Logit(y, X).fit(disp=0, maxiter=200)


def predict_xy(model: Logit, df: pd.DataFrame) -> np.ndarray:
    X = add_constant(encode_xy(df), has_constant="add")
    X = X[model.params.index]
    return model.predict(X).to_numpy()


def write_table(client: bigquery.Client, table: str, df: pd.DataFrame) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table, job_config=job_config, location="europe-west2")
    job.result()
    print(f"wrote {len(df)} row(s) -> {table}")


def separation_check(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    rows = []
    for col in X.columns:
        vals = X[col].unique()
        if not set(vals).issubset({0.0, 1.0}):
            continue
        true_mask = X[col] == 1
        rows.append({
            "feature": col,
            "n_true": int(true_mask.sum()), "n_true_positive": int(y[true_mask].sum()),
            "n_false": int((~true_mask).sum()), "n_false_positive": int(y[~true_mask].sum()),
        })
    return rows


def run_track(client: bigquery.Client, track: str) -> dict:
    print(f"=== {track}: sealed test evaluation (ONE TIME) ===")
    frozen = read_frozen_config(client, track)
    model_family = frozen["model_family"]
    print(f"  frozen model_family: {model_family!r}, feature_list ({len(frozen['feature_list'])} entries): "
          f"{list(frozen['feature_list'])}")

    df = load_track(client, track, splits=ALL_SPLITS)
    fit_pool = df[df["split"].isin(("train", "validation"))].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  fit pool (train+validation) n={len(fit_pool)}, test n={len(test_df)}")

    y_fit = fit_pool["y_goal"].astype(int)
    y_test = test_df["y_goal"].astype(bool)

    train_median_gk = float(fit_pool["shot_gk_distance_m"].median())
    n_null_gk_fit = int(fit_pool["shot_gk_distance_m"].isna().sum())
    n_null_gk_test = int(test_df["shot_gk_distance_m"].isna().sum())
    gk_flag_informative = n_null_gk_fit > 0
    print(f"  shot_gk_distance_m: fit-pool median={train_median_gk:.3f}, "
          f"n_null_fit={n_null_gk_fit}, n_null_test={n_null_gk_test}, flag_included={gk_flag_informative}")

    encode_fn = encode_event_candidate if track == "event" else encode_plus_candidate
    is_tree = model_family == "lightgbm_tree"

    X_fit = encode_fn(fit_pool, train_median_gk, for_tree=is_tree, gk_flag_informative=gk_flag_informative)
    X_test = encode_fn(test_df, train_median_gk, for_tree=is_tree, gk_flag_informative=gk_flag_informative)

    # --- dumb baseline: fit-pool goal rate ---
    dumb_rate = float(y_fit.mean())
    dumb_pred = np.full(len(test_df), dumb_rate)

    # --- XY baseline (v1), refit on train+validation ---
    xy_model = fit_xy(fit_pool)
    v1_pred = predict_xy(xy_model, test_df)

    # --- frozen candidate, refit on train+validation with the EXACT frozen family/hyperparameters ---
    separation_rows: list[dict] = []
    if is_tree:
        lgb_params = dict(
            n_estimators=int(frozen["n_estimators"]), max_depth=int(frozen["max_depth"]),
            num_leaves=int(frozen["num_leaves"]), learning_rate=float(frozen["learning_rate"]),
            min_child_samples=int(frozen["min_child_samples"]), subsample=float(frozen["subsample"]),
            colsample_bytree=float(frozen["colsample_bytree"]), random_state=42, verbosity=-1,
        )
        print(f"  refitting lightgbm_tree on fit pool: {lgb_params}")
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_fit, y_fit)
        candidate_pred = model.predict_proba(X_test)[:, 1]
    else:
        print("  refitting logistic_mle (statsmodels Logit, no penalty) on fit pool")
        Xc_fit = add_constant(X_fit, has_constant="add")
        logit_model = Logit(y_fit, Xc_fit).fit(disp=0, maxiter=200)
        Xc_test = add_constant(X_test, has_constant="add")[logit_model.params.index]
        candidate_pred = logit_model.predict(Xc_test).to_numpy()
        # Per this task's item 6: explicitly check for separation on the COMBINED
        # train+validation refit population, since the fit population just grew and a
        # thin group could behave differently than it did on train alone.
        separation_rows = separation_check(X_fit, y_fit)
        max_abs_coef_over_se = float((logit_model.params.abs() / logit_model.bse).replace([np.inf, -np.inf], np.nan).max())
        print(f"  logistic coefficient check: max |coef|/std_error = {max_abs_coef_over_se:.2f} "
              f"(a value in the hundreds/thousands would indicate separation; std_error itself: "
              f"max={logit_model.bse.max():.4f})")
        print(f"  logistic std errors: {dict(zip(logit_model.bse.index, logit_model.bse.round(4)))}")

    # --- ONE score on test per model, no iteration ---
    metrics_rows = []
    for model_name, pred, with_auc in (
        ("dumb_baseline", dumb_pred, False),
        ("v1", v1_pred, True),
        ("frozen_candidate", candidate_pred, True),
    ):
        m = compute_metrics(y_test.to_numpy(), pred, with_auc=with_auc)
        metrics_rows.append({
            "track": track, "split": "test", "model": model_name, "n": m["n"],
            "log_loss": m["log_loss"],
            "brier_score": m["brier_score"] if model_name != "dumb_baseline" else None,
            "roc_auc": m["roc_auc"], "model_family": model_family if model_name == "frozen_candidate" else None,
            "materialized_at": MATERIALIZED_AT,
        })
        print(f"  {model_name}: log_loss={m['log_loss']:.5f} brier={m['brier_score']:.5f} roc_auc={m['roc_auc']}")

    metrics_df = pd.DataFrame(metrics_rows)

    calibration_rows = []
    for model_name, pred in (("v1", v1_pred), ("frozen_candidate", candidate_pred)):
        cal = calibration_table(y_test.to_numpy(), pred)
        cal["track"] = track
        cal["split"] = "test"
        cal["model"] = model_name
        cal["materialized_at"] = MATERIALIZED_AT
        calibration_rows.append(cal)
    calibration_df = pd.concat(calibration_rows, ignore_index=True)

    predictions_df = pd.DataFrame({
        "track": track, "pass_event_id": test_df["pass_event_id"].to_numpy(), "split": "test",
        "dumb_baseline_prob": dumb_pred, "v1_predicted_prob": v1_pred,
        "frozen_candidate_predicted_prob": candidate_pred,
        "y_goal": y_test.to_numpy(), "materialized_at": MATERIALIZED_AT,
    })

    write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_test_v1_metrics", metrics_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_test_v1_calibration", calibration_df)
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_test_v1_predictions", predictions_df)
    print()

    return {
        "track": track, "model_family": model_family,
        "fit_n": len(fit_pool), "test_n": len(test_df), "test_positive": int(y_test.sum()),
        "metrics": metrics_df.to_dict(orient="records"),
        "calibration": calibration_df.to_dict(orient="records"),
        "separation_check": separation_rows,
        "gk_flag_informative": gk_flag_informative,
        "n_null_gk_fit": n_null_gk_fit, "n_null_gk_test": n_null_gk_test,
    }


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    results = {}
    for track in ("event", "plus"):
        results[track] = run_track(client, track)

    # Confirm the written tables contain split='test' rows only.
    for track in ("event", "plus"):
        sql = f"""
        SELECT DISTINCT split FROM `{PROJECT}.{ML_DATASET}.cxconvert_{track}_test_v1_metrics`
        UNION ALL
        SELECT DISTINCT split FROM `{PROJECT}.{ML_DATASET}.cxconvert_{track}_test_v1_calibration`
        UNION ALL
        SELECT DISTINCT split FROM `{PROJECT}.{ML_DATASET}.cxconvert_{track}_test_v1_predictions`
        """
        distinct_splits = {r["split"] for r in client.query(sql).result()}
        print(f"{track} written-table distinct split values: {distinct_splits}")
        assert distinct_splits == {"test"}, f"{track}: unexpected split values {distinct_splits}"

    import json
    from pathlib import Path
    out_dir = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxconvert_analysis" / "test_eval_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
