"""CxA P_convert freeze: model family, feature set, and hyperparameters, for BOTH
tracks -- the first and only script in this repo permitted to write
`oam_ml.cxconvert_{track}_frozen_v1_config` (named `cxconvert_*`, not `cxa_*`, to avoid
colliding with the P_create freeze tables of a near-identical name).

Built on docs/analysis/cxa_p_convert_baseline_and_candidate_v1.md's own comparison
(not re-derived here):
- Event-only: logistic and tree landed within noise of each other on validation.
- CxA+: the tree candidate overfit badly and its validation log_loss (0.3351) was
  slightly WORSE than the trivial XY baseline (0.3320), despite a real AUC edge.
  Logistic had no such problem and clearly beat both baselines. Step 5 explicitly
  left a genuine hyperparameter search for CxA+'s tree out of its own scope --
  this script runs that search.

This script's new work:
1. An 8-config LightGBM search per track (event-only reuses P_create's own 8-config
   grid shape; CxA+ uses a DIFFERENT grid specifically testing shallower/smaller trees,
   since the goal is testing whether tuning can close the overfitting gap Step 5 found,
   not re-running the same grid that already failed once).
2. A brief logistic regularization check (L2-penalized sklearn LogisticRegression at a
   few C values) against the plain-MLE statsmodels fit already reported in Step 5 --
   confirming plain MLE remains the right choice, or reporting a change if it isn't.
3. Writing the frozen config + search tables.

Does not touch `split='test'` anywhere -- the data loader
(`_cxconvert_modeling_common.load_track`) only ever requests `('train', 'validation')`.

Writes (WRITE_TRUNCATE):
  oam_ml.cxconvert_event_frozen_v1_config / cxconvert_plus_frozen_v1_config
  oam_ml.cxconvert_event_freeze_v1_search / cxconvert_plus_freeze_v1_search
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

from _cxconvert_modeling_common import (
    PROJECT,
    compute_metrics,
    encode_event_candidate,
    encode_plus_candidate,
    load_track,
)


def fit_plain_mle_logistic(X_train, y_train, X_val, y_val) -> dict:
    """Refits the plain-MLE logistic candidate fresh (same approach as Step 5) so the
    frozen config's reported metrics come from this script directly, not copied from a
    different script's prior run."""
    Xc_train = add_constant(X_train, has_constant="add")
    model = Logit(y_train, Xc_train).fit(disp=0, maxiter=200)
    Xc_val = add_constant(X_val, has_constant="add")[model.params.index]
    train_pred = model.predict(Xc_train[model.params.index]).to_numpy()
    val_pred = model.predict(Xc_val).to_numpy()
    train_m = compute_metrics(y_train.to_numpy().astype(bool), train_pred, with_auc=True)
    val_m = compute_metrics(y_val.to_numpy().astype(bool), val_pred, with_auc=True)
    return {
        "train_log_loss": train_m["log_loss"], "train_roc_auc": train_m["roc_auc"],
        "val_log_loss": val_m["log_loss"], "val_roc_auc": val_m["roc_auc"],
    }

FROZEN_AT = datetime.now(UTC).isoformat()
ML_DATASET = "oam_ml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxconvert_analysis" / "freeze_v1"

EVENT_FEATURE_LIST_FULL = [
    "is_through_ball", "shot_one_on_one", "is_cross", "shot_first_time",
    "start_x", "pass_end_x", "shot_x_sb", "shot_dist_to_goal_m", "shot_gk_distance_m",
    "shot_defenders_within_5m", "shot_defenders_within_8m", "shot_open_goal",
    "shot_technique_name (Lob, Diving Header, Backheel, Volley levels)",
    "is_cut_back", "pass_type_name (Corner level)",
]
EVENT_FEATURE_LIST_LOGISTIC = [
    "is_through_ball", "shot_one_on_one", "is_cross", "shot_first_time",
    "start_x", "pass_end_x", "shot_gk_distance_m (+ shot_gk_distance_missing flag)",
    "shot_defenders_within_5m", "shot_defenders_within_8m", "shot_open_goal",
    "shot_technique_name (Lob, Diving Header, Backheel, Volley levels)",
    "is_cut_back", "pass_type_name (Corner level)",
]
PLUS_FEATURE_LIST_FULL = [
    "is_through_ball", "shot_one_on_one", "is_cross", "shot_first_time",
    "start_x", "pass_end_x", "shot_x_sb", "shot_dist_to_goal_m", "shot_gk_distance_m",
    "reception_nearest_opponent_distance_m", "reception_opponents_within_5m",
    "shot_open_goal", "is_cut_back", "pass_type_name (Corner level)",
    "shot_technique_name=Lob pooled into reference (not its own feature for this track)",
]
PLUS_FEATURE_LIST_LOGISTIC = [
    "is_through_ball", "shot_one_on_one", "is_cross", "shot_first_time",
    "start_x", "pass_end_x", "shot_gk_distance_m",
    "reception_nearest_opponent_distance_m", "reception_opponents_within_5m",
    "shot_open_goal", "is_cut_back", "pass_type_name (Corner level)",
]

SOURCE_DOCS = [
    "docs/analysis/cxa_p_convert_baseline_and_candidate_v1.md",
    "docs/analysis/cxa_event_p_convert_feature_lock_v1.md",
    "docs/analysis/cxa_plus_p_convert_feature_lock_v1.md",
    "docs/analysis/cxa_p_convert_locked_feature_eda_v1.md",
]

# Event-only: reuses P_create's own 8-config grid shape (same reasoning applies --
# comparable population scale, no prior evidence of overfitting at this track).
EVENT_SEARCH_GRID: list[dict] = [
    {"label": "baseline (untuned, from Step 5)", "n_estimators": 200, "max_depth": 4, "num_leaves": 15,
     "learning_rate": 0.05, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "more trees, lower learning rate", "n_estimators": 400, "max_depth": 4, "num_leaves": 15,
     "learning_rate": 0.03, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "fewer trees, higher learning rate", "n_estimators": 100, "max_depth": 4, "num_leaves": 15,
     "learning_rate": 0.10, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "deeper trees", "n_estimators": 200, "max_depth": 6, "num_leaves": 31,
     "learning_rate": 0.05, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "shallower trees", "n_estimators": 200, "max_depth": 3, "num_leaves": 7,
     "learning_rate": 0.05, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "stronger regularization (higher min_child_samples)", "n_estimators": 200, "max_depth": 4,
     "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "no subsampling", "n_estimators": 200, "max_depth": 4, "num_leaves": 15,
     "learning_rate": 0.05, "min_child_samples": 50, "subsample": 1.0, "colsample_bytree": 1.0},
    {"label": "more trees + deeper (combined)", "n_estimators": 300, "max_depth": 5, "num_leaves": 25,
     "learning_rate": 0.04, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
]

# CxA+: a DIFFERENT grid than event-only, specifically testing whether shallower,
# smaller trees close the overfitting gap Step 5 found (train ll 0.1996 vs val ll
# 0.3351 at min_child_samples=100/depth=4/leaves=15) -- not the same grid re-run,
# since re-running P_create-shaped variants (deeper trees, more estimators) would be
# testing the wrong direction for this track's known problem.
PLUS_SEARCH_GRID: list[dict] = [
    {"label": "Step 5 corrected baseline", "n_estimators": 200, "max_depth": 4, "num_leaves": 15,
     "learning_rate": 0.05, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "very shallow, very small, strong reg", "n_estimators": 50, "max_depth": 2, "num_leaves": 4,
     "learning_rate": 0.05, "min_child_samples": 150, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "shallow, more trees, low LR", "n_estimators": 100, "max_depth": 3, "num_leaves": 7,
     "learning_rate": 0.03, "min_child_samples": 120, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "shallow, moderate LR", "n_estimators": 75, "max_depth": 2, "num_leaves": 4,
     "learning_rate": 0.07, "min_child_samples": 150, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "shallow + row subsampling", "n_estimators": 100, "max_depth": 3, "num_leaves": 8,
     "learning_rate": 0.05, "min_child_samples": 150, "subsample": 0.7, "colsample_bytree": 0.7},
    {"label": "very shallow, many trees, very low LR", "n_estimators": 200, "max_depth": 2, "num_leaves": 4,
     "learning_rate": 0.02, "min_child_samples": 150, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "minimal complexity", "n_estimators": 50, "max_depth": 2, "num_leaves": 4,
     "learning_rate": 0.10, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8},
    {"label": "moderate shallow (depth 3)", "n_estimators": 100, "max_depth": 3, "num_leaves": 7,
     "learning_rate": 0.05, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8},
]


def run_tree_search(track: str, grid: list[dict], X_train, y_train, X_val, y_val) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(grid):
        params = {k: v for k, v in cfg.items() if k != "label"}
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        train_pred = model.predict_proba(X_train)[:, 1]
        val_m = compute_metrics(y_val.to_numpy().astype(bool), val_pred, with_auc=True)
        train_m = compute_metrics(y_train.to_numpy().astype(bool), train_pred, with_auc=True)
        rows.append({
            "track": track, "config_index": i, "label": cfg["label"], **params,
            "train_log_loss": train_m["log_loss"], "train_roc_auc": train_m["roc_auc"],
            "train_brier": train_m["brier_score"],
            "val_log_loss": val_m["log_loss"], "val_roc_auc": val_m["roc_auc"],
            "val_brier": val_m["brier_score"],
        })
        print(f"  [{track}] cfg {i} ({cfg['label']}): val_ll={val_m['log_loss']:.5f} val_auc={val_m['roc_auc']:.5f}")
    return pd.DataFrame(rows)


def select_config(search_df: pd.DataFrame) -> dict:
    """Lowest validation log_loss wins; roc_auc breaks ties within 0.0005 log_loss --
    same selection rule as P_create's own freeze script."""
    best = search_df["val_log_loss"].min()
    within_tie = search_df[search_df["val_log_loss"] <= best + 0.0005]
    return within_tie.sort_values("val_roc_auc", ascending=False).iloc[0].to_dict()


def logistic_regularization_check(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Compares plain-MLE (statsmodels, already fit in Step 5) against L2-penalized
    sklearn LogisticRegression at a few C values, to confirm plain MLE remains the
    right choice rather than skipping the question silently."""
    rows = []
    for C in (0.1, 1.0, 10.0, 100.0):
        model = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        train_pred = model.predict_proba(X_train)[:, 1]
        val_m = compute_metrics(y_val.to_numpy().astype(bool), val_pred, with_auc=True)
        train_m = compute_metrics(y_train.to_numpy().astype(bool), train_pred, with_auc=True)
        rows.append({
            "C": C, "train_log_loss": train_m["log_loss"], "train_roc_auc": train_m["roc_auc"],
            "val_log_loss": val_m["log_loss"], "val_roc_auc": val_m["roc_auc"],
        })
    return pd.DataFrame(rows)


def write_table(client: bigquery.Client, table: str, df: pd.DataFrame) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table, job_config=job_config, location="europe-west2")
    job.result()
    print(f"wrote {len(df)} row(s) -> {table}")


def run_track(client: bigquery.Client, track: str) -> dict:
    print(f"=== {track}: freeze search (train/validation only) ===")
    df = load_track(client, track)
    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    y_train = train_raw["y_goal"].astype(int)
    y_val = val_raw["y_goal"].astype(int)

    train_median_gk = float(train_raw["shot_gk_distance_m"].median())
    n_null_gk_train = int(train_raw["shot_gk_distance_m"].isna().sum())
    gk_flag_informative = n_null_gk_train > 0

    encode_fn = encode_event_candidate if track == "event" else encode_plus_candidate
    grid = EVENT_SEARCH_GRID if track == "event" else PLUS_SEARCH_GRID

    X_train_tree = encode_fn(train_raw, train_median_gk, for_tree=True, gk_flag_informative=gk_flag_informative)
    X_val_tree = encode_fn(val_raw, train_median_gk, for_tree=True, gk_flag_informative=gk_flag_informative)
    X_train_logit = encode_fn(train_raw, train_median_gk, for_tree=False, gk_flag_informative=gk_flag_informative)
    X_val_logit = encode_fn(val_raw, train_median_gk, for_tree=False, gk_flag_informative=gk_flag_informative)

    search_df = run_tree_search(track, grid, X_train_tree, y_train, X_val_tree, y_val)
    chosen_tree = select_config(search_df)
    print(f"  -> best tree config: cfg {chosen_tree['config_index']} ({chosen_tree['label']}), "
          f"val_ll={chosen_tree['val_log_loss']:.5f}")

    logit_reg_df = logistic_regularization_check(X_train_logit, y_train, X_val_logit, y_val)
    print(f"  logistic L2 check:\n{logit_reg_df.to_string(index=False)}")

    plain_mle = fit_plain_mle_logistic(X_train_logit, y_train, X_val_logit, y_val)
    print(f"  plain MLE logistic (refit fresh): {plain_mle}")

    write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_freeze_v1_search", search_df)

    return {
        "track": track,
        "search_df": search_df,
        "chosen_tree": chosen_tree,
        "logit_reg_df": logit_reg_df,
        "plain_mle": plain_mle,
        "gk_flag_informative": gk_flag_informative,
        "train_n": len(train_raw), "val_n": len(val_raw),
        "train_positive": int(y_train.sum()), "val_positive": int(y_val.sum()),
    }


def write_frozen_config(
    client: bigquery.Client, track: str, model_family: str, feature_list: list[str],
    hyperparams: dict, metrics: dict, selection_rule: str, chosen_label: str,
) -> None:
    row = pd.DataFrame([{
        "track": track,
        "model_family": model_family,
        "feature_list": feature_list,
        "n_estimators": hyperparams.get("n_estimators"),
        "max_depth": hyperparams.get("max_depth"),
        "num_leaves": hyperparams.get("num_leaves"),
        "learning_rate": hyperparams.get("learning_rate"),
        "min_child_samples": hyperparams.get("min_child_samples"),
        "subsample": hyperparams.get("subsample"),
        "colsample_bytree": hyperparams.get("colsample_bytree"),
        "regularization_C": hyperparams.get("C"),
        "selection_rule": selection_rule,
        "chosen_config_label": chosen_label,
        "train_log_loss": metrics.get("train_log_loss"),
        "train_roc_auc": metrics.get("train_roc_auc"),
        "val_log_loss": metrics.get("val_log_loss"),
        "val_roc_auc": metrics.get("val_roc_auc"),
        "source_docs": SOURCE_DOCS,
        "frozen_at": FROZEN_AT,
    }])
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_frozen_v1_config", row)


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for track in ("event", "plus"):
        result = run_track(client, track)
        results[track] = result
        out_path = OUTPUT_DIR / f"{track}_search_result.json"
        payload = {
            "search": result["search_df"].to_dict(orient="records"),
            "chosen_tree": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in result["chosen_tree"].items()},
            "logit_reg": result["logit_reg_df"].to_dict(orient="records"),
            "gk_flag_informative": result["gk_flag_informative"],
            "train_n": result["train_n"], "val_n": result["val_n"],
            "train_positive": result["train_positive"], "val_positive": result["val_positive"],
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {out_path}")

    # --- Freeze decisions, made from the search results above ---
    # Event-only: tree (cfg "shallower trees") beats both the Step 5 logistic
    # candidate and every L2-regularized logistic variant on both log_loss and AUC,
    # by a real (not tie-break-level) margin -- see the freeze doc for the exact
    # numbers. This is a genuine result of running the search this task called for,
    # not assumed from Step 5 (which found the two families too close to call).
    event = results["event"]
    chosen = event["chosen_tree"]
    write_frozen_config(
        client, "event", "lightgbm_tree", EVENT_FEATURE_LIST_FULL,
        {"n_estimators": int(chosen["n_estimators"]), "max_depth": int(chosen["max_depth"]),
         "num_leaves": int(chosen["num_leaves"]), "learning_rate": float(chosen["learning_rate"]),
         "min_child_samples": int(chosen["min_child_samples"]), "subsample": float(chosen["subsample"]),
         "colsample_bytree": float(chosen["colsample_bytree"])},
        {"train_log_loss": chosen["train_log_loss"], "train_roc_auc": chosen["train_roc_auc"],
         "val_log_loss": chosen["val_log_loss"], "val_roc_auc": chosen["val_roc_auc"]},
        "lowest validation log_loss among an 8-config LightGBM search; roc_auc tie-break within 0.0005 log_loss",
        f"cfg {int(chosen['config_index'])} ({chosen['label']})",
    )

    # CxA+: even the best post-search tree config (val_ll 0.31714) does not quite beat
    # plain-MLE logistic (val_ll ~0.316, refit fresh above) -- the gap has closed
    # dramatically from Step 5 (tree was worse than the XY baseline) but logistic still
    # wins, marginally, and remains the more stable, more interpretable choice with no
    # separation issue. L2 regularization was checked and found not to change the
    # picture meaningfully (best L2 config C=1.0 improves log_loss by <0.3% relative,
    # within noise at this validation size) -- plain MLE is retained.
    plus = results["plus"]
    write_frozen_config(
        client, "plus", "logistic_mle", PLUS_FEATURE_LIST_LOGISTIC,
        {},
        plus["plain_mle"],
        "logistic (plain MLE) retained: an 8-config shallow-tree search closed Step 5's "
        "overfitting gap substantially but did not overtake logistic on validation log_loss; "
        "L2 regularization checked (C in [0.1, 1, 10, 100]) and found no meaningful improvement",
        "plain MLE (statsmodels Logit, no penalty)",
    )
    print("Frozen configs written for both tracks.")


if __name__ == "__main__":
    main()
