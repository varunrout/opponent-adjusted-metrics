"""CxA combined scorer: materializes `oam_serving.cxa_{track}_combined_v1`, the first
tables ever written to `oam_serving`, per
docs/analysis/cxa_combined_scorer_design_v1.md (the reviewed design spec this script
implements exactly, not redesigns).

For each track (event, plus):
1. Read BOTH frozen configs live (`oam_ml.cxa_{track}_frozen_v1_config` for P_create,
   `oam_ml.cxconvert_{track}_frozen_v1_config` for P_convert) -- model family, feature
   list, and hyperparameters are never hardcoded here, exactly as every prior freeze/
   test-eval script in this project already does.
2. Refit each frozen model on train+validation COMBINED (the same "final fit"
   population `materialize_cxa_test_eval_v1.py` / `materialize_cxconvert_test_eval_
   v1.py` already used to produce this project's own reported test metrics -- reusing
   that fit here, not inventing a different one).
3. Score P_create's model over its FULL population (all three splits) --
   `p_create_predicted_prob` for (almost) every pass.
4. Score P_convert's model over its own full population (already the `y_create=TRUE`
   subset by construction, per the P_convert pipeline) -- `p_convert_predicted_prob`.
5. Left-join P_convert's scores onto P_create's full population by `pass_event_id`.
   `p_convert_predicted_prob` / `cxa_combined_score` are NULL (never a placeholder)
   for every pass that never created a chance -- per the design doc's Option 1+2
   recommendation.
6. Verify row counts exactly match both tracks' known population/coverage sizes
   BEFORE writing anything -- a join mismatch here would silently corrupt every
   downstream number (this task's own explicit instruction).
7. Write `oam_serving.cxa_event_combined_v1` / `cxa_plus_combined_v1`, ALL THREE
   SPLITS included (per the design doc's decision 4 -- CxG's own real precedent,
   `cxg_event_v3_predictions`, also carries every split; any future public API must
   filter to `split='test'` itself, not something this table does).

Does not touch, modify, or re-freeze any `oam_ml.*_frozen_v1_config` table -- read
only. Does not change any locked feature set, model family, or hyperparameter.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import bigquery
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

import _cxa_modeling_common as pcreate
import _cxconvert_modeling_common as pconvert

PROJECT = "oam-varun-260819"
ML_DATASET = "oam_ml"
SERVING_DATASET = "oam_serving"
LOCATION = "europe-west2"
MATERIALIZED_AT = datetime.now(UTC).isoformat()
ALL_SPLITS = ("train", "validation", "test")

SOURCE_DOCS = [
    "docs/analysis/cxa_combined_scorer_design_v1.md",
    "docs/analysis/cxa_p_create_freeze_v1.md",
    "docs/analysis/cxa_p_create_test_eval_v1.md",
    "docs/analysis/cxa_p_convert_freeze_v1.md",
    "docs/analysis/cxa_p_convert_test_eval_v1.md",
]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxa_combined" / "v1"


def read_frozen_config(client: bigquery.Client, table: str) -> dict:
    rows = list(client.query(f"SELECT * FROM `{PROJECT}.{ML_DATASET}.{table}`").result())
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly 1 row in {table}, found {len(rows)}")
    return dict(rows[0].items())


def lgb_params_from_frozen(frozen: dict) -> dict:
    return dict(
        n_estimators=int(frozen["n_estimators"]), max_depth=int(frozen["max_depth"]),
        num_leaves=int(frozen["num_leaves"]), learning_rate=float(frozen["learning_rate"]),
        min_child_samples=int(frozen["min_child_samples"]), subsample=float(frozen["subsample"]),
        colsample_bytree=float(frozen["colsample_bytree"]), random_state=42, verbosity=-1,
    )


def write_table(client: bigquery.Client, table: str, df: pd.DataFrame) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table, job_config=job_config, location=LOCATION)
    job.result()
    print(f"wrote {len(df)} row(s) -> {table}")


def score_p_create(client: bigquery.Client, track: str) -> tuple[pd.DataFrame, dict]:
    """Returns (df with pass_event_id/match_id/split/y_create/p_create_predicted_prob,
    the frozen config dict actually used)."""
    frozen = read_frozen_config(client, f"cxa_{track}_frozen_v1_config")
    print(f"  [P_create/{track}] frozen model_family={frozen['model_family']!r}, "
          f"{len(frozen['feature_list'])} features")

    df = pcreate.load_track(client, track, splits=ALL_SPLITS)
    fit_pool = df[df["split"].isin(("train", "validation"))].reset_index(drop=True)
    y_fit = fit_pool["y_create"].astype(int)

    if track == "event":
        X_fit = pcreate.encode_event_candidate(fit_pool)
        X_all = pcreate.encode_event_candidate(df)
    else:
        train_median_dist = float(fit_pool["reception_nearest_opponent_distance_m"].median())
        print(f"  [P_create/plus] fit-pool median reception_nearest_opponent_distance_m: {train_median_dist:.3f}")
        X_fit = pcreate.encode_plus_candidate(fit_pool, train_median_dist)
        X_all = pcreate.encode_plus_candidate(df, train_median_dist)

    lgb_params = lgb_params_from_frozen(frozen)
    print(f"  [P_create/{track}] refitting lightgbm_tree on fit pool (n={len(X_fit)}): {lgb_params}")
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_fit, y_fit)
    p_create_pred = model.predict_proba(X_all)[:, 1]

    out = pd.DataFrame({
        "pass_event_id": df["pass_event_id"].to_numpy(),
        "match_id": df["match_id"].to_numpy(),
        "split": df["split"].to_numpy(),
        "y_create": df["y_create"].to_numpy(),
        "p_create_predicted_prob": p_create_pred,
    })
    return out, frozen


def score_p_convert(client: bigquery.Client, track: str) -> tuple[pd.DataFrame, dict]:
    """Returns (df with pass_event_id/shot_event_id/y_goal/p_convert_predicted_prob,
    the frozen config dict actually used). Population is already y_create=TRUE by
    construction (the whole cxconvert_{track}_v1_training_matrix)."""
    frozen = read_frozen_config(client, f"cxconvert_{track}_frozen_v1_config")
    model_family = frozen["model_family"]
    print(f"  [P_convert/{track}] frozen model_family={model_family!r}, "
          f"{len(frozen['feature_list'])} features")

    # shot_event_id isn't in _cxconvert_modeling_common's SELECT list -- load it
    # separately by the same key and merge, rather than editing that shared module.
    shot_id_sql = f"""
        SELECT pass_event_id, shot_event_id
        FROM `{PROJECT}.oam_features.cxconvert_{track}_v1_training_matrix`
    """
    shot_ids = client.query(shot_id_sql, location=LOCATION).to_dataframe()

    df = pconvert.load_track(client, track, splits=ALL_SPLITS)
    fit_pool = df[df["split"].isin(("train", "validation"))].reset_index(drop=True)
    y_fit = fit_pool["y_goal"].astype(int)

    train_median_gk = float(fit_pool["shot_gk_distance_m"].median())
    n_null_gk_fit = int(fit_pool["shot_gk_distance_m"].isna().sum())
    gk_flag_informative = n_null_gk_fit > 0
    print(f"  [P_convert/{track}] fit-pool median shot_gk_distance_m={train_median_gk:.3f}, "
          f"n_null_fit={n_null_gk_fit}, flag_included={gk_flag_informative}")

    encode_fn = pconvert.encode_event_candidate if track == "event" else pconvert.encode_plus_candidate
    is_tree = model_family == "lightgbm_tree"

    X_fit = encode_fn(fit_pool, train_median_gk, for_tree=is_tree, gk_flag_informative=gk_flag_informative)
    X_all = encode_fn(df, train_median_gk, for_tree=is_tree, gk_flag_informative=gk_flag_informative)

    if is_tree:
        lgb_params = lgb_params_from_frozen(frozen)
        print(f"  [P_convert/{track}] refitting lightgbm_tree on fit pool (n={len(X_fit)}): {lgb_params}")
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_fit, y_fit)
        p_convert_pred = model.predict_proba(X_all)[:, 1]
    else:
        print(f"  [P_convert/{track}] refitting logistic_mle (statsmodels Logit) on fit pool (n={len(X_fit)})")
        Xc_fit = add_constant(X_fit, has_constant="add")
        logit_model = Logit(y_fit, Xc_fit).fit(disp=0, maxiter=200)
        Xc_all = add_constant(X_all, has_constant="add")[logit_model.params.index]
        p_convert_pred = logit_model.predict(Xc_all).to_numpy()

    out = pd.DataFrame({
        "pass_event_id": df["pass_event_id"].to_numpy(),
        "y_goal": df["y_goal"].to_numpy(),
        "p_convert_predicted_prob": p_convert_pred,
    })
    out = out.merge(shot_ids, on="pass_event_id", how="left")
    return out, frozen


def run_track(client: bigquery.Client, track: str) -> dict:
    print(f"=== {track} ===")
    create_df, create_frozen = score_p_create(client, track)
    convert_df, convert_frozen = score_p_convert(client, track)

    combined = create_df.merge(
        convert_df[["pass_event_id", "shot_event_id", "p_convert_predicted_prob", "y_goal"]],
        on="pass_event_id", how="left",
    )
    # Nullable boolean dtype: y_goal is NaN (not True/False) for every non-chance-
    # creating row after the left join -- plain bool can't represent that, and a
    # mixed object dtype (True/False/nan) risks a bad BigQuery schema autodetect.
    combined["y_create"] = combined["y_create"].astype("boolean")
    combined["y_goal"] = combined["y_goal"].astype("boolean")
    combined["cxa_combined_score"] = combined["p_create_predicted_prob"] * combined["p_convert_predicted_prob"]
    combined["p_create_model_family"] = create_frozen["model_family"]
    combined["p_create_model_version"] = f"cxa_{track}_frozen_v1"
    combined["p_convert_model_family"] = np.where(
        combined["p_convert_predicted_prob"].notna(), convert_frozen["model_family"], None
    )
    combined["p_convert_model_version"] = np.where(
        combined["p_convert_predicted_prob"].notna(), f"cxconvert_{track}_frozen_v1", None
    )
    combined["materialized_at"] = MATERIALIZED_AT
    combined["source_docs"] = [SOURCE_DOCS] * len(combined)

    # --- Verification: row counts, BEFORE writing anything ---
    n_total = len(combined)
    n_convert_pop = len(convert_df)
    n_nonnull_combined = int(combined["cxa_combined_score"].notna().sum())
    n_nonnull_pconvert = int(combined["p_convert_predicted_prob"].notna().sum())
    n_ycreate_true_in_create = int(combined["y_create"].sum())

    print(f"  P_create population (all splits): {n_total}")
    print(f"  P_convert population (all splits): {n_convert_pop}")
    print(f"  non-null p_convert_predicted_prob in combined: {n_nonnull_pconvert}")
    print(f"  non-null cxa_combined_score in combined: {n_nonnull_combined}")
    print(f"  y_create=TRUE rows in P_create population: {n_ycreate_true_in_create}")

    problems = []
    if n_nonnull_pconvert != n_convert_pop:
        problems.append(
            f"non-null p_convert count ({n_nonnull_pconvert}) != P_convert population size ({n_convert_pop})"
        )
    if n_nonnull_combined != n_nonnull_pconvert:
        problems.append(
            f"non-null cxa_combined_score ({n_nonnull_combined}) != non-null p_convert ({n_nonnull_pconvert})"
        )
    if n_ycreate_true_in_create != n_convert_pop:
        problems.append(
            f"y_create=TRUE count in P_create population ({n_ycreate_true_in_create}) "
            f"!= P_convert population size ({n_convert_pop}) -- population mismatch between the two matrices"
        )
    if problems:
        raise RuntimeError(f"{track}: join verification FAILED, refusing to write: " + "; ".join(problems))
    print(f"  [{track}] join verification PASSED -- row counts match exactly, writing table.")

    write_table(client, f"{PROJECT}.{SERVING_DATASET}.cxa_{track}_combined_v1", combined)

    return {
        "track": track, "n_total": n_total, "n_convert_pop": n_convert_pop,
        "n_nonnull_combined": n_nonnull_combined,
        "combined_df": combined,
    }


def combined_quality_metric(result: dict) -> dict:
    """Per this task's item 5: log_loss/roc_auc/brier of cxa_combined_score against
    y_goal, restricted to y_create=TRUE, test split only. Also reports
    p_convert_predicted_prob alone against the same population/target for comparison."""
    df = result["combined_df"]
    pop = df[(df["split"] == "test") & df["cxa_combined_score"].notna()].copy()
    y_true = pop["y_goal"].to_numpy().astype(bool)

    combined_metrics = pconvert.compute_metrics(y_true, pop["cxa_combined_score"].to_numpy(), with_auc=True)
    pconvert_alone_metrics = pconvert.compute_metrics(y_true, pop["p_convert_predicted_prob"].to_numpy(), with_auc=True)

    print(f"  [{result['track']}] combined-quality (test, y_create=TRUE, n={len(pop)}):")
    print(f"    cxa_combined_score  vs y_goal: {combined_metrics}")
    print(f"    p_convert_alone     vs y_goal: {pconvert_alone_metrics}")

    return {
        "track": result["track"], "n": len(pop),
        "cxa_combined_score": combined_metrics,
        "p_convert_alone": pconvert_alone_metrics,
    }


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    quality = {}
    for track in ("event", "plus"):
        result = run_track(client, track)
        results[track] = result
        quality[track] = combined_quality_metric(result)

    summary = {
        track: {
            "n_total": r["n_total"], "n_convert_pop": r["n_convert_pop"],
            "n_nonnull_combined": r["n_nonnull_combined"],
            "combined_quality": quality[track],
        }
        for track, r in results.items()
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
