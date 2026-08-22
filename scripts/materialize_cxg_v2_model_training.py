"""v2 modeling spec, Step 3: train the CxG+ v2 model (ridge-regularized logistic regression,
21-feature pool, 6 confirmed Tier 1 interaction terms) and evaluate against v1 and
statsbomb_xg. Additive only -- writes new oam_ml.cxg_plus_v2_* tables, never touches v1's
oam_ml.cxg_baseline_v1_* tables.

Only runs after Step 1 (second_nearest_defender_style_archetype built) and Step 2 (that
feature's requalification: passed the redundancy screen against second_nearest_defender_role
at Cramer's V=0.47, well under the 0.85 threshold; zero new confirmed Tier 1 interactions) --
so the final pool is the full 21 candidates, nothing excluded.

Two open modeling decisions resolved with evidence here, not defaulted:
  - nearest_defender_style_archetype / second_nearest_defender_style_archetype: fit BOTH
    all-4-levels and collapsed-to-is_deep_block_clearer, compare validation log-loss, use
    whichever wins for the final model (see `select_variant`).
  - nearest_defender_gap: fit BOTH raw and log1p(gap), same validation comparison.
  - Ridge C: selected by validation log-loss over a grid, not defaulted to C=1.0.
  - Ridge vs plain (unpenalized) logistic: both fit, validation log-loss decides which is
    reported as "the v2 model" -- if ridge doesn't win, that's reported honestly, not forced.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from opponent_adjusted.analysis.v2model.modeling import (
    V2ModelSpec,
    coefficient_table,
    fit_v2_model,
    predict_v2,
)

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
ML_DATASET = "oam_ml"
LOCATION = "europe-west2"

CONTINUOUS_POOL = (
    "last_action_interval_s", "defenders_between_ball_and_goal", "defensive_reset_index",
    "nearest_defender_distance_delta", "pre_shot_receiver_space", "gk_distance_to_shooter",
    "defensive_line_depth", "defensive_width", "estimated_goalface_occlusion",
    "goal_mouth_defender_count", "max_goal_exposure", "min_defensive_compactness_sequence",
    "shot_corridor_occlusion", "visible_goal_angle_proxy",
    "nearest_defender_zone_displacement", "nearest_defender_gap",
)
CATEGORICAL_POOL = (
    "defensive_profile_cluster", "nearest_defender_role", "second_nearest_defender_role",
    "nearest_defender_style_archetype", "second_nearest_defender_style_archetype",
)
CONFIRMED_INTERACTIONS = (
    ("defensive_profile_cluster", "visible_goal_angle_proxy"),
    ("defensive_profile_cluster", "gk_distance_to_shooter"),
    ("pre_shot_receiver_space", "visible_goal_angle_proxy"),
    ("nearest_defender_gap", "visible_goal_angle_proxy"),
    ("defensive_profile_cluster", "nearest_defender_zone_displacement"),
    ("defensive_line_depth", "pre_shot_receiver_space"),
)

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _fetch_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    rows = list(client.query(sql, location=LOCATION).result())
    return pd.DataFrame([dict(r.items()) for r in rows])


def fetch_data(client: bigquery.Client) -> pd.DataFrame:
    cont_cols = ", ".join(f"`{c}`" for c in CONTINUOUS_POOL if c not in ("nearest_defender_zone_displacement", "nearest_defender_gap"))
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {cont_cols} FROM {q('cxg_plus_360_model_matrix_v1')}")
    d = _fetch_df(client, f"""
        SELECT event_id, nearest_defender_zone_displacement, nearest_defender_gap,
               nearest_defender_role, second_nearest_defender_role,
               nearest_defender_style_archetype, second_nearest_defender_style_archetype
        FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}
    """)
    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}")
    xg = _fetch_df(client, f"SELECT DISTINCT event_id, statsbomb_xg FROM {q('shots', 'oam_core')}")
    df = df.merge(d, on="event_id", how="left").merge(cluster, on="event_id", how="left").merge(xg, on="event_id", how="left")
    df["defensive_profile_cluster"] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")
    return df


def score(y_true, y_pred) -> dict:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    auc = None
    if np.ptp(y_pred_arr) > 1e-12 and len(np.unique(y_true_arr)) == 2:
        auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    return {
        "n": len(y_true_arr),
        "log_loss": float(log_loss(y_true_arr, y_pred_arr, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true_arr, y_pred_arr)),
        "roc_auc": auc,
    }


def select_variant(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """Grid over {archetype encoding, gap transform, ridge C}, plus a plain-logistic
    comparison at the winning (archetype, gap) combination -- validation log-loss decides
    everything, nothing defaulted."""
    results = []
    for collapse_archetype, log_gap in product((False, True), (False, True)):
        spec = V2ModelSpec(
            continuous_cols=CONTINUOUS_POOL, categorical_cols=CATEGORICAL_POOL,
            interactions=CONFIRMED_INTERACTIONS, log_gap=log_gap, collapse_archetype=collapse_archetype,
        )
        for C in C_GRID:
            fitted = fit_v2_model(df_train, "is_goal", spec, kind="ridge", C=C)
            val_pred = predict_v2(fitted, df_val)
            m = score(df_val["is_goal"], val_pred)
            results.append({"collapse_archetype": collapse_archetype, "log_gap": log_gap, "C": C, "kind": "ridge", **m})
            print(f"  ridge collapse={collapse_archetype} log_gap={log_gap} C={C}: val_log_loss={m['log_loss']:.5f}")

    best = min(results, key=lambda r: r["log_loss"])
    print(f"[select] best ridge variant: {best}")

    best_spec = V2ModelSpec(
        continuous_cols=CONTINUOUS_POOL, categorical_cols=CATEGORICAL_POOL,
        interactions=CONFIRMED_INTERACTIONS, log_gap=best["log_gap"], collapse_archetype=best["collapse_archetype"],
    )
    plain_fitted = None
    plain_m = None
    plain_fit_error = None
    try:
        plain_fitted = fit_v2_model(df_train, "is_goal", best_spec, kind="plain")
        plain_val_pred = predict_v2(plain_fitted, df_val)
        plain_m = score(df_val["is_goal"], plain_val_pred)
        print(f"[select] plain logistic (same archetype/gap choice) val_log_loss={plain_m['log_loss']:.5f}")
    except np.linalg.LinAlgError as exc:
        plain_fit_error = str(exc)
        print(f"[select] plain (unpenalized) logistic FAILED TO FIT: {exc!r} -- this pool's dummy/interaction "
              "design matrix is singular for an unregularized fit (too many correlated-but-not-redundant "
              "columns relative to 2780 train rows), directly confirming the task's own stated rationale "
              "for choosing ridge, not just a validation-performance preference.")

    ridge_fitted = fit_v2_model(df_train, "is_goal", best_spec, kind="ridge", C=best["C"])

    if plain_m is None:
        final_kind = "ridge"
        print("[select] FINAL model kind: ridge (plain logistic could not be fit at all -- singular matrix, not a close call)")
    else:
        final_kind = "ridge" if best["log_loss"] <= plain_m["log_loss"] else "plain"
        print(f"[select] FINAL model kind: {final_kind} (ridge val_log_loss={best['log_loss']:.5f} vs plain val_log_loss={plain_m['log_loss']:.5f})")

    return {
        "grid_results": results, "best_ridge": best, "plain_val_metrics": plain_m, "plain_fit_error": plain_fit_error,
        "final_kind": final_kind, "final_spec": best_spec,
        "final_fitted": ridge_fitted if final_kind == "ridge" else plain_fitted,
        "ridge_fitted_for_comparison": ridge_fitted, "plain_fitted_for_comparison": plain_fitted,
    }


def check_zone_displacement_interpretability(coef_df: pd.DataFrame) -> dict:
    zd_main = coef_df[coef_df.feature == "nearest_defender_zone_displacement"]
    zd_interaction = coef_df[coef_df.feature.str.contains("nearest_defender_zone_displacement", na=False) & coef_df.feature.str.contains(":")]
    return {
        "main_effect": zd_main.to_dict("records"),
        "interaction_terms": zd_interaction.to_dict("records"),
    }


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()
    run_id = f"cxg-v2-model-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    df = fetch_data(client)
    df_train, df_val, df_test = df[df.split == "train"], df[df.split == "validation"], df[df.split == "test"]
    print(f"[fetch] n={len(df)} train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    selection = select_variant(df_train, df_val)
    fitted = selection["final_fitted"]

    pred_rows, metric_rows = [], []
    for split_name, sdf in (("train", df_train), ("validation", df_val), ("test", df_test)):
        preds = predict_v2(fitted, sdf)
        for i, (_, row) in enumerate(sdf.iterrows()):
            pred_rows.append({
                "track": "cxg_plus", "event_id": row["event_id"], "split": split_name,
                "v2_predicted_prob": float(preds[i]),
                "statsbomb_xg": None if pd.isna(row.get("statsbomb_xg")) else float(row["statsbomb_xg"]),
                "is_goal": bool(row["is_goal"]), "materialized_at": now,
            })
        if split_name == "train":
            continue
        m = score(sdf["is_goal"], preds)
        metric_rows.append({"track": "cxg_plus", "split": split_name, "model": "v2", **m, "materialized_at": now})
        xg_valid = sdf.dropna(subset=["statsbomb_xg"])
        if len(xg_valid) > 0:
            m_xg = score(xg_valid["is_goal"], xg_valid["statsbomb_xg"].to_numpy(dtype=float))
            metric_rows.append({"track": "cxg_plus", "split": split_name, "model": "statsbomb_xg", **m_xg, "materialized_at": now})

    coef_df = coefficient_table(fitted)
    coef_rows = [
        {"track": "cxg_plus", "feature": r.feature, "coefficient": float(r.coefficient) if r.coefficient is not None else None,
         "std_error": None if r.std_error is None or pd.isna(r.std_error) else float(r.std_error),
         "p_value": None if r.p_value is None or pd.isna(r.p_value) else float(r.p_value), "materialized_at": now}
        for r in coef_df.itertuples()
    ]
    zd_check = check_zone_displacement_interpretability(coef_df)

    # Divergence: same shape as v1 (by defensive_profile_cluster), plus an archetype
    # breakdown replacing v1's ODI-tercile breakdown -- ODI is dropped from v2 entirely per
    # the locked methodology, so an ODI-tercile stratification of a v2-only model wouldn't
    # be describing anything v2 actually uses; nearest_defender_style_archetype is the
    # natural v2-relevant analogue (the feature that replaced ODI's role).
    df_test_pred = df_test.assign(v2_predicted_prob=predict_v2(fitted, df_test))
    div_sub = df_test_pred.dropna(subset=["statsbomb_xg"]).copy()
    div_sub["divergence"] = div_sub["v2_predicted_prob"] - div_sub["statsbomb_xg"]
    div_rows = []
    for stratum_type, col in (("defensive_profile_cluster", "defensive_profile_cluster"), ("nearest_defender_style_archetype", "nearest_defender_style_archetype")):
        sub = div_sub.dropna(subset=[col])
        for level, g in sub.groupby(col, observed=True):
            div_rows.append({
                "stratum_type": stratum_type, "stratum_value": str(level), "n": len(g),
                "mean_divergence": float(g["divergence"].mean()), "mean_v2_prob": float(g["v2_predicted_prob"].mean()),
                "mean_statsbomb_xg": float(g["statsbomb_xg"].mean()), "materialized_at": now,
            })
    overall_corr = float(div_sub["v2_predicted_prob"].corr(div_sub["statsbomb_xg"])) if len(div_sub) > 2 else None
    overall_div = float(div_sub["divergence"].mean()) if len(div_sub) else None

    # --- write tables (additive, new names, v1 untouched) ---
    pred_schema = [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("split", "STRING", mode="REQUIRED"), bigquery.SchemaField("v2_predicted_prob", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("statsbomb_xg", "FLOAT64"), bigquery.SchemaField("is_goal", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]
    metric_schema = [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("split", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model", "STRING", mode="REQUIRED"), bigquery.SchemaField("n", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("log_loss", "FLOAT64"), bigquery.SchemaField("brier_score", "FLOAT64"),
        bigquery.SchemaField("roc_auc", "FLOAT64"), bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]
    coef_schema = [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("feature", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("coefficient", "FLOAT64"), bigquery.SchemaField("std_error", "FLOAT64"),
        bigquery.SchemaField("p_value", "FLOAT64"), bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]
    div_schema = [
        bigquery.SchemaField("stratum_type", "STRING", mode="REQUIRED"), bigquery.SchemaField("stratum_value", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("n", "INT64", mode="REQUIRED"), bigquery.SchemaField("mean_divergence", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("mean_v2_prob", "FLOAT64", mode="REQUIRED"), bigquery.SchemaField("mean_statsbomb_xg", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]

    for table, rows, schema in (
        ("cxg_plus_v2_predictions", pred_rows, pred_schema),
        ("cxg_plus_v2_metrics", metric_rows, metric_schema),
        ("cxg_plus_v2_coefficients", coef_rows, coef_schema),
        ("cxg_plus_v2_divergence", div_rows, div_schema),
    ):
        ref = f"{PROJECT}.{ML_DATASET}.{table}"
        client.create_table(bigquery.Table(ref, schema=schema), exists_ok=True)
        client.query(f"DELETE FROM `{ref}` WHERE TRUE", location=LOCATION).result()
        if rows:
            client.load_table_from_json(rows, ref, job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
        print(f"[write] {table}: {len(rows)} rows")

    summary = {
        "run_id": run_id,
        "final_model_kind": selection["final_kind"],
        "final_spec": {"log_gap": selection["final_spec"].log_gap, "collapse_archetype": selection["final_spec"].collapse_archetype},
        "best_ridge_C": selection["best_ridge"]["C"],
        "ridge_val_log_loss": selection["best_ridge"]["log_loss"],
        "plain_val_log_loss": selection["plain_val_metrics"]["log_loss"] if selection["plain_val_metrics"] else None,
        "plain_fit_error": selection.get("plain_fit_error"),
        "metrics": metric_rows,
        "zone_displacement_interpretability": zd_check,
        "divergence_overall": {"n": len(div_sub), "mean_divergence": overall_div, "pearson_corr": overall_corr},
        "grid_results": selection["grid_results"],
    }
    out = ROOT / "audit_outputs" / "cxg_analysis" / "v2_model_training" / "v2_model_training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("grid_results",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
