"""v3 model training, BOTH tracks. CxG event-wide (`cxg_event`) gets its first
feature-engineered model beyond v1's kitchen-sink baseline; CxG+ (`cxg_plus`) gets v2's
21-feature pool plus Phase C's 3 new qualifying features (24 total). Additive only -- writes
new `oam_ml.cxg_event_v3_*` / `cxg_plus_v3_*` tables, never touches `cxg_baseline_v1_*` or
`cxg_plus_v2_*`.

MODEL-SPEC DECISIONS, resolved with evidence (not defaulted), per track:

- CxG event-wide plain-vs-ridge: the task instructs NOT to default to ridge just because
  CxG+ needed it -- CxG event-wide's pool is much smaller (8 features, 1 interaction term).
  Both `plain` and `ridge` are fit at every transform combination; if plain fits without a
  `LinAlgError`, its validation log-loss is compared directly against ridge's, and whichever
  is lower ships (same "no forced preference" discipline as v2's own plain-vs-ridge check).
- CxG+ ridge re-verification: v2 found plain logistic literally singular on the 21-feature
  pool (`LinAlgError`, `v2_model_training_summary.json`). Re-attempted here on the full
  24-feature v3 pool (not assumed to still be singular for the same reason) -- see
  `select_variant`'s try/except, which reports the actual outcome.
- New-feature transforms: `defensive_action_rate_30m` and `cross_match_defensive_rate` are
  both non-negative and right-skewed (train-split quantiles confirmed live before writing
  this script: `defensive_action_rate_30m` p50≈2.5-2.8 vs max≈15-23, an 8x+ tail;
  `cross_match_defensive_rate` p50≈2.4-2.7 vs max≈3.9, a much milder ~1.5x tail) -- both are
  grid-tested raw vs log1p, validation log-loss decides. `territorial_dominance_last_15m`
  ranges [-1, 1] (confirmed live) and can be negative, so log1p is not mathematically
  applicable -- left raw, not tested (documented, not silently skipped).
- `nearest_defender_gap`'s log1p transform and the archetype-collapse decision are CARRIED
  FORWARD unchanged from v2's own already-evidenced winning config
  (`log_gap=True, collapse_archetype=False`, `v2_model_training_summary.json`) -- v2 already
  settled these with a grid search; re-litigating them here would just re-derive the same
  answer at extra cost, and v2's own code/tables are frozen regardless.
- `cross_match_defensive_rate`'s cold-start nulls get an explicit `_was_missing` indicator +
  median imputation (`V3ModelSpec.missing_indicator_cols`), never row-dropped -- per the
  task's explicit instruction, mirroring the archetype categoricals' explicit-missing-
  category convention rather than silently imputing away the "no prior match" signal.
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

from opponent_adjusted.analysis.v3model.modeling import (
    V3ModelSpec,
    coefficient_table,
    fit_v3_model,
    predict_v3,
)

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
ML_DATASET = "oam_ml"
LOCATION = "europe-west2"

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
MISSING_INDICATOR_COLS = ("cross_match_defensive_rate",)

# --- CxG event-wide pool (8 candidates, all continuous, 1 confirmed interaction) ----------
CXG_EVENT_CONTINUOUS = (
    "shot_x_sb", "first_box_entry_to_shot_s", "last_box_entry_to_shot_s",
    "last_action_interval_s", "regain_height_speed_interaction",
    "defensive_action_rate_30m", "territorial_dominance_last_15m", "cross_match_defensive_rate",
)
CXG_EVENT_INTERACTIONS = (("defensive_action_rate_30m", "territorial_dominance_last_15m"),)

# --- CxG+ pool (24 candidates: v2's 21 + Phase C's 3, 6 confirmed interactions unchanged) -
CXG_PLUS_CONTINUOUS = (
    "last_action_interval_s", "defenders_between_ball_and_goal", "defensive_reset_index",
    "nearest_defender_distance_delta", "pre_shot_receiver_space", "gk_distance_to_shooter",
    "defensive_line_depth", "defensive_width", "estimated_goalface_occlusion",
    "goal_mouth_defender_count", "max_goal_exposure", "min_defensive_compactness_sequence",
    "shot_corridor_occlusion", "visible_goal_angle_proxy",
    "nearest_defender_zone_displacement", "nearest_defender_gap",
    "defensive_action_rate_30m", "territorial_dominance_last_15m", "cross_match_defensive_rate",
)  # 19
CXG_PLUS_CATEGORICAL = (
    "defensive_profile_cluster", "nearest_defender_role", "second_nearest_defender_role",
    "nearest_defender_style_archetype", "second_nearest_defender_style_archetype",
)  # 5
CXG_PLUS_INTERACTIONS = (
    ("defensive_profile_cluster", "visible_goal_angle_proxy"),
    ("defensive_profile_cluster", "gk_distance_to_shooter"),
    ("pre_shot_receiver_space", "visible_goal_angle_proxy"),
    ("nearest_defender_gap", "visible_goal_angle_proxy"),
    ("defensive_profile_cluster", "nearest_defender_zone_displacement"),
    ("defensive_line_depth", "pre_shot_receiver_space"),
)
CXG_PLUS_FIXED_LOG1P = ("nearest_defender_gap",)  # v2's already-evidenced winning config, carried forward


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _fetch_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    rows = list(client.query(sql, location=LOCATION).result())
    return pd.DataFrame([dict(r.items()) for r in rows])


def fetch_new_phase_c_cols(client: bigquery.Client) -> pd.DataFrame:
    return _fetch_df(client, f"""
        SELECT event_id, defensive_action_rate_30m, territorial_dominance_last_15m, cross_match_defensive_rate
        FROM {q('cxg_training_matrix_v1', FEATURE_DATASET)}
    """)


def fetch_cxg_event(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in CXG_EVENT_CONTINUOUS if c not in ("defensive_action_rate_30m", "territorial_dominance_last_15m", "cross_match_defensive_rate"))
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {cols} FROM {q('cxg_event_model_matrix_v1')}")
    xg = _fetch_df(client, f"SELECT DISTINCT event_id, statsbomb_xg FROM {q('shots', 'oam_core')}")
    return df.merge(fetch_new_phase_c_cols(client), on="event_id", how="left").merge(xg, on="event_id", how="left")


def fetch_cxg_plus(client: bigquery.Client) -> pd.DataFrame:
    cont_cols = ", ".join(f"`{c}`" for c in CXG_PLUS_CONTINUOUS if c not in ("nearest_defender_zone_displacement", "nearest_defender_gap", "defensive_action_rate_30m", "territorial_dominance_last_15m", "cross_match_defensive_rate"))
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {cont_cols} FROM {q('cxg_plus_360_model_matrix_v1')}")
    d = _fetch_df(client, f"""
        SELECT event_id, nearest_defender_zone_displacement, nearest_defender_gap,
               nearest_defender_role, second_nearest_defender_role,
               nearest_defender_style_archetype, second_nearest_defender_style_archetype
        FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}
    """)
    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}")
    xg = _fetch_df(client, f"SELECT DISTINCT event_id, statsbomb_xg FROM {q('shots', 'oam_core')}")
    df = (df.merge(d, on="event_id", how="left").merge(cluster, on="event_id", how="left")
            .merge(fetch_new_phase_c_cols(client), on="event_id", how="left").merge(xg, on="event_id", how="left"))
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


def select_variant(
    df_train: pd.DataFrame, df_val: pd.DataFrame, continuous: tuple, categorical: tuple,
    interactions: tuple, fixed_log1p: tuple, always_try_plain: bool,
) -> dict:
    """Grid over {log1p(defensive_action_rate_30m), log1p(cross_match_defensive_rate)} x
    ridge C, plus a plain-logistic attempt at the winning transform combo. `territorial_
    dominance_last_15m` is never included in the log1p grid (can be negative)."""
    results = []
    for log_rate, log_cm in product((False, True), (False, True)):
        log1p_cols = fixed_log1p + tuple(c for c, flag in (("defensive_action_rate_30m", log_rate), ("cross_match_defensive_rate", log_cm)) if flag and c in continuous)
        spec = V3ModelSpec(
            continuous_cols=continuous, categorical_cols=categorical, interactions=interactions,
            log1p_cols=log1p_cols, missing_indicator_cols=tuple(c for c in MISSING_INDICATOR_COLS if c in continuous),
        )
        for C in C_GRID:
            fitted = fit_v3_model(df_train, "is_goal", spec, kind="ridge", C=C)
            m = score(df_val["is_goal"], predict_v3(fitted, df_val))
            results.append({"log_rate30": log_rate, "log_cm": log_cm, "C": C, "kind": "ridge", **m})
        if always_try_plain:
            try:
                fitted_p = fit_v3_model(df_train, "is_goal", spec, kind="plain")
                m_p = score(df_val["is_goal"], predict_v3(fitted_p, df_val))
                results.append({"log_rate30": log_rate, "log_cm": log_cm, "C": None, "kind": "plain", **m_p})
            except Exception as exc:  # noqa: BLE001 -- statsmodels raises various convergence/separation errors on a singular fit
                results.append({"log_rate30": log_rate, "log_cm": log_cm, "C": None, "kind": "plain", "fit_error": str(exc)})

    valid = [r for r in results if "log_loss" in r]
    best = min(valid, key=lambda r: r["log_loss"])
    print(f"[select] best overall: {best}")

    best_log1p = fixed_log1p + tuple(c for c, flag in (("defensive_action_rate_30m", best["log_rate30"]), ("cross_match_defensive_rate", best["log_cm"])) if flag and c in continuous)
    best_spec = V3ModelSpec(
        continuous_cols=continuous, categorical_cols=categorical, interactions=interactions,
        log1p_cols=best_log1p, missing_indicator_cols=tuple(c for c in MISSING_INDICATOR_COLS if c in continuous),
    )
    ridge_at_best = [r for r in valid if r["kind"] == "ridge" and r["log_rate30"] == best["log_rate30"] and r["log_cm"] == best["log_cm"]]
    best_ridge = min(ridge_at_best, key=lambda r: r["log_loss"])
    plain_at_best = next((r for r in results if r["kind"] == "plain" and r["log_rate30"] == best["log_rate30"] and r["log_cm"] == best["log_cm"]), None)
    plain_ok = plain_at_best is not None and "log_loss" in plain_at_best

    if plain_ok and plain_at_best["log_loss"] <= best_ridge["log_loss"]:
        final_kind, final_C = "plain", None
    else:
        final_kind, final_C = "ridge", best_ridge["C"]
    print(f"[select] final: kind={final_kind} C={final_C} log1p_cols={best_log1p} "
          f"(ridge_val_ll={best_ridge['log_loss']:.5f}, plain_val_ll={plain_at_best.get('log_loss') if plain_at_best else None})")

    fitted = fit_v3_model(df_train, "is_goal", best_spec, kind=final_kind, C=(final_C or 1.0))
    return {
        "grid_results": results, "best_ridge": best_ridge, "plain_at_best": plain_at_best,
        "final_kind": final_kind, "final_C": final_C, "final_spec": best_spec, "final_fitted": fitted,
    }


def run_track(client: bigquery.Client, track: str, df: pd.DataFrame, continuous: tuple, categorical: tuple,
              interactions: tuple, fixed_log1p: tuple, divergence_strata: tuple, now: str) -> dict:
    df_train, df_val, df_test = df[df.split == "train"], df[df.split == "validation"], df[df.split == "test"]
    print(f"[{track}] n={len(df)} train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    selection = select_variant(df_train, df_val, continuous, categorical, interactions, fixed_log1p, always_try_plain=True)
    fitted = selection["final_fitted"]

    pred_rows, metric_rows = [], []
    for split_name, sdf in (("train", df_train), ("validation", df_val), ("test", df_test)):
        preds = predict_v3(fitted, sdf)
        for i, (_, row) in enumerate(sdf.iterrows()):
            pred_rows.append({
                "track": track, "event_id": row["event_id"], "split": split_name,
                "v3_predicted_prob": float(preds[i]),
                "statsbomb_xg": None if pd.isna(row.get("statsbomb_xg")) else float(row["statsbomb_xg"]),
                "is_goal": bool(row["is_goal"]), "materialized_at": now,
            })
        if split_name == "train":
            continue
        m = score(sdf["is_goal"], preds)
        metric_rows.append({"track": track, "split": split_name, "model": "v3", **m, "materialized_at": now})
        xg_valid = sdf.dropna(subset=["statsbomb_xg"])
        if len(xg_valid) > 0:
            m_xg = score(xg_valid["is_goal"], xg_valid["statsbomb_xg"].to_numpy(dtype=float))
            metric_rows.append({"track": track, "split": split_name, "model": "statsbomb_xg", **m_xg, "materialized_at": now})

    coef_df = coefficient_table(fitted)
    coef_rows = [
        {"track": track, "feature": r.feature, "coefficient": float(r.coefficient) if r.coefficient is not None else None,
         "std_error": None if r.std_error is None or pd.isna(r.std_error) else float(r.std_error),
         "p_value": None if r.p_value is None or pd.isna(r.p_value) else float(r.p_value), "materialized_at": now}
        for r in coef_df.itertuples()
    ]

    df_test_pred = df_test.assign(v3_predicted_prob=predict_v3(fitted, df_test))
    div_sub = df_test_pred.dropna(subset=["statsbomb_xg"]).copy()
    div_sub["divergence"] = div_sub["v3_predicted_prob"] - div_sub["statsbomb_xg"]
    div_rows = []
    for stratum_type in divergence_strata:
        sub = div_sub.dropna(subset=[stratum_type])
        for level, g in sub.groupby(stratum_type, observed=True):
            div_rows.append({
                "stratum_type": stratum_type, "stratum_value": str(level), "n": len(g),
                "mean_divergence": float(g["divergence"].mean()), "mean_v3_predicted_prob": float(g["v3_predicted_prob"].mean()),
                "mean_statsbomb_xg": float(g["statsbomb_xg"].mean()), "materialized_at": now,
            })
    if not divergence_strata and len(div_sub):
        # CxG event-wide has no categorical pool member to stratify by (all 8 candidates are
        # continuous) -- v1 baseline's own divergence table was cxg_plus-only for the same
        # structural reason. Write a single "overall" row rather than leaving the table
        # empty, so the overall divergence figure still has a queryable row on record.
        div_rows.append({
            "stratum_type": "overall", "stratum_value": "all", "n": len(div_sub),
            "mean_divergence": float(div_sub["divergence"].mean()), "mean_v3_predicted_prob": float(div_sub["v3_predicted_prob"].mean()),
            "mean_statsbomb_xg": float(div_sub["statsbomb_xg"].mean()), "materialized_at": now,
        })
    overall_corr = float(div_sub["v3_predicted_prob"].corr(div_sub["statsbomb_xg"])) if len(div_sub) > 2 else None
    overall_div = float(div_sub["divergence"].mean()) if len(div_sub) else None

    return {
        "track": track, "selection_summary": {
            "final_kind": selection["final_kind"], "final_C": selection["final_C"],
            "log1p_cols": list(selection["final_spec"].log1p_cols),
            "ridge_val_log_loss": selection["best_ridge"]["log_loss"],
            "plain_val_log_loss": selection["plain_at_best"].get("log_loss") if selection["plain_at_best"] else None,
            "plain_fit_error": selection["plain_at_best"].get("fit_error") if selection["plain_at_best"] and "log_loss" not in selection["plain_at_best"] else None,
        },
        "pred_rows": pred_rows, "metric_rows": metric_rows, "coef_rows": coef_rows, "div_rows": div_rows,
        "divergence_overall": {"n": len(div_sub), "mean_divergence": overall_div, "pearson_corr": overall_corr},
        "grid_results": selection["grid_results"],
    }


def _pred_schema(prob_col: str) -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("split", "STRING", mode="REQUIRED"), bigquery.SchemaField(prob_col, "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("statsbomb_xg", "FLOAT64"), bigquery.SchemaField("is_goal", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]


def _metric_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("split", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model", "STRING", mode="REQUIRED"), bigquery.SchemaField("n", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("log_loss", "FLOAT64"), bigquery.SchemaField("brier_score", "FLOAT64"),
        bigquery.SchemaField("roc_auc", "FLOAT64"), bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]


def _coef_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("feature", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("coefficient", "FLOAT64"), bigquery.SchemaField("std_error", "FLOAT64"),
        bigquery.SchemaField("p_value", "FLOAT64"), bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]


def _div_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("stratum_type", "STRING", mode="REQUIRED"), bigquery.SchemaField("stratum_value", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("n", "INT64", mode="REQUIRED"), bigquery.SchemaField("mean_divergence", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("mean_v3_predicted_prob", "FLOAT64", mode="REQUIRED"), bigquery.SchemaField("mean_statsbomb_xg", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]


def write_track_tables(client: bigquery.Client, track_prefix: str, result: dict) -> None:
    for table, rows, schema in (
        (f"{track_prefix}_v3_predictions", result["pred_rows"], _pred_schema("v3_predicted_prob")),
        (f"{track_prefix}_v3_metrics", result["metric_rows"], _metric_schema()),
        (f"{track_prefix}_v3_coefficients", result["coef_rows"], _coef_schema()),
        (f"{track_prefix}_v3_divergence", result["div_rows"], _div_schema()),
    ):
        ref = f"{PROJECT}.{ML_DATASET}.{table}"
        client.create_table(bigquery.Table(ref, schema=schema), exists_ok=True)
        client.query(f"DELETE FROM `{ref}` WHERE TRUE", location=LOCATION).result()
        if rows:
            client.load_table_from_json(rows, ref, job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
        print(f"[write] {table}: {len(rows)} rows")


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()
    run_id = f"cxg-v3-model-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    df_event = fetch_cxg_event(client)
    result_event = run_track(client, "cxg_event", df_event, CXG_EVENT_CONTINUOUS, (), CXG_EVENT_INTERACTIONS, (), (), now)
    write_track_tables(client, "cxg_event", result_event)

    df_plus = fetch_cxg_plus(client)
    result_plus = run_track(client, "cxg_plus", df_plus, CXG_PLUS_CONTINUOUS, CXG_PLUS_CATEGORICAL, CXG_PLUS_INTERACTIONS,
                             CXG_PLUS_FIXED_LOG1P, ("defensive_profile_cluster", "nearest_defender_style_archetype"), now)
    write_track_tables(client, "cxg_plus", result_plus)

    summary = {
        "run_id": run_id,
        "cxg_event": {k: v for k, v in result_event.items() if k not in ("pred_rows", "metric_rows", "coef_rows", "div_rows")},
        "cxg_plus": {k: v for k, v in result_plus.items() if k not in ("pred_rows", "metric_rows", "coef_rows", "div_rows")},
        "cxg_event_metrics": result_event["metric_rows"],
        "cxg_plus_metrics": result_plus["metric_rows"],
        "cxg_event_coefficients": result_event["coef_rows"],
        "cxg_plus_coefficients": result_plus["coef_rows"],
    }
    out = ROOT / "audit_outputs" / "cxg_analysis" / "v3_model_training" / "v3_model_training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cxg_event", "cxg_plus")}, indent=2, default=str))
    print(json.dumps({"cxg_event_selection": summary["cxg_event"]["selection_summary"], "cxg_plus_selection": summary["cxg_plus"]["selection_summary"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
