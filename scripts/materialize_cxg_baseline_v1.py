"""Materialize the dumb baseline + v1 kitchen-sink logistic regression (first modeling
stage, CxG + CxG+ tracks) -- deliberately unrefined: full candidate pool, additive only, no
interaction terms, no PCA, ODI trio kept despite weak univariate signal. Feature
trimming/interactions are a separate, later task that will use this v1 as its comparison
point.

Reads candidates from `oam_analysis.cxg_bivariate_candidate_pool_v1` (does not re-derive).
Uses the existing train/validation/test split already on `cxg_event_model_matrix_v1` /
`cxg_plus_360_model_matrix_v1` -- not a new split.

Writes 4 new `oam_ml` tables (INSERT-only, scoped delete-then-insert by track, never
CREATE OR REPLACE):
  - cxg_baseline_v1_predictions   one row per shot per split, both models + statsbomb_xg
  - cxg_baseline_v1_metrics       log_loss/brier/auc per track/split/model
  - cxg_baseline_v1_coefficients  v1's fitted logistic regression coefficients
  - cxg_baseline_v1_divergence    CxG+ only: v1 vs statsbomb_xg divergence by cluster/ODI tercile

statsbomb_xg lives on oam_core.shots and is NOT in Gold -- joined in read-only for this
analysis only, not materialized into oam_features (see report for the explicit call on why).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy import stats as scipy_stats

from opponent_adjusted.analysis.baseline.contracts import (
    CXG_BASELINE_V1_COEFFICIENTS,
    CXG_BASELINE_V1_DIVERGENCE,
    CXG_BASELINE_V1_METRICS,
    CXG_BASELINE_V1_PREDICTIONS,
)
from opponent_adjusted.analysis.baseline.modeling import (
    calibration_table,
    coefficient_table,
    dumb_baseline_prob,
    fit_v1_model,
    predict_v1,
    score_predictions,
)

PROJECT = "oam-varun-260819"
ML_DATASET = "oam_ml"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

CXG_EVENT_POOL = (
    "shot_x_sb",
    "first_box_entry_to_shot_s",
    "last_action_interval_s",
    "last_box_entry_to_shot_s",
    "regain_height_speed_interaction",
)
CXG_PLUS_NUMERIC_POOL = (
    "last_action_interval_s",
    "defenders_between_ball_and_goal",
    "defensive_reset_index",
    "nearest_defender_distance_delta",
    "pre_shot_receiver_space",
    "gk_distance_to_shooter",
    "defensive_line_depth",
    "defensive_width",
    "estimated_goalface_occlusion",
    "goal_mouth_defender_count",
    "max_goal_exposure",
    "min_defensive_compactness_sequence",
    "shot_corridor_occlusion",
    "visible_goal_angle_proxy",
    "gk_odi",
    "mean_backline_odi",
    "nearest_defender_odi",
)
CXG_PLUS_CATEGORICAL = "defensive_profile_cluster"
ODI_TRIO = ("nearest_defender_odi", "gk_odi", "mean_backline_odi")


def q(table: str, dataset: str = ML_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _fetch_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    rows = list(client.query(sql, location=LOCATION).result())
    return pd.DataFrame([dict(r.items()) for r in rows])


def _schema_from_contract(contract) -> list[bigquery.SchemaField]:
    type_map = {"string": "STRING", "int64": "INT64", "float64": "FLOAT64", "bool": "BOOL"}
    return [
        bigquery.SchemaField(c.name, type_map[c.arrow_type], mode=("REQUIRED" if not c.nullable else "NULLABLE"))
        for c in contract.columns
    ]


def _ensure_table(client: bigquery.Client, contract) -> None:
    client.create_table(bigquery.Table(f"{PROJECT}.{ML_DATASET}.{contract.name}", schema=_schema_from_contract(contract)), exists_ok=True)


def _write(client: bigquery.Client, contract, rows: list[dict], track: str) -> None:
    _ensure_table(client, contract)
    client.query(
        f"DELETE FROM {q(contract.name)} WHERE track = @track",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("track", "STRING", track)]),
        location=LOCATION,
    ).result()
    if not rows:
        return
    job_config = bigquery.LoadJobConfig(schema=_schema_from_contract(contract), write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ML_DATASET}.{contract.name}", job_config=job_config, location=LOCATION).result()


def _write_divergence(client: bigquery.Client, rows: list[dict]) -> None:
    contract = CXG_BASELINE_V1_DIVERGENCE
    _ensure_table(client, contract)
    client.query(f"DELETE FROM {q(contract.name)} WHERE stratum_type LIKE 'defensive_profile_cluster' OR stratum_type LIKE 'odi_tercile%%'", location=LOCATION).result()
    if not rows:
        return
    job_config = bigquery.LoadJobConfig(schema=_schema_from_contract(contract), write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ML_DATASET}.{contract.name}", job_config=job_config, location=LOCATION).result()


def verify_step0(client: bigquery.Client) -> None:
    df = _fetch_df(client, f"SELECT track, column_name FROM {q('cxg_bivariate_candidate_pool_v1', ANALYSIS_DATASET)}")
    live_event = set(df[df.track == "cxg_event"].column_name)
    live_plus = set(df[df.track == "cxg_plus"].column_name)
    expected_event = set(CXG_EVENT_POOL)
    expected_plus = set(CXG_PLUS_NUMERIC_POOL) | {CXG_PLUS_CATEGORICAL}
    if live_event != expected_event or live_plus != expected_plus:
        raise SystemExit(
            f"Step 0 FAILED -- candidate pool mismatch.\n cxg_event live={sorted(live_event)} expected={sorted(expected_event)}\n"
            f" cxg_plus live={sorted(live_plus)} expected={sorted(expected_plus)}"
        )
    print(f"[step0] OK -- cxg_event={len(live_event)} cxg_plus={len(live_plus)}, exact match to task spec")


def fetch_cxg_event(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in CXG_EVENT_POOL)
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {cols} FROM {q('cxg_event_model_matrix_v1', ANALYSIS_DATASET)}")
    xg = _fetch_df(client, f"SELECT DISTINCT event_id, statsbomb_xg FROM {q('shots', 'oam_core')}")
    return df.merge(xg, on="event_id", how="left")


def fetch_cxg_plus(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in CXG_PLUS_NUMERIC_POOL if c not in ODI_TRIO)
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {cols} FROM {q('cxg_plus_360_model_matrix_v1', ANALYSIS_DATASET)}")
    odi = _fetch_df(client, f"SELECT event_id, gk_odi, mean_backline_odi, nearest_defender_odi FROM {q('cxg_odi_features_v1', ANALYSIS_DATASET)}")
    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1', ANALYSIS_DATASET)}")
    df = df.merge(odi, on="event_id", how="left").merge(cluster, on="event_id", how="left")
    df[CXG_PLUS_CATEGORICAL] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")
    xg = _fetch_df(client, f"SELECT DISTINCT event_id, statsbomb_xg FROM {q('shots', 'oam_core')}")
    return df.merge(xg, on="event_id", how="left")


def _tercile(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"])


def run_track(client: bigquery.Client, track: str, df: pd.DataFrame, continuous_cols: tuple, categorical_col: str | None, now: str) -> dict:
    df_train = df[df.split == "train"].copy()
    df_val = df[df.split == "validation"].copy()
    df_test = df[df.split == "test"].copy()

    dumb_p = dumb_baseline_prob(df_train, "is_goal")
    fitted = fit_v1_model(df_train, "is_goal", continuous_cols, categorical_col)

    pred_rows, metric_rows = [], []
    split_frames = {"train": df_train, "validation": df_val, "test": df_test}
    for split_name, sdf in split_frames.items():
        v1_pred = predict_v1(fitted, sdf)
        dumb_pred = np.full(len(sdf), dumb_p)
        for i, (_, row) in enumerate(sdf.iterrows()):
            pred_rows.append({
                "track": track,
                "event_id": row["event_id"],
                "split": split_name,
                "dumb_baseline_prob": dumb_p,
                "v1_predicted_prob": float(v1_pred[i]),
                "statsbomb_xg": None if pd.isna(row.get("statsbomb_xg")) else float(row["statsbomb_xg"]),
                "is_goal": bool(row["is_goal"]),
                "materialized_at": now,
            })
        if split_name == "train":
            continue  # metrics reported for validation/test only, per task
        for model_name, pred in (("dumb_baseline", dumb_pred), ("v1", v1_pred)):
            m = score_predictions(sdf["is_goal"], pred)
            metric_rows.append({"track": track, "split": split_name, "model": model_name, **m, "materialized_at": now})
        xg_valid = sdf.dropna(subset=["statsbomb_xg"])
        if len(xg_valid) > 0:
            m = score_predictions(xg_valid["is_goal"], xg_valid["statsbomb_xg"].to_numpy(dtype=float))
            metric_rows.append({"track": track, "split": split_name, "model": "statsbomb_xg", **m, "materialized_at": now})

    coef_df = coefficient_table(fitted)
    coef_rows = [
        {"track": track, "feature": r.feature, "coefficient": float(r.coefficient), "std_error": float(r.std_error), "p_value": float(r.p_value), "materialized_at": now}
        for r in coef_df.itertuples()
    ]

    val_calib = calibration_table(df_val["is_goal"], predict_v1(fitted, df_val))
    val_calib_dumb = calibration_table(df_val["is_goal"], np.full(len(df_val), dumb_p))
    test_calib = calibration_table(df_test["is_goal"], predict_v1(fitted, df_test))
    test_calib_dumb = calibration_table(df_test["is_goal"], np.full(len(df_test), dumb_p))

    return {
        "pred_rows": pred_rows,
        "metric_rows": metric_rows,
        "coef_rows": coef_rows,
        "dumb_prob": dumb_p,
        "calibration": {
            "validation_v1": val_calib.to_dict("records"),
            "validation_dumb": val_calib_dumb.to_dict("records"),
            "test_v1": test_calib.to_dict("records"),
            "test_dumb": test_calib_dumb.to_dict("records"),
        },
        "df_test": df_test.assign(v1_predicted_prob=predict_v1(fitted, df_test)),
    }


def divergence_analysis(df_test_plus: pd.DataFrame, now: str) -> tuple[list[dict], dict]:
    sub = df_test_plus.dropna(subset=["statsbomb_xg"]).copy()
    sub["divergence"] = sub["v1_predicted_prob"] - sub["statsbomb_xg"]

    rows = []
    for cluster, g in sub.groupby(CXG_PLUS_CATEGORICAL, observed=True):
        rows.append({
            "stratum_type": "defensive_profile_cluster", "stratum_value": str(cluster), "n": len(g),
            "mean_divergence": float(g["divergence"].mean()), "mean_v1_prob": float(g["v1_predicted_prob"].mean()),
            "mean_statsbomb_xg": float(g["statsbomb_xg"].mean()), "materialized_at": now,
        })

    for odi_col in ODI_TRIO:
        odi_sub = sub.dropna(subset=[odi_col]).copy()
        if len(odi_sub) < 20:
            continue
        odi_sub["_tercile"] = _tercile(odi_sub[odi_col])
        for tercile, g in odi_sub.groupby("_tercile", observed=True):
            rows.append({
                "stratum_type": f"odi_tercile_{odi_col}", "stratum_value": str(tercile), "n": len(g),
                "mean_divergence": float(g["divergence"].mean()), "mean_v1_prob": float(g["v1_predicted_prob"].mean()),
                "mean_statsbomb_xg": float(g["statsbomb_xg"].mean()), "materialized_at": now,
            })

    overall_corr = float(sub["v1_predicted_prob"].corr(sub["statsbomb_xg"], method="pearson")) if len(sub) > 2 else None
    summary = {"n_with_statsbomb_xg": len(sub), "overall_pearson_corr": overall_corr, "mean_divergence": float(sub["divergence"].mean())}
    return rows, summary


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()

    verify_step0(client)

    df_event = fetch_cxg_event(client)
    df_plus = fetch_cxg_plus(client)
    print(f"[fetch] cxg_event={len(df_event)} rows, cxg_plus={len(df_plus)} rows")

    event_result = run_track(client, "cxg_event", df_event, CXG_EVENT_POOL, None, now)
    plus_numeric = tuple(c for c in CXG_PLUS_NUMERIC_POOL)
    plus_result = run_track(client, "cxg_plus", df_plus, plus_numeric, CXG_PLUS_CATEGORICAL, now)

    # Beats-baseline gate (Step 2's explicit stop condition)
    for track, result in (("cxg_event", event_result), ("cxg_plus", plus_result)):
        val_v1 = next(m for m in result["metric_rows"] if m["split"] == "validation" and m["model"] == "v1")
        val_dumb = next(m for m in result["metric_rows"] if m["split"] == "validation" and m["model"] == "dumb_baseline")
        test_v1 = next(m for m in result["metric_rows"] if m["split"] == "test" and m["model"] == "v1")
        test_dumb = next(m for m in result["metric_rows"] if m["split"] == "test" and m["model"] == "dumb_baseline")
        ok = (
            val_v1["log_loss"] < val_dumb["log_loss"] and val_v1["brier_score"] < val_dumb["brier_score"]
            and test_v1["log_loss"] < test_dumb["log_loss"] and test_v1["brier_score"] < test_dumb["brier_score"]
        )
        print(f"[gate] {track}: v1 beats dumb baseline (val+test, log_loss+brier) = {ok}")
        if not ok:
            print(f"[gate] {track} val_v1={val_v1} val_dumb={val_dumb} test_v1={test_v1} test_dumb={test_dumb}")
            raise SystemExit(f"STOP: v1 does not beat the dumb baseline for track={track!r} -- see printed metrics above.")

    div_rows, div_summary_plus = divergence_analysis(plus_result["df_test"], now)

    # CxG has no opponent-adjustment features, so only the correlation/gap comparison applies
    # (Step 3.1-2) -- no divergence-by-defensive-quality breakdown, per the task's own scope.
    event_xg_sub = event_result["df_test"].dropna(subset=["statsbomb_xg"])
    event_corr = float(event_xg_sub["v1_predicted_prob"].corr(event_xg_sub["statsbomb_xg"], method="pearson")) if len(event_xg_sub) > 2 else None

    for track, result in (("cxg_event", event_result), ("cxg_plus", plus_result)):
        _write(client, CXG_BASELINE_V1_PREDICTIONS, result["pred_rows"], track)
        _write(client, CXG_BASELINE_V1_METRICS, result["metric_rows"], track)
        _write(client, CXG_BASELINE_V1_COEFFICIENTS, result["coef_rows"], track)
    _write_divergence(client, div_rows)

    summary = {
        "cxg_event": {"n_predictions": len(event_result["pred_rows"]), "n_coefficients": len(event_result["coef_rows"]), "statsbomb_xg_corr_test": event_corr},
        "cxg_plus": {"n_predictions": len(plus_result["pred_rows"]), "n_coefficients": len(plus_result["coef_rows"]), "divergence_rows": len(div_rows), **div_summary_plus},
        "calibration": {"cxg_event": event_result["calibration"], "cxg_plus": plus_result["calibration"]},
        "metrics": {"cxg_event": event_result["metric_rows"], "cxg_plus": plus_result["metric_rows"]},
        "coefficients": {"cxg_event": event_result["coef_rows"], "cxg_plus": plus_result["coef_rows"]},
        "divergence": div_rows,
    }
    out_path = ROOT / "audit_outputs" / "cxg_analysis" / "baseline" / "baseline_v1_run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("calibration", "metrics", "coefficients", "divergence")}, indent=2, default=str))


if __name__ == "__main__":
    main()
