"""Shared data-loading, encoding, and metrics helpers for the CxA P_create model
ladder (dumb baseline -> XY baseline v1 -> candidate models), reused by
`materialize_cxa_baseline_v1.py` and `materialize_cxa_candidate_v1.py`. Same
cross-import pattern as `materialize_gold_cxg.py` importing from
`audit_cxg_e13_f1_f15.py`.

Encoding fixes applied here, per docs/analysis/cxa_p_create_locked_feature_eda_v1.md's
"what would break a first baseline model" section -- stated explicitly, not left
implicit:

1. `pass_technique_name = Straight` (CxA+ only, 25 total rows population-wide) is
   pooled into the reference/"other technique" category: only an `Outswinging` dummy
   is created, so `Straight` rows (and every other non-Outswinging technique) collapse
   into the same reference level as an ordinary untagged pass. No separate `Straight`
   column is ever created.
2. `start_x` is clipped to `[0, 120]` before use anywhere (2 event-only rows sit at
   120.7/120.9, a benign corner-kick-position quirk -- see the EDA doc).
3. CxA+'s 154 rows (0.12%) with no computed 360 reception geometry are **not
   dropped** -- they are flagged with an explicit `reception_geometry_missing`
   indicator (same convention as CxG v3's `<col>_was_missing`, see
   `opponent_adjusted/analysis/v3model/modeling.py`) and the two affected continuous
   columns are imputed: `reception_nearest_opponent_distance_m` with the TRAIN split's
   median distance (fit train-only, no leakage), `reception_opponents_within_5m` with
   0 (a defensible "no opponents counted in an uncaptured frame" sentinel, distinct
   from a real observed 0, which is why the missingness flag exists alongside it).
   This choice -- flag + impute, not drop -- keeps all 133,143 rows in the model
   (dropping would discard real labels for 0.12% of the population for a
   data-completeness issue affecting only 3 of the CxA+ feature columns) and is stated
   here rather than left implicit.
4. `pass_body_part_name = No Touch` is excluded entirely from the CxA+ candidate
   feature set (stays held out per
   docs/analysis/cxa_plus_p_create_feature_lock_v1.md). It remains one of the 10
   locked event-only features and IS included in the event-only candidate set.

Feature encoding: each locked categorical feature is one-hot encoded on ONLY its
locked level(s) -- e.g. `pass_type_name` contributes `pass_type_Corner` and
`pass_type_FreeKick` dummies; every other `pass_type_name` value (including the null
"open play" level) collapses into the implicit reference category. This is a
deliberate, stated modelling choice: the candidate feature set is the locked set, not
every raw categorical level that happens to exist on the column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
ML_DATASET = "oam_ml"
LOCATION = "europe-west2"

EVENT_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix`"
PLUS_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix`"
SPLITS_TABLE = f"`{PROJECT}.{ANALYSIS_DATASET}.cxa_match_splits_v1`"

# Only train and validation are ever queried -- test stays sealed (split policy step
# 9-10: freeze and final report come after this comparison is reviewed, not before).
TRAIN_VALIDATION_SPLITS = ("train", "validation")

EVENT_LOAD_SQL = f"""
SELECT
  e.pass_event_id, e.match_id, m.split, e.y_create,
  e.start_x, e.start_y, e.end_x,
  e.is_cross, e.is_through_ball, e.is_switch, e.is_cut_back,
  e.pass_type_name, e.play_pattern_name, e.pass_technique_name, e.pass_body_part_name
FROM {EVENT_MATRIX} e
JOIN {SPLITS_TABLE} m USING (match_id)
WHERE m.split IN UNNEST(@splits)
"""

PLUS_LOAD_SQL = f"""
SELECT
  e.pass_event_id, e.match_id, m.split, e.y_create,
  e.start_x, e.start_y, e.end_x,
  e.is_cross, e.is_through_ball, e.is_switch, e.is_cut_back,
  e.pass_type_name, e.play_pattern_name, e.pass_technique_name,
  e.reception_nearest_opponent_distance_m, e.reception_opponents_within_5m
FROM {PLUS_MATRIX} e
JOIN {SPLITS_TABLE} m USING (match_id)
WHERE m.split IN UNNEST(@splits)
"""


def load_track(client: bigquery.Client, track: str) -> pd.DataFrame:
    sql = EVENT_LOAD_SQL if track == "event" else PLUS_LOAD_SQL
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("splits", "STRING", list(TRAIN_VALIDATION_SPLITS))
        ]
    )
    df = client.query(sql, job_config=job_config, location=LOCATION).to_dataframe()
    df["start_x"] = df["start_x"].clip(lower=0, upper=120)  # encoding fix 2
    return df


def encode_xy(df: pd.DataFrame) -> pd.DataFrame:
    """The deliberately minimal XY baseline (v1) design matrix -- start_x, start_y
    only. NOT the locked feature set."""
    out = pd.DataFrame({"start_x": df["start_x"], "start_y": df["start_y"]}, index=df.index)
    return out


def encode_event_candidate(df: pd.DataFrame) -> pd.DataFrame:
    """Full 10-feature event-only locked set, encoded as 12 numeric columns (2
    numeric + 10 boolean/one-hot dummies across the 4 boolean flags and the 4 locked
    categorical levels across 3 categorical features -- pass_type_name and
    play_pattern_name each contribute 2 locked levels, pass_technique_name and
    pass_body_part_name each contribute 1)."""
    out = pd.DataFrame(index=df.index)
    out["start_x"] = df["start_x"]
    out["end_x"] = df["end_x"]
    out["is_cross"] = df["is_cross"].astype(int)
    out["is_through_ball"] = df["is_through_ball"].astype(int)
    out["is_switch"] = df["is_switch"].astype(int)
    out["is_cut_back"] = df["is_cut_back"].astype(int)
    out["pass_type_Corner"] = (df["pass_type_name"] == "Corner").astype(int)
    out["pass_type_FreeKick"] = (df["pass_type_name"] == "Free Kick").astype(int)
    out["play_pattern_Counter"] = (df["play_pattern_name"] == "From Counter").astype(int)
    out["play_pattern_Corner"] = (df["play_pattern_name"] == "From Corner").astype(int)
    out["pass_technique_Outswinging"] = (df["pass_technique_name"] == "Outswinging").astype(int)
    out["pass_bodypart_NoTouch"] = (df["pass_body_part_name"] == "No Touch").astype(int)
    return out.astype("float64")


def encode_plus_candidate(df: pd.DataFrame, train_median_dist: float) -> pd.DataFrame:
    """Full 9-feature CxA+ locked set (No Touch excluded, Straight pooled into
    reference -- see module docstring), encoded as 13 numeric columns: 4 numeric
    (2 shared-base + 2 reception, both reception columns imputed) + 1 missingness
    flag + 4 boolean + 5 one-hot dummies (2 pass_type + 2 play_pattern +
    1 pass_technique)."""
    out = pd.DataFrame(index=df.index)
    out["start_x"] = df["start_x"]
    out["end_x"] = df["end_x"]
    missing = df["reception_nearest_opponent_distance_m"].isna()
    out["reception_geometry_missing"] = missing.astype(int)
    out["reception_nearest_opponent_distance_m"] = df["reception_nearest_opponent_distance_m"].fillna(
        train_median_dist
    )
    out["reception_opponents_within_5m"] = df["reception_opponents_within_5m"].fillna(0)
    out["is_cross"] = df["is_cross"].astype(int)
    out["is_through_ball"] = df["is_through_ball"].astype(int)
    out["is_switch"] = df["is_switch"].astype(int)
    out["is_cut_back"] = df["is_cut_back"].astype(int)
    out["pass_type_Corner"] = (df["pass_type_name"] == "Corner").astype(int)
    out["pass_type_FreeKick"] = (df["pass_type_name"] == "Free Kick").astype(int)
    out["play_pattern_Counter"] = (df["play_pattern_name"] == "From Counter").astype(int)
    out["play_pattern_Corner"] = (df["play_pattern_name"] == "From Corner").astype(int)
    out["pass_technique_Outswinging"] = (df["pass_technique_name"] == "Outswinging").astype(int)
    return out.astype("float64")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, with_auc: bool = True) -> dict:
    result = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, y_pred, labels=[False, True])),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, y_pred)) if with_auc else None
    return result


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile-binned predicted-vs-actual reliability table. Bins are formed on the
    predicted probability itself (qcut), so bin edges differ per model/split -- this
    is a calibration check (does predicted rank order track actual rate), not a
    fixed-probability-band reliability diagram."""
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    try:
        frame["decile"] = pd.qcut(frame["y_pred"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        frame["decile"] = 0
    grouped = frame.groupby("decile").agg(
        n=("y_true", "size"),
        mean_predicted=("y_pred", "mean"),
        mean_actual=("y_true", "mean"),
    )
    grouped = grouped.reset_index()
    return grouped


def write_table(client: bigquery.Client, table: str, df: pd.DataFrame) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table, job_config=job_config, location=LOCATION)
    job.result()
    print(f"wrote {len(df)} row(s) -> {table}")
