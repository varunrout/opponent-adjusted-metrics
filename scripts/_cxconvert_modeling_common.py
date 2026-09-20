"""Shared data-loading, encoding, and metrics helpers for the CxA P_convert model
ladder (dumb baseline -> XY baseline v1 -> candidate models), for BOTH tracks. Mirrors
`_cxa_modeling_common.py`'s (P_create's) structure and conventions.

Encoding fixes applied here, per docs/analysis/cxa_p_convert_locked_feature_eda_v1.md's
"what would break a first baseline model" section -- stated explicitly, not left
implicit:

1. **`shot_technique_name = Lob` on CxA+ (only 16 total train+validation rows) is
   pooled into the reference/baseline category** -- no separate dummy is created for
   CxA+; every CxA+ row (Lob or not) collapses into the implicit `Normal`/other
   reference level for this feature. Event-only keeps `Lob` as its own dummy (62 rows
   there, above the 100-row floor the EDA used... actually just under it, but far less
   thin than CxA+'s 16 -- kept as its own level per the task's explicit instruction to
   keep it for event-only).
2. **`shot_gk_distance_m` nulls are handled with an explicit missingness flag +
   train-only-fit imputation**, not silent mean-imputation: a `shot_gk_distance_missing`
   indicator (1/0) is added, and the null values are imputed with the TRAIN split's own
   median (fit train-only, applied to both train and validation -- no leakage).
3. **The `shot_x_sb` / `shot_dist_to_goal_m` / `shot_gk_distance_m` near-duplicate
   cluster (|r| 0.93-0.97 both tracks, per the EDA) is handled by dropping two of the
   three for the LOGISTIC candidate only**: `shot_gk_distance_m` is kept as the sole
   distance-to-goal representative, `shot_x_sb` and `shot_dist_to_goal_m` are dropped
   from the logistic design matrix. `shot_gk_distance_m` was chosen over the other two
   because the feature-lock docs' own train-vs-validation confirmation showed it was
   the single most stable of the trio in both tracks (event: -5.50 train -> -5.59
   validation, essentially flat; plus: -4.84 -> -4.68, also essentially flat -- neither
   `shot_x_sb` nor `shot_dist_to_goal_m` matched that stability as closely). The TREE
   candidate keeps all three unmodified, per the EDA's own note that tree models
   tolerate this collinearity natively.
4. **No pitch-bounds clipping is applied** -- the EDA found no out-of-range values in
   either track's locked numeric features (a real, checked result, not an omission).

Feature encoding: each locked categorical feature is one-hot encoded on ONLY its
locked level(s) -- consistent with P_create's own convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"

DIST_EXPR = "SQRT(POW(120-shot_x_sb,2)+POW(40-shot_y_sb,2))*(105.0/120.0)"

EVENT_LOAD_SQL = f"""
SELECT
  pass_event_id, split, y_goal,
  start_x, pass_end_x, shot_x_sb, shot_y_sb, {DIST_EXPR} AS shot_dist_to_goal_m,
  shot_gk_distance_m, shot_defenders_within_5m, shot_defenders_within_8m,
  is_through_ball, shot_one_on_one, is_cross, shot_first_time, shot_open_goal,
  shot_technique_name, is_cut_back, pass_type_name
FROM `{PROJECT}.{FEATURES_DATASET}.cxconvert_event_v1_training_matrix`
WHERE split IN UNNEST(@splits)
"""

PLUS_LOAD_SQL = f"""
SELECT
  pass_event_id, split, y_goal,
  start_x, pass_end_x, shot_x_sb, shot_y_sb, {DIST_EXPR} AS shot_dist_to_goal_m,
  shot_gk_distance_m,
  reception_nearest_opponent_distance_m, reception_opponents_within_5m,
  is_through_ball, shot_one_on_one, is_cross, shot_first_time, shot_open_goal,
  shot_technique_name, is_cut_back, pass_type_name
FROM `{PROJECT}.{FEATURES_DATASET}.cxconvert_plus_v1_training_matrix`
WHERE split IN UNNEST(@splits)
"""

TRAIN_VALIDATION_SPLITS = ("train", "validation")


def load_track(client: bigquery.Client, track: str, splits: tuple[str, ...] = TRAIN_VALIDATION_SPLITS) -> pd.DataFrame:
    sql = EVENT_LOAD_SQL if track == "event" else PLUS_LOAD_SQL
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("splits", "STRING", list(splits))]
    )
    return client.query(sql, job_config=job_config, location=LOCATION).to_dataframe()


def encode_xy(df: pd.DataFrame) -> pd.DataFrame:
    """The deliberately minimal XY baseline (v1) design matrix -- shot_x_sb,
    shot_y_sb only (the shot's own location, the P_convert analogue of P_create's
    start_x/start_y pass-origin baseline). NOT the locked feature set."""
    return pd.DataFrame({"shot_x_sb": df["shot_x_sb"], "shot_y_sb": df["shot_y_sb"]}, index=df.index)


def _gk_distance_columns(df: pd.DataFrame, train_median_gk: float, include_flag: bool) -> pd.DataFrame:
    """`include_flag` must be decided from TRAIN's null count, not this split's own --
    a flag column that is constant-zero on train (as happens for the CxA+ track, which
    has 0 null shot_gk_distance_m rows in train+validation) makes the logistic design
    matrix singular (a column with no variation contributes nothing and its Hessian row
    is degenerate). When train has no nulls, the flag is omitted entirely for that
    track rather than kept as a dead, always-zero column -- imputation is still applied
    (harmless no-op when nothing is null), the missingness-safe *mechanism* is present
    for both tracks, but only event-only's null pattern (4 train / 1 validation rows)
    actually needs the flag to carry information."""
    out = pd.DataFrame(index=df.index)
    missing = df["shot_gk_distance_m"].isna()
    if include_flag:
        out["shot_gk_distance_missing"] = missing.astype(int)
    out["shot_gk_distance_m"] = df["shot_gk_distance_m"].fillna(train_median_gk)
    return out


def encode_event_candidate(
    df: pd.DataFrame, train_median_gk: float, *, for_tree: bool, gk_flag_informative: bool = True
) -> pd.DataFrame:
    """Full 15-feature event-only locked set. Logistic drops shot_x_sb/
    shot_dist_to_goal_m (multicollinearity mitigation #3); tree keeps all three."""
    out = pd.DataFrame(index=df.index)
    out["is_through_ball"] = df["is_through_ball"].astype(int)
    out["shot_one_on_one"] = df["shot_one_on_one"].astype(int)
    out["is_cross"] = df["is_cross"].astype(int)
    out["shot_first_time"] = df["shot_first_time"].astype(int)
    out["start_x"] = df["start_x"]
    out["pass_end_x"] = df["pass_end_x"]
    if for_tree:
        out["shot_x_sb"] = df["shot_x_sb"]
        out["shot_dist_to_goal_m"] = df["shot_dist_to_goal_m"]
    out = out.join(_gk_distance_columns(df, train_median_gk, gk_flag_informative))
    out["shot_defenders_within_5m"] = df["shot_defenders_within_5m"]
    out["shot_defenders_within_8m"] = df["shot_defenders_within_8m"]
    out["shot_open_goal"] = df["shot_open_goal"].astype(int)
    out["shot_technique_Lob"] = (df["shot_technique_name"] == "Lob").astype(int)
    out["shot_technique_DivingHeader"] = (df["shot_technique_name"] == "Diving Header").astype(int)
    out["shot_technique_Backheel"] = (df["shot_technique_name"] == "Backheel").astype(int)
    out["shot_technique_Volley"] = (df["shot_technique_name"] == "Volley").astype(int)
    out["is_cut_back"] = df["is_cut_back"].astype(int)
    out["pass_type_Corner"] = (df["pass_type_name"] == "Corner").astype(int)
    return out.astype("float64")


def encode_plus_candidate(
    df: pd.DataFrame, train_median_gk: float, *, for_tree: bool, gk_flag_informative: bool = False
) -> pd.DataFrame:
    """Full 15-feature CxA+ locked set. `shot_technique_name=Lob` pooled into
    reference (encoding fix #1 -- NO dummy created for CxA+, unlike event-only).
    Logistic drops shot_x_sb/shot_dist_to_goal_m; tree keeps all three.
    `gk_flag_informative` defaults False for this track -- CxA+'s train+validation has
    0 null shot_gk_distance_m rows, so the flag would be a constant-zero column (see
    `_gk_distance_columns`'s docstring)."""
    out = pd.DataFrame(index=df.index)
    out["is_through_ball"] = df["is_through_ball"].astype(int)
    out["shot_one_on_one"] = df["shot_one_on_one"].astype(int)
    out["is_cross"] = df["is_cross"].astype(int)
    out["shot_first_time"] = df["shot_first_time"].astype(int)
    out["start_x"] = df["start_x"]
    out["pass_end_x"] = df["pass_end_x"]
    if for_tree:
        out["shot_x_sb"] = df["shot_x_sb"]
        out["shot_dist_to_goal_m"] = df["shot_dist_to_goal_m"]
    out = out.join(_gk_distance_columns(df, train_median_gk, gk_flag_informative))
    out["reception_nearest_opponent_distance_m"] = df["reception_nearest_opponent_distance_m"]
    out["reception_opponents_within_5m"] = df["reception_opponents_within_5m"]
    out["shot_open_goal"] = df["shot_open_goal"].astype(int)
    # shot_technique_name=Lob: pooled into reference, no dummy created (encoding fix #1)
    out["is_cut_back"] = df["is_cut_back"].astype(int)
    out["pass_type_Corner"] = (df["pass_type_name"] == "Corner").astype(int)
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
    return grouped.reset_index()
