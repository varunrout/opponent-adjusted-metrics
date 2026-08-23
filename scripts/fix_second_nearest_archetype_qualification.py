"""Fix missing/stale qualification for `second_nearest_defender_style_archetype` (CxG+ only).

ROOT CAUSE (Step 1 investigation, documented here rather than assumed):
`materialize_cxg_v2_21st_feature_requalification.py` (Task #11's Step 2) computed this
feature's 20 existing Tier-1 rows using the SAME standard `fit_interaction` function used
for every other pair in this pipeline -- not an ad-hoc/simplified method. Every apparently
"suspect" pattern in those rows is fully explained by ordinary, expected behavior:
  - `interaction_coef`/`interaction_se` are NULL for all 20 rows because this feature is a
    4-level categorical -- `fit_interaction` only reports a single interaction coefficient
    when the interaction term has exactly 1 degree of freedom (`len(interaction) == 1`);
    a 4-level-categorical-vs-anything pair always has >=3 interaction dummy columns, so
    `interaction_coef`/`se` are correctly `None` by that function's own documented contract
    (`testing.py:90-96`) -- the SAME behavior every other categorical pairing in this table
    already exhibits (re-verified live: `nearest_defender_style_archetype`'s own existing
    rows show the identical null pattern).
  - `validated_on_val_split` is NULL for all 20 rows because none of the 20 raw p-values
    crossed the `materialize_cxg_v2_21st_feature_requalification.py`-documented `p_fdr<0.10`
    gate that trigger a validation refit (`21st_feature...:203`) -- re-verified live: the
    smallest of the 20 `interaction_p_fdr` values is 0.382, nowhere near 0.10.
  - The "identical p_fdr across different pairs" pattern (e.g. 0.6577642452990192 shared by
    two DIFFERENT p_raw values, 0.4176 and 0.4211) is the standard Benjamini-Hochberg
    monotonicity-enforcement (cummin) artifact -- re-verified live: sorting all 20 rows by
    `interaction_p_raw` produces a perfectly non-decreasing `interaction_p_fdr` sequence,
    exactly what a correct BH-FDR implementation produces when two ranks are close together
    in the calibration family. Not a bug.

So the 20 existing rows are NOT a computational bug. They ARE, however, now genuinely STALE:
they were tested against the 20-feature CxG+ pool as it stood after Task #11, but the pool
has since grown to 23 (Phase C requalification added `defensive_action_rate_30m`,
`territorial_dominance_last_15m`, `cross_match_defensive_rate`) -- this feature was never
tested against those 3 newest members. That staleness, not a bug, is why this script deletes
and cleanly re-runs Tier 1 for this feature against the full CURRENT pool, rather than
patching the old 20 rows in place.

Separately, and independently of the bivariate question: `cxg_split_univariate_v1` and
`cxg_feature_correlation_v1` genuinely have zero rows for this feature -- that gap is real,
confirmed live, and this script fills it using the REAL established convention for
categoricals in this pipeline (verified live, NOT the task-brief's stated premise): this
project's `cxg_split_univariate_v1` table EXCLUDES every categorical feature by design
(verified live: `nearest_defender_style_archetype`, `second_nearest_defender_role`,
`nearest_defender_role`, `defensive_profile_cluster` all have ZERO rows there too, matching
`materialize_cxg_v2_pool_requalification.py`'s own documented correction of this exact false
premise in an earlier task). The real convention -- per-level goal-rate stability across
splits, reported not written to that table -- is followed here instead, consistent with
every other categorical in this pipeline.
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
from statsmodels.stats.multitest import multipletests

from opponent_adjusted.analysis.bivariate.testing import (
    fit_categorical_interaction_saturated,
    fit_interaction,
    validates_categorical_fallback_on_split,
    validates_on_split,
)

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"
REDUNDANCY_THRESHOLD = 0.85

FEATURE = "second_nearest_defender_style_archetype"

CXG_PLUS_CONTINUOUS_POOL = (
    "cross_match_defensive_rate", "defenders_between_ball_and_goal", "defensive_action_rate_30m",
    "defensive_line_depth", "defensive_reset_index", "defensive_width", "estimated_goalface_occlusion",
    "gk_distance_to_shooter", "goal_mouth_defender_count", "last_action_interval_s", "max_goal_exposure",
    "min_defensive_compactness_sequence", "nearest_defender_distance_delta", "nearest_defender_gap",
    "nearest_defender_zone_displacement", "pre_shot_receiver_space", "shot_corridor_occlusion",
    "territorial_dominance_last_15m", "visible_goal_angle_proxy",
)  # 19
CXG_PLUS_OTHER_CATEGORICAL = ("defensive_profile_cluster", "nearest_defender_role", "second_nearest_defender_role")  # 3, excludes itself


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _fetch_df(client: bigquery.Client, sql: str, params: list | None = None) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
    rows = list(client.query(sql, job_config=job_config, location=LOCATION).result())
    return pd.DataFrame([dict(r.items()) for r in rows])


def _schema(columns: list[tuple[str, str, bool]]) -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(name, sql_type, mode=("REQUIRED" if required else "NULLABLE")) for name, sql_type, required in columns]


def _coerce_row_to_schema(row: dict, schema: list[bigquery.SchemaField]) -> dict:
    out = {}
    type_casters = {"STRING": str, "INT64": int, "FLOAT64": float, "BOOL": bool}
    for field in schema:
        v = row.get(field.name)
        if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            out[field.name] = None
        else:
            out[field.name] = type_casters[field.field_type](v)
    return out


# --- Step 1 (re-verification, printed only) -----------------------------------------------

def reverify_root_cause(client: bigquery.Client) -> dict:
    existing = _fetch_df(client, f"""
        SELECT feature_a, feature_b, interaction_coef, interaction_p_raw, interaction_p_fdr, validated_on_val_split
        FROM {q('cxg_bivariate_interaction_v1')}
        WHERE track = 'cxg_plus' AND tier = 1 AND (feature_a = @f OR feature_b = @f)
        ORDER BY interaction_p_raw
    """, params=[bigquery.ScalarQueryParameter("f", "STRING", FEATURE)])
    monotonic = bool((existing["interaction_p_fdr"].diff().dropna() >= -1e-12).all())
    min_p_fdr = float(existing["interaction_p_fdr"].min())
    other_categorical_coef_null = _fetch_df(client, f"""
        SELECT COUNTIF(interaction_coef IS NULL) AS n_null, COUNT(*) AS n
        FROM {q('cxg_bivariate_interaction_v1')}
        WHERE track = 'cxg_plus' AND tier = 1
          AND (feature_a = 'nearest_defender_style_archetype' OR feature_b = 'nearest_defender_style_archetype')
    """).iloc[0]
    verdict = {
        "n_existing_rows": int(len(existing)),
        "p_fdr_sequence_is_monotonic_bh_consistent": monotonic,
        "min_p_fdr_existing": min_p_fdr,
        "any_existing_row_below_0_10": bool((existing["interaction_p_fdr"] < 0.10).any()),
        "other_4level_categorical_null_coef_rate": f"{other_categorical_coef_null['n_null']}/{other_categorical_coef_null['n']}",
        "conclusion": "no computational bug found -- null coef/se and null validated_on_val_split are expected given "
                      "categorical multi-df interactions and no pair crossing p_fdr<0.10; duplicate p_fdr values are "
                      "a standard BH monotonicity artifact (confirmed monotonic when sorted by p_raw). Rows are stale "
                      "(pool grew 20->23 since Task #11), not buggy -- deleted and re-run for that reason.",
    }
    print(json.dumps(verdict, indent=2, default=str))
    return verdict


# --- Data fetch -----------------------------------------------------------------------------

def fetch_data(client: bigquery.Client) -> pd.DataFrame:
    cont_cols = ", ".join(f"`{c}`" for c in CXG_PLUS_CONTINUOUS_POOL if c not in ("nearest_defender_zone_displacement", "nearest_defender_gap", "defensive_action_rate_30m", "territorial_dominance_last_15m", "cross_match_defensive_rate"))
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {cont_cols} FROM {q('cxg_plus_360_model_matrix_v1')}")

    d = _fetch_df(client, f"""
        SELECT event_id, nearest_defender_zone_displacement, nearest_defender_gap,
               nearest_defender_role, second_nearest_defender_role,
               second_nearest_defender_style_archetype
        FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}
    """)
    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}")
    phase_c = _fetch_df(client, f"""
        SELECT event_id, defensive_action_rate_30m, territorial_dominance_last_15m, cross_match_defensive_rate
        FROM {q('cxg_training_matrix_v1', FEATURE_DATASET)}
    """)

    df = df.merge(d, on="event_id", how="left").merge(cluster, on="event_id", how="left").merge(phase_c, on="event_id", how="left")
    df["defensive_profile_cluster"] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")
    return df


# --- Step 2a: univariate -- real convention is per-level stability, NOT a univariate row ---

def categorical_stability(df: pd.DataFrame) -> list[dict]:
    rows = []
    for split_name, sdf in df.groupby("split"):
        sub = sdf.dropna(subset=[FEATURE])
        for level, g in sub.groupby(FEATURE, observed=True):
            rows.append({"split": split_name, "level": str(level), "n": int(len(g)), "goal_rate": float(g["is_goal"].mean())})
    return rows


# --- Step 2b: correlation/redundancy -- Cramer's V vs the 3 other categoricals only --------

def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    ct = pd.crosstab(a, b)
    n = ct.values.sum()
    if n == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return None, int(n)
    expected = ct.sum(axis=1).to_numpy().reshape(-1, 1) * ct.sum(axis=0).to_numpy().reshape(1, -1) / n
    chi2 = float(((ct.to_numpy() - expected) ** 2 / expected).sum())
    r, k = ct.shape
    denom = min(r - 1, k - 1)
    v = float(np.sqrt(chi2 / (n * denom))) if denom > 0 else None
    return v, int(n)


def materialize_correlation(client: bigquery.Client, df_train: pd.DataFrame, now: str) -> list[dict]:
    client.query(
        f"DELETE FROM {q('cxg_feature_correlation_v1')} WHERE track = 'cxg_plus' AND (feature_a = @f OR feature_b = @f)",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("f", "STRING", FEATURE)]),
        location=LOCATION,
    ).result()

    rows = []
    for other in sorted(CXG_PLUS_OTHER_CATEGORICAL):
        a, b = sorted((FEATURE, other))
        sub = df_train[[FEATURE, other]].dropna()
        v, n = cramers_v(sub[FEATURE], sub[other])
        is_redundant = v is not None and v >= REDUNDANCY_THRESHOLD
        rows.append({
            "track": "cxg_plus", "feature_a": a, "feature_b": b, "r_train": v, "n_train": n,
            "is_redundant": bool(is_redundant),
            "resolution": "dropped_a" if is_redundant else "kept_both_moderate",
            "resolution_reason": (
                f"Cramer's V (categorical-categorical association, NOT Pearson r) = {v:.4f} n={n}. "
                + ("Exceeds redundancy threshold." if is_redundant else "Below the 0.85 redundancy threshold -- kept both.")
            ),
            "materialized_at": now,
        })
    # Continuous pool: documented as not computed, same established convention as
    # materialize_cxg_v2_21st_feature_requalification.py's redundancy_check -- a categorical
    # and a continuous feature structurally cannot be near-duplicates, so no correlation row
    # is written for those pairs (not silently omitted -- explicitly not applicable).
    schema = _schema([
        ("track", "STRING", True), ("feature_a", "STRING", True), ("feature_b", "STRING", True),
        ("r_train", "FLOAT64", False), ("n_train", "INT64", True), ("is_redundant", "BOOL", True),
        ("resolution", "STRING", True), ("resolution_reason", "STRING", False), ("materialized_at", "STRING", True),
    ])
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_correlation_v1",
                                 job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
    print(f"[step2b] wrote {len(rows)} Cramer's V rows: {[(r['feature_a'], r['feature_b'], round(r['r_train'], 4) if r['r_train'] else None) for r in rows]}")
    return rows


def materialize_pool(client: bigquery.Client, redundant_against: set, now: str) -> None:
    if redundant_against:
        print(f"[step2c] NOT adding to pool -- redundant against {redundant_against}")
        return
    client.query(
        f"DELETE FROM {q('cxg_bivariate_candidate_pool_v1')} WHERE track = 'cxg_plus' AND column_name = @f",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("f", "STRING", FEATURE)]),
        location=LOCATION,
    ).result()
    row = {
        "track": "cxg_plus", "feature_family": "opponent_adjusted", "column_name": FEATURE, "data_type": "STRING",
        "qualification_reason": "task11_categorical_backfilled_no_signal_gate_per_no_drop_for_weak_signal_principle",
        "materialized_at": now,
    }
    schema = _schema([
        ("track", "STRING", True), ("feature_family", "STRING", True), ("column_name", "STRING", True),
        ("data_type", "STRING", True), ("qualification_reason", "STRING", True), ("materialized_at", "STRING", True),
    ])
    client.load_table_from_json([row], f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_candidate_pool_v1",
                                 job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
    print(f"[step2c] added {FEATURE} to cxg_bivariate_candidate_pool_v1 (cxg_plus)")


# --- Step 3: bivariate Tier 1, clean delete + full re-run vs the CURRENT pool --------------

def run_tier1(df_train: pd.DataFrame, df_val: pd.DataFrame, now: str) -> list[dict]:
    partners = list(CXG_PLUS_CONTINUOUS_POOL) + list(CXG_PLUS_OTHER_CATEGORICAL)
    rows = []
    for other in partners:
        a, b = sorted((FEATURE, other))
        other_categorical = other in CXG_PLUS_OTHER_CATEGORICAL
        a_categorical = a == FEATURE or a in CXG_PLUS_OTHER_CATEGORICAL
        b_categorical = b == FEATURE or b in CXG_PLUS_OTHER_CATEGORICAL
        result = fit_interaction(df_train, "is_goal", a, b, a_categorical, b_categorical)
        used_fallback = False
        if result.fit_status == "fit_failed" and other_categorical:
            result = fit_categorical_interaction_saturated(df_train, "is_goal", a, b)
            used_fallback = True
        rows.append({
            "track": "cxg_plus", "tier": 1, "feature_a": a, "feature_b": b, "n_train": result.n_train,
            "interaction_coef": result.interaction_coef, "interaction_se": result.interaction_se,
            "interaction_p_raw": result.interaction_p_raw, "interaction_p_fdr": None, "lr_stat": result.lr_stat,
            "main_effect_a_coef": result.main_effect_a_coef, "main_effect_b_coef": result.main_effect_b_coef,
            "validated_on_val_split": None, "fit_status": result.fit_status, "materialized_at": now,
            "_used_fallback": used_fallback, "_other_categorical": other_categorical,
        })
    print(f"[step3] {len(rows)} pairs tested; fit_failed={sum(1 for r in rows if r['fit_status']=='fit_failed')}; fallback_used={sum(1 for r in rows if r['_used_fallback'])}")
    return rows


def calibrate_and_validate(client: bigquery.Client, new_rows: list[dict], df_val: pd.DataFrame) -> list[dict]:
    existing = _fetch_df(client, f"""
        SELECT feature_a, feature_b, interaction_p_raw
        FROM {q('cxg_bivariate_interaction_v1')}
        WHERE track = 'cxg_plus' AND tier = 1 AND interaction_p_raw IS NOT NULL
          AND feature_a != @f AND feature_b != @f
    """, params=[bigquery.ScalarQueryParameter("f", "STRING", FEATURE)])
    all_p = list(existing.interaction_p_raw)
    pair_order = list(zip(existing.feature_a, existing.feature_b))
    for r in new_rows:
        if r["interaction_p_raw"] is not None:
            all_p.append(r["interaction_p_raw"])
            pair_order.append((r["feature_a"], r["feature_b"]))

    _, p_fdr_all, _, _ = multipletests(all_p, method="fdr_bh")
    p_fdr_by_pair = dict(zip(pair_order, p_fdr_all))
    for r in new_rows:
        key = (r["feature_a"], r["feature_b"])
        if key in p_fdr_by_pair:
            r["interaction_p_fdr"] = float(p_fdr_by_pair[key])

    for r in new_rows:
        if r["interaction_p_fdr"] is not None and r["interaction_p_fdr"] < 0.10:
            a, b = r["feature_a"], r["feature_b"]
            a_cat = a == FEATURE or a in CXG_PLUS_OTHER_CATEGORICAL
            b_cat = b == FEATURE or b in CXG_PLUS_OTHER_CATEGORICAL
            if r["_used_fallback"]:
                r["validated_on_val_split"] = validates_categorical_fallback_on_split(df_val, "is_goal", a, b)
            else:
                r["validated_on_val_split"] = validates_on_split(df_val, "is_goal", a, b, r["interaction_coef"], a_cat, b_cat)

    confirmed = [r for r in new_rows if r["validated_on_val_split"] is True]
    print(f"[step3] fdr<0.10={sum(1 for r in new_rows if r['interaction_p_fdr'] is not None and r['interaction_p_fdr']<0.10)}; confirmed={len(confirmed)}")
    for r in confirmed:
        print(f"  CONFIRMED: {r['feature_a']} x {r['feature_b']} p_fdr={r['interaction_p_fdr']:.4g}")
    return new_rows


def write_bivariate(client: bigquery.Client, new_rows: list[dict]) -> None:
    schema = _schema([
        ("track", "STRING", True), ("tier", "INT64", True), ("feature_a", "STRING", True), ("feature_b", "STRING", True),
        ("n_train", "INT64", True), ("interaction_coef", "FLOAT64", False), ("interaction_se", "FLOAT64", False),
        ("interaction_p_raw", "FLOAT64", False), ("interaction_p_fdr", "FLOAT64", False), ("lr_stat", "FLOAT64", False),
        ("main_effect_a_coef", "FLOAT64", False), ("main_effect_b_coef", "FLOAT64", False),
        ("validated_on_val_split", "BOOL", False), ("fit_status", "STRING", True), ("materialized_at", "STRING", True),
    ])
    client.query(
        f"DELETE FROM {q('cxg_bivariate_interaction_v1')} WHERE track = 'cxg_plus' AND tier = 1 AND (feature_a = @f OR feature_b = @f)",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("f", "STRING", FEATURE)]),
        location=LOCATION,
    ).result()
    clean_rows = [_coerce_row_to_schema({k: v for k, v in r.items() if not k.startswith("_")}, schema) for r in new_rows]
    client.load_table_from_json(clean_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_interaction_v1",
                                 job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
    print(f"[step3] deleted 20 stale rows, wrote {len(clean_rows)} fresh rows")


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()

    root_cause = reverify_root_cause(client)

    df = fetch_data(client)
    df_train = df[df.split == "train"]
    df_val = df[df.split == "validation"]
    print(f"[fetch] n={len(df)} train={len(df_train)} val={len(df_val)}")

    stability = categorical_stability(df)
    print(f"[step2a] per-level stability (not written to cxg_split_univariate_v1 -- categoricals excluded by real established convention): {stability}")

    corr_rows = materialize_correlation(client, df_train, now)
    redundant_against = {r["feature_a"] if r["feature_a"] != FEATURE else r["feature_b"] for r in corr_rows if r["is_redundant"]}
    materialize_pool(client, redundant_against, now)

    tier1_rows = run_tier1(df_train, df_val, now)
    tier1_rows = calibrate_and_validate(client, tier1_rows, df_val)
    write_bivariate(client, tier1_rows)

    summary = {
        "root_cause": root_cause,
        "categorical_stability": stability,
        "correlation_rows": corr_rows,
        "redundant_against": sorted(redundant_against),
        "added_to_pool": len(redundant_against) == 0,
        "tier1_pairs_tested": len(tier1_rows),
        "confirmed_pairs": [{"a": r["feature_a"], "b": r["feature_b"], "p_fdr": r["interaction_p_fdr"]} for r in tier1_rows if r["validated_on_val_split"] is True],
    }
    out = ROOT / "audit_outputs" / "cxg_analysis" / "second_nearest_archetype_qualification_fix" / "run_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
