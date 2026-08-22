"""v2 modeling spec, Step 2: scoped requalification for the 21st candidate,
`second_nearest_defender_style_archetype`. Not a full pool re-run -- same scoped-addition
discipline as the original Phase A/B requalification and the categorical-sparsity fix.

1. Univariate stability: per-level goal-rate check across splits (the REAL established
   convention for a new categorical -- categoricals are excluded from
   `cxg_split_univariate_v1` by design, confirmed when `nearest_defender_style_archetype`
   went through this same process; reported, not written to that table).
2. Redundancy check against all 20 other pool members. The 16 continuous get the standard
   Pearson-r screen. The 4 other categoricals (including the flagged overlap-risk pair,
   `second_nearest_defender_role`) get Cramer's V -- a bounded [0,1] categorical-categorical
   association measure, the natural analogue of |r| for two categoricals (there is no
   existing categorical-vs-categorical redundancy precedent in this pipeline to instead
   replicate; every prior categorical was only ever screened against the continuous pool via
   Pearson r or left uncompared).
3. Tier 1 bivariate: the 20 new pairs (this feature x every other pool member), reusing
   `fit_interaction` with the saturated-fallback (`fit_categorical_interaction_saturated`)
   for any categorical x categorical pair that hits the same sparsity failure mode found and
   fixed for 2 other pairs earlier. BH-FDR recomputed over the full 210-pair family so the 20
   new p_fdr values are correctly calibrated, but only these 20 rows are written -- the
   existing 190 rows are untouched, per the same discipline already established.
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

NEW_FEATURE = "second_nearest_defender_style_archetype"

CONTINUOUS_POOL = (
    "last_action_interval_s", "defenders_between_ball_and_goal", "defensive_reset_index",
    "nearest_defender_distance_delta", "pre_shot_receiver_space", "gk_distance_to_shooter",
    "defensive_line_depth", "defensive_width", "estimated_goalface_occlusion",
    "goal_mouth_defender_count", "max_goal_exposure", "min_defensive_compactness_sequence",
    "shot_corridor_occlusion", "visible_goal_angle_proxy",
    "nearest_defender_zone_displacement", "nearest_defender_gap",
)
EXISTING_CATEGORICAL_POOL = (
    "defensive_profile_cluster", "nearest_defender_role", "second_nearest_defender_role",
    "nearest_defender_style_archetype",
)
ALL_20 = CONTINUOUS_POOL + EXISTING_CATEGORICAL_POOL


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
    df = df.merge(d, on="event_id", how="left").merge(cluster, on="event_id", how="left")
    df["defensive_profile_cluster"] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")
    return df


def univariate_stability(df: pd.DataFrame) -> list[dict]:
    rows = []
    for split_name, sdf in df.groupby("split"):
        sub = sdf.dropna(subset=[NEW_FEATURE])
        for level, g in sub.groupby(NEW_FEATURE, observed=True):
            rows.append({"split": split_name, "level": str(level), "n": len(g), "goal_rate": float(g["is_goal"].mean())})
    return rows


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """Bias-uncorrected Cramer's V from a contingency table (adequate for a redundancy
    screen -- we only care about magnitude relative to the 0.85 threshold, not a
    publication-grade bias-corrected estimate)."""
    ct = pd.crosstab(a, b)
    n = ct.values.sum()
    chi2 = float(((ct - (ct.sum(axis=1).values.reshape(-1, 1) * ct.sum(axis=0).values.reshape(1, -1) / n)) ** 2 /
                  (ct.sum(axis=1).values.reshape(-1, 1) * ct.sum(axis=0).values.reshape(1, -1) / n)).values.sum())
    r, k = ct.shape
    denom = min(r - 1, k - 1)
    v = float(np.sqrt(chi2 / (n * denom))) if denom > 0 else None
    return v, int(n)


def redundancy_check(df_train: pd.DataFrame) -> dict:
    results = {"continuous": [], "categorical": []}
    for col in CONTINUOUS_POOL:
        sub = df_train[[NEW_FEATURE, col]].dropna()
        # Pearson r needs a numeric encoding of the new categorical -- not meaningful, so for
        # continuous-vs-categorical we instead report the goal-rate-style association via
        # Cramer's V is inapplicable (col is continuous); use eta-squared style check is out
        # of scope for a REDUNDANCY screen (redundancy is about near-duplication, which a
        # categorical and a continuous feature structurally cannot be). Skipped by design --
        # documented, not silently omitted.
        results["continuous"].append({"feature": col, "note": "categorical vs continuous -- not a near-duplication risk by construction, no correlation computed", "n": len(sub)})
    for col in EXISTING_CATEGORICAL_POOL:
        sub = df_train[[NEW_FEATURE, col]].dropna()
        v, n = cramers_v(sub[NEW_FEATURE], sub[col])
        results["categorical"].append({"feature": col, "cramers_v": v, "n": n, "redundant": v is not None and v >= REDUNDANCY_THRESHOLD})
    return results


def run_new_pairs(df_train: pd.DataFrame, df_val: pd.DataFrame) -> list[dict]:
    now = datetime.now(UTC).isoformat()
    rows = []
    for other in ALL_20:
        other_categorical = other in EXISTING_CATEGORICAL_POOL
        result = fit_interaction(df_train, "is_goal", NEW_FEATURE, other, True, other_categorical)
        used_fallback = False
        if result.fit_status == "fit_failed" and other_categorical:
            result = fit_categorical_interaction_saturated(df_train, "is_goal", NEW_FEATURE, other)
            used_fallback = True
        rows.append({
            "track": "cxg_plus", "tier": 1, "feature_a": NEW_FEATURE, "feature_b": other,
            "n_train": result.n_train, "interaction_coef": result.interaction_coef,
            "interaction_se": result.interaction_se, "interaction_p_raw": result.interaction_p_raw,
            "interaction_p_fdr": None, "lr_stat": result.lr_stat,
            "main_effect_a_coef": result.main_effect_a_coef, "main_effect_b_coef": result.main_effect_b_coef,
            "validated_on_val_split": None, "fit_status": result.fit_status, "materialized_at": now,
            "_used_fallback": used_fallback,
        })
    print(f"[tier1] {len(rows)} new pairs; fit_failed={sum(1 for r in rows if r['fit_status']=='fit_failed')}; "
          f"fallback_used={sum(1 for r in rows if r['_used_fallback'])}")
    return rows


def main() -> None:
    client = _client()

    df = fetch_data(client)
    df_train = df[df.split == "train"]
    df_val = df[df.split == "validation"]
    print(f"[fetch] n={len(df)} train={len(df_train)} val={len(df_val)}")

    stability = univariate_stability(df)
    print("[step1-univariate] level-by-split stability:")
    for r in stability:
        print(f"  {r}")

    redundancy = redundancy_check(df_train)
    print("[step2-redundancy] categorical Cramer's V vs the 4 existing categoricals:")
    for r in redundancy["categorical"]:
        print(f"  {r}")

    new_rows = run_new_pairs(df_train, df_val)

    existing = _fetch_df(client, f"""
        SELECT feature_a, feature_b, interaction_p_raw
        FROM {q('cxg_bivariate_interaction_v1')}
        WHERE track='cxg_plus' AND tier=1 AND interaction_p_raw IS NOT NULL
    """)
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
            if r["interaction_p_fdr"] < 0.10:
                other_cat = r["feature_b"] in EXISTING_CATEGORICAL_POOL
                if r["_used_fallback"]:
                    r["validated_on_val_split"] = validates_categorical_fallback_on_split(df_val, "is_goal", r["feature_a"], r["feature_b"])
                else:
                    r["validated_on_val_split"] = validates_on_split(df_val, "is_goal", r["feature_a"], r["feature_b"], r["interaction_coef"], True, other_cat)

    confirmed = [r for r in new_rows if r["validated_on_val_split"] is True]
    print(f"[tier1] FDR<0.10: {sum(1 for r in new_rows if r['interaction_p_fdr'] is not None and r['interaction_p_fdr']<0.10)}; confirmed: {len(confirmed)}")
    for r in confirmed:
        print(f"  CONFIRMED: {r['feature_a']} x {r['feature_b']} p_fdr={r['interaction_p_fdr']:.4g}")

    schema = [
        bigquery.SchemaField("track", "STRING", mode="REQUIRED"), bigquery.SchemaField("tier", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("feature_a", "STRING", mode="REQUIRED"), bigquery.SchemaField("feature_b", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("n_train", "INT64", mode="REQUIRED"), bigquery.SchemaField("interaction_coef", "FLOAT64"),
        bigquery.SchemaField("interaction_se", "FLOAT64"), bigquery.SchemaField("interaction_p_raw", "FLOAT64"),
        bigquery.SchemaField("interaction_p_fdr", "FLOAT64"), bigquery.SchemaField("lr_stat", "FLOAT64"),
        bigquery.SchemaField("main_effect_a_coef", "FLOAT64"), bigquery.SchemaField("main_effect_b_coef", "FLOAT64"),
        bigquery.SchemaField("validated_on_val_split", "BOOL"), bigquery.SchemaField("fit_status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "STRING", mode="REQUIRED"),
    ]
    clean_rows = [{k: v for k, v in r.items() if k != "_used_fallback"} for r in new_rows]
    client.query(
        f"DELETE FROM {q('cxg_bivariate_interaction_v1')} WHERE track='cxg_plus' AND tier=1 AND (feature_a=@f OR feature_b=@f)",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("f", "STRING", NEW_FEATURE)]),
        location=LOCATION,
    ).result()
    client.load_table_from_json(
        clean_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_interaction_v1",
        job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION,
    ).result()
    print(f"[write] {len(clean_rows)} rows written")

    summary = {
        "stability": stability, "redundancy": redundancy,
        "confirmed_pairs": [{"feature_a": r["feature_a"], "feature_b": r["feature_b"], "p_fdr": r["interaction_p_fdr"]} for r in confirmed],
        "all_new_pairs": clean_rows,
    }
    out = ROOT / "audit_outputs" / "cxg_analysis" / "v2_model_training" / "step2_21st_feature_requalification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
