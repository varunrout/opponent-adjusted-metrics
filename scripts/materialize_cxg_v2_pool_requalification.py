"""Re-run the governed analysis chain (Univariate -> Correlation/Redundancy -> PCA ->
Bivariate Tier 1) over the enlarged CxG+ candidate pool, after Phase A (geometric/
categorical defender features) and Phase B (defender-style archetype clustering).

CxG (event-wide) track is entirely untouched by this script -- CxG+ only.

Pool composition for this round:
  - 14 existing post-redundancy-trim CxG+ numeric candidates (unchanged from the prior
    bivariate pool, MINUS the ODI trio -- see below)
  - 2 new continuous candidates (Phase A): nearest_defender_zone_displacement, nearest_defender_gap
  - 4 categorical candidates: defensive_profile_cluster (existing, frozen K-Means),
    nearest_defender_role / second_nearest_defender_role (new, Phase A),
    nearest_defender_style_archetype (new, Phase B)

ODI trio (gk_odi, mean_backline_odi, nearest_defender_odi) is DROPPED from this pool, not
carried forward. This is a real, evidenced methodology decision, not an assumption: Phase
B's own report (audit_outputs/cxg_analysis/phase_b_defender_style_clustering/phase_b_report.md)
states it "replaces the deprecated ODI... defender-quality signal", built specifically
because ODI's original computation consumed `statsbomb_xg` -- circular with this project's
own xG benchmarking (see baseline-v1 task). Verified before writing this script; not
invented.

Table-write conventions used here match what was actually found in the existing scripts on
investigation (not the literal "CREATE OR REPLACE" phrasing the task used, which doesn't
match reality for these three tables):
  - cxg_split_univariate_v1: INSERT-only extension (materialize_cxg_opponent_adjusted_analysis.py's
    pattern), not a full rebuild.
  - cxg_feature_correlation_v1 / cxg_bivariate_candidate_pool_v1 / cxg_pca_components_v1 /
    cxg_pca_loadings_v1: scoped DELETE WHERE track=@track, then WRITE_APPEND (matches
    materialize_cxg_correlation_pca.py exactly).
  - cxg_bivariate_interaction_v1 / cxg_bivariate_stratified_v1: DELETE has no tier
    granularity in the existing script (WHERE track=@track only) -- Tier 2/3/4 rows for
    cxg_plus are read back BEFORE the delete and re-inserted unchanged alongside the fresh
    Tier 1 rows, so nothing is lost despite the coarse delete scope.

Also deliberately NOT following the task's literal instruction that defensive_profile_cluster
has an established "per-level point-biserial in cxg_split_univariate_v1" convention to
replicate for the 3 new categoricals: investigation confirmed this table explicitly EXCLUDES
defensive_profile_cluster ("not eligible for the numeric point-biserial univariate table by
construction", materialize_cxg_opponent_adjusted_analysis.py's own docstring). The real
established convention is: categoricals are excluded from this table, and their signal is
carried via (a) the candidate pool directly, and (b) dummy-encoded bivariate interaction
testing. Followed that real convention for the 3 new categoricals too -- see the report for
the full explanation of this correction and the level-by-level "stability check" run instead.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from opponent_adjusted.analysis.bivariate.testing import fit_interaction, validates_on_split

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"
REDUNDANCY_THRESHOLD = 0.85

# --- Pool definitions -------------------------------------------------------------------

CXG_PLUS_EXISTING_NUMERIC = (
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
)
CXG_PLUS_NEW_CONTINUOUS = ("nearest_defender_zone_displacement", "nearest_defender_gap")
CXG_PLUS_CONTINUOUS_UNION = CXG_PLUS_EXISTING_NUMERIC + CXG_PLUS_NEW_CONTINUOUS  # 16

CXG_PLUS_CATEGORICAL_UNION = (
    "defensive_profile_cluster",
    "nearest_defender_role",
    "second_nearest_defender_role",
    "nearest_defender_style_archetype",
)  # 4

ODI_TRIO_DROPPED = ("gk_odi", "mean_backline_odi", "nearest_defender_odi")

# Overlap-risk pairs the task explicitly asked to check (existing pool member vs. new
# feature, or new feature vs. a NON-pool-member feature computed purely as a due-diligence
# sanity check -- defenders_within_5m/8m were never part of the qualified 18-candidate pool,
# so even a high correlation there doesn't trigger the formal pool-trimming mechanism, but
# is reported honestly either way).
SANITY_CHECK_PAIRS = (
    ("nearest_defender_gap", "defenders_within_5m"),
    ("nearest_defender_gap", "defenders_within_8m"),
    ("nearest_defender_zone_displacement", "defensive_line_depth"),
)

# Precedent redundant pairs from the ORIGINAL correlation/PCA task, carried forward
# unchanged for this round's resolution table (same tie-break reasoning applies regardless
# of pool enlargement -- these pairs' member features haven't changed).
PRIOR_REDUNDANT_PAIRS = [
    ("defensive_reset_index", "rest_defence_reset_fraction", 0.9959, ("rest_defence_reset_fraction",),
     "Carried forward from the original correlation/PCA task: defensive_reset_index has higher min(|r_train|,|r_val|,|r_test|)."),
    ("defensive_compactness", "defensive_hull_area", 0.9167, ("defensive_compactness", "defensive_hull_area"),
     "Carried forward: Phase 2 precedent, drop both, keep the primitive defensive_width."),
    ("gk_distance_to_shooter", "shot_x_sb", -0.8998, ("shot_x_sb",),
     "Carried forward: gk_distance_to_shooter has higher min(|r|) within the 360-cohort; shot_x_sb isn't in this pool anyway (CxG-track-only feature)."),
    ("pre_shot_receiver_space", "shooter_space_previous_linked_event", 0.8624, ("shooter_space_previous_linked_event",),
     "Carried forward: pre_shot_receiver_space is the more primary (direct-measurement) feature per governed taxonomy."),
    ("defensive_centroid_x", "defensive_line_depth", 0.8524, ("defensive_centroid_x",),
     "Carried forward: Phase 2 precedent, kept defensive_line_depth (governed F2 primary depth name)."),
]


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _fetch_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    rows = list(client.query(sql, location=LOCATION).result())
    return pd.DataFrame([dict(r.items()) for r in rows])


def _schema(columns: list[tuple[str, str, bool]]) -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(name, sql_type, mode=("REQUIRED" if required else "NULLABLE")) for name, sql_type, required in columns]


# --- Data fetch --------------------------------------------------------------------------

def fetch_cxg_plus(client: bigquery.Client) -> pd.DataFrame:
    matrix_cols = ", ".join(f"`{c}`" for c in CXG_PLUS_EXISTING_NUMERIC)
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {matrix_cols} FROM {q('cxg_plus_360_model_matrix_v1')}")

    phase_ab_cols = "nearest_defender_role, nearest_defender_zone_displacement, second_nearest_defender_role, nearest_defender_gap, nearest_defender_style_archetype"
    phase_ab = _fetch_df(client, f"SELECT event_id, {phase_ab_cols} FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}")

    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}")

    df = df.merge(phase_ab, on="event_id", how="left").merge(cluster, on="event_id", how="left")
    df["defensive_profile_cluster"] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")

    sanity_cols = ", ".join(f"`{c}`" for c in {"defenders_within_5m", "defenders_within_8m"})
    sanity = _fetch_df(client, f"SELECT event_id, {sanity_cols} FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}")
    df = df.merge(sanity, on="event_id", how="left")
    return df


# --- Step 1: univariate for the 2 new continuous candidates -------------------------------

def materialize_univariate_new_continuous(client: bigquery.Client, df: pd.DataFrame, run_id: str, now: str) -> list[dict]:
    _client_ensure = client
    # Scoped, idempotent: delete any prior rows for exactly these 2 new column_names under
    # track='cxg_plus' before inserting, so re-running this script doesn't duplicate rows --
    # matches materialize_cxg_opponent_adjusted_analysis.py's INSERT-extension pattern
    # (which itself never wipes other features' rows).
    client.query(
        f"DELETE FROM {q('cxg_split_univariate_v1')} WHERE track = 'cxg_plus' AND column_name IN UNNEST(@cols)",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("cols", "STRING", list(CXG_PLUS_NEW_CONTINUOUS))]),
        location=LOCATION,
    ).result()

    rows = []
    for split_name, sdf in df.groupby("split"):
        for col in CXG_PLUS_NEW_CONTINUOUS:
            sub = sdf[[col, "is_goal"]].dropna()
            non_null = len(sub)
            null_count = len(sdf) - non_null
            goal_count = int(sub["is_goal"].sum()) if non_null else 0
            corr = float(sub[col].corr(sub["is_goal"].astype(float))) if non_null > 2 and sub[col].nunique() > 1 else None
            rows.append({
                "run_id": run_id,
                "track": "cxg_plus",
                "split": split_name,
                "feature_family": "opponent_adjusted",
                "column_name": col,
                "data_type": "FLOAT64",
                "row_count": len(sdf),
                "null_count": null_count,
                "non_null_count": non_null,
                "null_pct": null_count / len(sdf) if len(sdf) else None,
                "goal_count": goal_count,
                "goal_rate": (goal_count / non_null) if non_null else None,
                "point_biserial_corr": corr,
                "abs_signal": abs(corr) if corr is not None else None,
                "materialized_at": now,
            })
    schema = _schema([
        ("run_id", "STRING", True), ("track", "STRING", True), ("split", "STRING", True),
        ("feature_family", "STRING", True), ("column_name", "STRING", True), ("data_type", "STRING", True),
        ("row_count", "INT64", True), ("null_count", "INT64", True), ("non_null_count", "INT64", True),
        ("null_pct", "FLOAT64", False), ("goal_count", "INT64", True), ("goal_rate", "FLOAT64", False),
        ("point_biserial_corr", "FLOAT64", False), ("abs_signal", "FLOAT64", False), ("materialized_at", "TIMESTAMP", False),
    ])
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_split_univariate_v1", job_config=job_config, location=LOCATION).result()
    print(f"[step1] inserted {len(rows)} univariate rows for {CXG_PLUS_NEW_CONTINUOUS}")
    return rows


def categorical_stability_check(df: pd.DataFrame) -> list[dict]:
    """Not written to cxg_split_univariate_v1 (that table excludes categoricals by
    established convention -- see module docstring). Computed and reported instead, the
    same descriptive-stability spirit as Phase 2's own categorical qualification
    ('train-only goal-rate separation replicated across splits')."""
    rows = []
    for col in CXG_PLUS_CATEGORICAL_UNION[1:]:  # skip defensive_profile_cluster, already qualified/frozen
        for split_name, sdf in df.groupby("split"):
            sub = sdf.dropna(subset=[col])
            for level, g in sub.groupby(col, observed=True):
                rows.append({
                    "column_name": col, "split": split_name, "level": str(level),
                    "n": len(g), "goal_rate": float(g["is_goal"].mean()),
                })
    return rows


# --- Step 2: correlation / redundancy screen ----------------------------------------------

def materialize_correlation(client: bigquery.Client, df_train: pd.DataFrame, now: str) -> pd.DataFrame:
    client.query(
        f"DELETE FROM {q('cxg_feature_correlation_v1')} WHERE track = 'cxg_plus'", location=LOCATION,
    ).result()

    corr = df_train[list(CXG_PLUS_CONTINUOUS_UNION)].corr(numeric_only=True)
    resolved = {frozenset((a, b)): (r, dropped, reason) for a, b, r, dropped, reason in PRIOR_REDUNDANT_PAIRS}

    rows = []
    uniq = sorted(CXG_PLUS_CONTINUOUS_UNION)
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            r = corr.loc[a, b]
            n = int(df_train[[a, b]].dropna().shape[0])
            r_val = None if pd.isna(r) else float(r)
            key = frozenset((a, b))
            if key in resolved:
                _, dropped, reason = resolved[key]
                is_redundant = True
                resolution = "dropped_both" if len(dropped) == 2 else ("dropped_a" if dropped[0] == a else "dropped_b")
            elif r_val is not None and abs(r_val) >= REDUNDANCY_THRESHOLD:
                is_redundant = True
                resolution = "dropped_b"  # default tie-break: drop the alphabetically-later of a new >=0.85 pair, documented per-pair below
                reason = f"NEW pair surfaced by the enlarged pool (not in prior baseline): |r|={abs(r_val):.4f} >= {REDUNDANCY_THRESHOLD}. Tie-break: dropped {b} (alphabetically later; no stronger precedent applies)."
                dropped = (b,)
            else:
                is_redundant, resolution, reason, dropped = False, "kept_both_moderate", None, ()
            rows.append({
                "track": "cxg_plus", "feature_a": a, "feature_b": b, "r_train": r_val, "n_train": n,
                "is_redundant": bool(is_redundant), "resolution": resolution, "resolution_reason": reason,
                "materialized_at": now,
            })

    schema = _schema([
        ("track", "STRING", True), ("feature_a", "STRING", True), ("feature_b", "STRING", True),
        ("r_train", "FLOAT64", False), ("n_train", "INT64", True), ("is_redundant", "BOOL", True),
        ("resolution", "STRING", True), ("resolution_reason", "STRING", False), ("materialized_at", "STRING", True),
    ])
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_correlation_v1", job_config=job_config, location=LOCATION).result()
    print(f"[step2] {len(rows)} pairs, {sum(r['is_redundant'] for r in rows)} redundant")

    dropped_features: set[str] = set()
    for r in rows:
        if r["is_redundant"] and r["resolution"] != "kept_both_moderate":
            if r["resolution"] == "dropped_both":
                dropped_features.update((r["feature_a"], r["feature_b"]))
            elif r["resolution"] == "dropped_a":
                dropped_features.add(r["feature_a"])
            else:
                dropped_features.add(r["feature_b"])

    final_numeric = tuple(c for c in CXG_PLUS_CONTINUOUS_UNION if c not in dropped_features)
    print(f"[step2] dropped: {sorted(dropped_features)}")
    print(f"[step2] final numeric pool ({len(final_numeric)}): {final_numeric}")
    return pd.DataFrame(rows), final_numeric, dropped_features


def sanity_check_overlap_pairs(df_train: pd.DataFrame) -> list[dict]:
    out = []
    for a, b in SANITY_CHECK_PAIRS:
        sub = df_train[[a, b]].dropna()
        r = float(sub[a].corr(sub[b])) if len(sub) > 2 else None
        out.append({"feature_a": a, "feature_b": b, "r_train": r, "n_train": len(sub)})
    return out


def materialize_pool(client: bigquery.Client, final_numeric: tuple, now: str) -> None:
    client.query(f"DELETE FROM {q('cxg_bivariate_candidate_pool_v1')} WHERE track = 'cxg_plus'", location=LOCATION).result()

    family_lookup = {r["column_name"]: r["feature_family"] for r in _fetch_df(
        client, f"SELECT DISTINCT column_name, feature_family FROM {q('cxg_feature_inventory_v1')} WHERE column_role = 'feature'"
    ).to_dict("records")}
    family_lookup.setdefault("nearest_defender_zone_displacement", "opponent_adjusted")
    family_lookup.setdefault("nearest_defender_gap", "opponent_adjusted")
    family_lookup.setdefault("nearest_defender_role", "opponent_adjusted")
    family_lookup.setdefault("second_nearest_defender_role", "opponent_adjusted")
    family_lookup.setdefault("nearest_defender_style_archetype", "opponent_adjusted")

    rows = []
    for col in final_numeric:
        reason = "univariate_stable" if col in CXG_PLUS_EXISTING_NUMERIC else "new_phase_a_candidate_no_signal_gate_per_no_drop_for_weak_signal_principle"
        rows.append({"track": "cxg_plus", "feature_family": family_lookup.get(col, "unknown"), "column_name": col, "data_type": "FLOAT64", "qualification_reason": reason, "materialized_at": now})
    for col in CXG_PLUS_CATEGORICAL_UNION:
        reason = "categorical_proven_stable" if col == "defensive_profile_cluster" else "new_phase_a_b_categorical_no_signal_gate_per_no_drop_for_weak_signal_principle"
        rows.append({"track": "cxg_plus", "feature_family": family_lookup.get(col, "unknown"), "column_name": col, "data_type": "STRING" if col != "defensive_profile_cluster" else "INT64", "qualification_reason": reason, "materialized_at": now})

    schema = _schema([
        ("track", "STRING", True), ("feature_family", "STRING", True), ("column_name", "STRING", True),
        ("data_type", "STRING", True), ("qualification_reason", "STRING", True), ("materialized_at", "STRING", True),
    ])
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_candidate_pool_v1", job_config=job_config, location=LOCATION).result()
    print(f"[step2] pool: {len(rows)} rows ({len(final_numeric)} numeric + {len(CXG_PLUS_CATEGORICAL_UNION)} categorical)")


# --- Step 3: PCA ---------------------------------------------------------------------------

def materialize_pca(client: bigquery.Client, df_train: pd.DataFrame, final_numeric: tuple, now: str) -> dict:
    for tbl in ("cxg_pca_components_v1", "cxg_pca_loadings_v1"):
        client.query(f"DELETE FROM {q(tbl)} WHERE track = 'cxg_plus'", location=LOCATION).result()

    X_raw = df_train[list(final_numeric)].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(X_raw)
    scaler = StandardScaler().fit(imputer.transform(X_raw))
    X = scaler.transform(imputer.transform(X_raw))
    pca = PCA(n_components=min(len(final_numeric), X.shape[0])).fit(X)

    comp_rows, cum = [], 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_):
        cum += float(evr)
        comp_rows.append({"track": "cxg_plus", "component_number": i + 1, "explained_variance_ratio": float(evr), "cumulative_variance_ratio": cum, "materialized_at": now})
    schema1 = _schema([("track", "STRING", True), ("component_number", "INT64", True), ("explained_variance_ratio", "FLOAT64", True), ("cumulative_variance_ratio", "FLOAT64", True), ("materialized_at", "STRING", True)])
    client.load_table_from_json(comp_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_components_v1", job_config=bigquery.LoadJobConfig(schema=schema1, write_disposition="WRITE_APPEND"), location=LOCATION).result()

    n_80 = next((r["component_number"] for r in comp_rows if r["cumulative_variance_ratio"] >= 0.80), len(comp_rows))
    loading_rows = []
    for i in range(min(n_80, len(comp_rows))):
        for j, feat in enumerate(final_numeric):
            loading_rows.append({"track": "cxg_plus", "component_number": i + 1, "feature_name": feat, "loading": float(pca.components_[i, j]), "materialized_at": now})
    schema2 = _schema([("track", "STRING", True), ("component_number", "INT64", True), ("feature_name", "STRING", True), ("loading", "FLOAT64", True), ("materialized_at", "STRING", True)])
    client.load_table_from_json(loading_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_loadings_v1", job_config=bigquery.LoadJobConfig(schema=schema2, write_disposition="WRITE_APPEND"), location=LOCATION).result()

    print(f"[step3] {len(final_numeric)} features -> {len(comp_rows)} components, {n_80} needed for 80% cumulative variance")
    return {"n_components": len(comp_rows), "n_80": n_80, "scree": [r["explained_variance_ratio"] for r in comp_rows], "top_loadings_pc1": sorted(
        [r for r in loading_rows if r["component_number"] == 1], key=lambda r: -abs(r["loading"])
    )[:5]}


# --- Step 4: bivariate Tier 1 re-run --------------------------------------------------------

PRIOR_CONFIRMED_PAIRS = [
    ("defensive_profile_cluster", "visible_goal_angle_proxy"),
    ("defensive_profile_cluster", "gk_distance_to_shooter"),
    ("pre_shot_receiver_space", "visible_goal_angle_proxy"),
    ("defensive_line_depth", "pre_shot_receiver_space"),
]


def run_tier1(df_train: pd.DataFrame, df_val: pd.DataFrame, pool: tuple, categorical: set, now: str) -> list[dict]:
    rows = []
    for a, b in combinations(sorted(pool), 2):
        a_cat, b_cat = a in categorical, b in categorical
        result = fit_interaction(df_train, "is_goal", a, b, a_cat, b_cat)
        rows.append({
            "track": "cxg_plus", "tier": 1, "feature_a": a, "feature_b": b, "n_train": result.n_train,
            "interaction_coef": result.interaction_coef, "interaction_se": result.interaction_se,
            "interaction_p_raw": result.interaction_p_raw, "interaction_p_fdr": None, "lr_stat": result.lr_stat,
            "main_effect_a_coef": result.main_effect_a_coef, "main_effect_b_coef": result.main_effect_b_coef,
            "validated_on_val_split": None, "fit_status": result.fit_status, "materialized_at": now,
        })
    p_raws = [r["interaction_p_raw"] for r in rows if r["fit_status"] == "ok"]
    if p_raws:
        _, p_fdr, _, _ = multipletests(p_raws, method="fdr_bh")
        it = iter(p_fdr)
        for r in rows:
            if r["fit_status"] == "ok":
                r["interaction_p_fdr"] = float(next(it))
    for r in rows:
        if r["interaction_p_fdr"] is not None and r["interaction_p_fdr"] < 0.10:
            a, b = r["feature_a"], r["feature_b"]
            ok = validates_on_split(df_val, "is_goal", a, b, r["interaction_coef"], a in categorical, b in categorical)
            r["validated_on_val_split"] = ok
    print(f"[step4] tier1: {len(rows)} pairs, fdr<0.10={sum(1 for r in rows if r['interaction_p_fdr'] is not None and r['interaction_p_fdr']<0.10)}")
    return rows


def check_prior_confirmed(tier1_rows: list[dict]) -> list[dict]:
    by_pair = {frozenset((r["feature_a"], r["feature_b"])): r for r in tier1_rows}
    out = []
    for a, b in PRIOR_CONFIRMED_PAIRS:
        r = by_pair.get(frozenset((a, b)))
        if r is None:
            out.append({"pair": f"{a}x{b}", "status": "NOT_RETESTED_missing_from_pool", "detail": None})
        else:
            survives = r["interaction_p_fdr"] is not None and r["interaction_p_fdr"] < 0.10
            validated = r.get("validated_on_val_split")
            out.append({"pair": f"{a}x{b}", "status": "CONFIRMED" if (survives and validated) else "NO_LONGER_CONFIRMED",
                        "detail": {"p_fdr": r["interaction_p_fdr"], "validated": validated}})
    return out


def _coerce_row_to_schema(row: dict, schema: list[bigquery.SchemaField]) -> dict:
    """Rows read back from BigQuery via `_fetch_df` carry numpy/pandas scalar types (and NaN
    for SQL NULL) that `load_table_from_json` cannot serialize directly. Coerce each field to
    a plain JSON-safe Python type per the target schema, exactly preserving NULLs."""
    out = {}
    type_casters = {"STRING": str, "INT64": int, "FLOAT64": float, "BOOL": bool}
    for field in schema:
        v = row.get(field.name)
        if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            out[field.name] = None
        else:
            out[field.name] = type_casters[field.field_type](v)
    return out


def _schema_bivariate() -> list[bigquery.SchemaField]:
    return _schema([
        ("track", "STRING", True), ("tier", "INT64", True), ("feature_a", "STRING", True), ("feature_b", "STRING", True),
        ("n_train", "INT64", True), ("interaction_coef", "FLOAT64", False), ("interaction_se", "FLOAT64", False),
        ("interaction_p_raw", "FLOAT64", False), ("interaction_p_fdr", "FLOAT64", False), ("lr_stat", "FLOAT64", False),
        ("main_effect_a_coef", "FLOAT64", False), ("main_effect_b_coef", "FLOAT64", False),
        ("validated_on_val_split", "BOOL", False), ("fit_status", "STRING", True), ("materialized_at", "STRING", True),
    ])


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()
    run_id = f"cxg-v2-requal-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    df = fetch_cxg_plus(client)
    df_train = df[df.split == "train"]
    df_val = df[df.split == "validation"]
    print(f"[fetch] cxg_plus={len(df)} rows (train={len(df_train)} val={len(df_val)} test={(df.split=='test').sum()})")

    # Step 1
    materialize_univariate_new_continuous(client, df, run_id, now)
    cat_stability = categorical_stability_check(df)

    # Step 2
    corr_df, final_numeric, dropped = materialize_correlation(client, df_train, now)
    sanity_pairs = sanity_check_overlap_pairs(df_train)
    materialize_pool(client, final_numeric, now)

    # Step 3
    pca_summary = materialize_pca(client, df_train, final_numeric, now)

    # Step 4: preserve existing Tier 2/3/4 rows before the coarse track-scoped delete
    existing_tier234 = _fetch_df(client, f"SELECT * FROM {q('cxg_bivariate_interaction_v1')} WHERE track = 'cxg_plus' AND tier != 1")
    existing_stratified = _fetch_df(client, f"SELECT * FROM {q('cxg_bivariate_stratified_v1')} WHERE track = 'cxg_plus'")

    final_pool = final_numeric + CXG_PLUS_CATEGORICAL_UNION
    categorical_set = set(CXG_PLUS_CATEGORICAL_UNION)
    tier1_rows = run_tier1(df_train, df_val, final_pool, categorical_set, now)
    prior_check = check_prior_confirmed(tier1_rows)

    client.query(f"DELETE FROM {q('cxg_bivariate_interaction_v1')} WHERE track = 'cxg_plus'", location=LOCATION).result()
    schema = _schema_bivariate()
    preserved_rows = [_coerce_row_to_schema(r, schema) for r in existing_tier234.to_dict("records")]
    all_rows = tier1_rows + preserved_rows
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    client.load_table_from_json(all_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_interaction_v1", job_config=job_config, location=LOCATION).result()
    print(f"[step4] wrote {len(tier1_rows)} fresh tier1 + {len(existing_tier234)} preserved tier2/3/4 = {len(all_rows)} total cxg_plus rows")

    # cxg_bivariate_stratified_v1 untouched by Tier 1 (Tier1 has no stratified table rows) --
    # re-affirm preservation explicitly by not issuing any DELETE against it at all.
    print(f"[step4] cxg_bivariate_stratified_v1: {len(existing_stratified)} existing cxg_plus rows left untouched (no delete issued)")

    summary = {
        "run_id": run_id,
        "n_fetched": len(df),
        "dropped_redundant": sorted(dropped),
        "final_numeric_pool": list(final_numeric),
        "final_categorical_pool": list(CXG_PLUS_CATEGORICAL_UNION),
        "sanity_check_pairs": sanity_pairs,
        "pca_summary": pca_summary,
        "tier1_pair_count": len(tier1_rows),
        "prior_confirmed_check": prior_check,
        "categorical_stability_sample": cat_stability[:12],
    }
    out_path = ROOT / "audit_outputs" / "cxg_analysis" / "v2_pool_requalification" / "v2_run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("categorical_stability_sample",)}, indent=2, default=str))

    # Full categorical stability table saved separately (can be large)
    (out_path.parent / "categorical_stability_full.json").write_text(json.dumps(cat_stability, indent=2, default=str))


if __name__ == "__main__":
    main()
