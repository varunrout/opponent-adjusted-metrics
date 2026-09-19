"""Statistical qualification for Phase C's 6 rolling-window candidate features (commit
234d0c9), BOTH tracks -- CxG event-wide (`cxg_event`) and CxG+ (`cxg_plus`).

Unlike every prior requalification round (Phase A/B/v2, all CxG+-only), Phase C's features
live on `oam_features.cxg_event_context_features` -- an event-wide table shared by both
tracks -- so this is the first requalification pass that touches the CxG event-wide track's
candidate pool (previously just 5 features: `shot_x_sb`, `last_action_interval_s`,
`last_box_entry_to_shot_s`, `regain_height_speed_interaction`, `first_box_entry_to_shot_s`,
confirmed live from `cxg_bivariate_candidate_pool_v1` before writing this script).

SCOPED ADDITION, not a full pool re-run -- same discipline as every prior requalification
round: only the 6 new features get univariate rows; only pairs involving at least one new
feature get (re)tested for correlation/bivariate; every existing untouched row is left alone
unless a NEW feature is found redundant with an EXISTING one (rare, checked explicitly, see
`_resolve_redundancy`).

TIE-BREAK CONVENTION (redundant pairs, r_train >= 0.85), matching the format found live in
`cxg_feature_correlation_v1` ("X has higher min(|r_train|,|r_val|,|r_test|)") and the
`materialize_cxg_v2_pool_requalification.py` precedent for pairs with no prior-round
resolution to carry forward:
  - new-vs-new redundant pair: keep whichever of the two has the higher min(|point-biserial
    r| across train/val/test) -- i.e. the more universally-informative of the pair, computed
    from this same script's own Step 1 univariate rows. Drop the other.
  - new-vs-existing redundant pair: the existing feature is an already-qualified, possibly
    interaction-confirmed pool member; the newcomer is dropped by default (conservative --
    do not disturb an established member for a late-arriving near-duplicate) unless flagged
    otherwise in the run output for manual review.

Writes (INSERT/scoped-DELETE-then-INSERT, matching `materialize_cxg_v2_pool_requalification.
py`'s documented conventions -- never CREATE OR REPLACE):
  - cxg_split_univariate_v1: scoped INSERT for the 6 new features x 2 tracks x 3 splits.
  - cxg_feature_correlation_v1: scoped to pairs involving >=1 new feature, both tracks.
  - cxg_bivariate_candidate_pool_v1: scoped ADD for new-feature survivors, both tracks
    (existing rows untouched unless an existing feature loses a new-vs-existing redundancy
    tie-break -- reported explicitly if this happens; expected empty).
  - cxg_pca_components_v1 / cxg_pca_loadings_v1: full per-track rewrite (diagnostic-only,
    pool-composition-dependent by construction -- same convention as every prior round).
  - cxg_bivariate_interaction_v1 (tier=1 only): scoped ADD for pairs involving >=1 new
    feature, both tracks; existing tier-1/2/3/4 rows preserved untouched (re-read and
    re-inserted around the scoped delete, same precedent as
    `materialize_cxg_v2_pool_requalification.py`).
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

NEW_FEATURES = (
    "defensive_action_rate_15m",
    "defensive_action_rate_30m",
    "defensive_action_rate_45m",
    "defensive_action_rate_60m",
    "territorial_dominance_last_15m",
    "cross_match_defensive_rate",
)
NEW_FEATURE_FAMILY = "event_context"

CXG_EVENT_EXISTING_POOL = (
    "shot_x_sb",
    "last_action_interval_s",
    "last_box_entry_to_shot_s",
    "regain_height_speed_interaction",
    "first_box_entry_to_shot_s",
)

CXG_PLUS_EXISTING_NUMERIC = (
    "last_action_interval_s", "defenders_between_ball_and_goal", "defensive_reset_index",
    "nearest_defender_distance_delta", "pre_shot_receiver_space", "gk_distance_to_shooter",
    "defensive_line_depth", "defensive_width", "estimated_goalface_occlusion",
    "goal_mouth_defender_count", "max_goal_exposure", "min_defensive_compactness_sequence",
    "shot_corridor_occlusion", "visible_goal_angle_proxy",
)
CXG_PLUS_PHASE_AB_NUMERIC = ("nearest_defender_zone_displacement", "nearest_defender_gap")
CXG_PLUS_EXISTING_POOL = CXG_PLUS_EXISTING_NUMERIC + CXG_PLUS_PHASE_AB_NUMERIC  # 16
CXG_PLUS_CATEGORICAL = (
    "defensive_profile_cluster", "nearest_defender_role",
    "second_nearest_defender_role", "nearest_defender_style_archetype",
)  # 4

TRACKS = {
    "cxg_event": {"existing_continuous": CXG_EVENT_EXISTING_POOL, "categorical": ()},
    "cxg_plus": {"existing_continuous": CXG_PLUS_EXISTING_POOL, "categorical": CXG_PLUS_CATEGORICAL},
}

# Due-diligence-only sanity check (not a formal pool member on either side, same spirit as
# the prior round's `defenders_within_5m/8m` checks): the 15m field-tilt extension vs. its
# frozen 5m parent, same underlying construction, different window.
SANITY_PAIR = ("territorial_dominance_last_15m", "territorial_dominance_last_5m")

PRIOR_CONFIRMED_CXG_PLUS_PAIRS = [
    ("defensive_profile_cluster", "visible_goal_angle_proxy"),
    ("defensive_profile_cluster", "gk_distance_to_shooter"),
    ("pre_shot_receiver_space", "visible_goal_angle_proxy"),
    ("defensive_line_depth", "pre_shot_receiver_space"),
    ("defensive_profile_cluster", "nearest_defender_zone_displacement"),
    ("nearest_defender_gap", "visible_goal_angle_proxy"),
]


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


# --- Step 0: confirm live null-rate state matches Phase C's report -----------------------

def step0_confirm(client: bigquery.Client) -> dict:
    sql = f"""
    SELECT
      has_360_frame,
      COUNT(*) n,
      COUNTIF(defensive_action_rate_15m IS NULL) null_15m,
      COUNTIF(defensive_action_rate_30m IS NULL) null_30m,
      COUNTIF(defensive_action_rate_45m IS NULL) null_45m,
      COUNTIF(defensive_action_rate_60m IS NULL) null_60m,
      COUNTIF(territorial_dominance_last_15m IS NULL) null_tilt,
      COUNTIF(cross_match_defensive_rate IS NULL) null_cm
    FROM {q('cxg_training_matrix_v1', FEATURE_DATASET)}
    GROUP BY has_360_frame
    """
    df = _fetch_df(client, sql)
    result = {bool(r["has_360_frame"]): r for r in df.to_dict("records")}
    plus = result.get(True, {})
    total_n = int(df["n"].sum())
    total_tilt = int(df["null_tilt"].sum())
    total_cm = int(df["null_cm"].sum())
    summary = {
        "event_wide_n": total_n,
        "event_wide_null_tilt": total_tilt,
        "event_wide_null_cm": total_cm,
        "event_wide_null_cm_pct": total_cm / total_n if total_n else None,
        "plus_n": int(plus.get("n", 0)),
        "plus_null_tilt": int(plus.get("null_tilt", 0)),
        "plus_null_cm": int(plus.get("null_cm", 0)),
        "plus_null_cm_pct": (plus.get("null_cm", 0) / plus["n"]) if plus.get("n") else None,
        "rate_windows_all_zero_null": all(int(df[c].sum()) == 0 for c in ("null_15m", "null_30m", "null_45m", "null_60m")),
    }
    expected = {"event_wide_null_tilt": 16, "event_wide_null_cm": 906, "plus_null_tilt": 3, "plus_null_cm": 273}
    discrepancies = {k: (summary[k], v) for k, v in expected.items() if summary[k] != v}
    summary["discrepancies_vs_phase_c_report"] = discrepancies
    if discrepancies or not summary["rate_windows_all_zero_null"]:
        raise RuntimeError(f"Step 0 discrepancy vs Phase C report -- stopping: {discrepancies}")
    print(f"[step0] confirmed: {json.dumps(summary, indent=2, default=str)}")
    return summary


# --- Data fetch -----------------------------------------------------------------------------

def fetch_new_features(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in NEW_FEATURES)
    return _fetch_df(client, f"SELECT event_id, {cols}, territorial_dominance_last_5m FROM {q('cxg_training_matrix_v1', FEATURE_DATASET)}")


def fetch_cxg_event(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in CXG_EVENT_EXISTING_POOL)
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, has_360_frame, {cols} FROM {q('cxg_event_model_matrix_v1')}")
    return df.merge(fetch_new_features(client), on="event_id", how="left")


def fetch_cxg_plus(client: bigquery.Client) -> pd.DataFrame:
    matrix_cols = ", ".join(f"`{c}`" for c in CXG_PLUS_EXISTING_NUMERIC)
    df = _fetch_df(client, f"SELECT event_id, split, is_goal, {matrix_cols} FROM {q('cxg_plus_360_model_matrix_v1')}")
    phase_ab_cols = "nearest_defender_role, nearest_defender_zone_displacement, second_nearest_defender_role, nearest_defender_gap, nearest_defender_style_archetype"
    phase_ab = _fetch_df(client, f"SELECT event_id, {phase_ab_cols} FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}")
    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}")
    df = df.merge(phase_ab, on="event_id", how="left").merge(cluster, on="event_id", how="left")
    df["defensive_profile_cluster"] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")
    return df.merge(fetch_new_features(client), on="event_id", how="left")


# --- Step 1: univariate for the 6 new features, both tracks ------------------------------

def materialize_univariate(client: bigquery.Client, dfs: dict[str, pd.DataFrame], run_id: str, now: str) -> dict:
    client.query(
        f"DELETE FROM {q('cxg_split_univariate_v1')} WHERE track IN UNNEST(@tracks) AND column_name IN UNNEST(@cols)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("tracks", "STRING", list(TRACKS)),
            bigquery.ArrayQueryParameter("cols", "STRING", list(NEW_FEATURES)),
        ]),
        location=LOCATION,
    ).result()

    rows = []
    per_split_r: dict[tuple[str, str, str], float | None] = {}
    for track, df in dfs.items():
        for split_name, sdf in df.groupby("split"):
            for col in NEW_FEATURES:
                sub = sdf[[col, "is_goal"]].dropna()
                non_null = len(sub)
                null_count = len(sdf) - non_null
                goal_count = int(sub["is_goal"].sum()) if non_null else 0
                corr = float(sub[col].corr(sub["is_goal"].astype(float))) if non_null > 2 and sub[col].nunique() > 1 else None
                per_split_r[(track, col, split_name)] = corr
                rows.append({
                    "run_id": run_id, "track": track, "split": split_name,
                    "feature_family": NEW_FEATURE_FAMILY, "column_name": col, "data_type": "FLOAT64",
                    "row_count": len(sdf), "null_count": null_count, "non_null_count": non_null,
                    "null_pct": null_count / len(sdf) if len(sdf) else None,
                    "goal_count": goal_count, "goal_rate": (goal_count / non_null) if non_null else None,
                    "point_biserial_corr": corr, "abs_signal": abs(corr) if corr is not None else None,
                    "materialized_at": now,
                })
    schema = _schema([
        ("run_id", "STRING", True), ("track", "STRING", True), ("split", "STRING", True),
        ("feature_family", "STRING", True), ("column_name", "STRING", True), ("data_type", "STRING", True),
        ("row_count", "INT64", True), ("null_count", "INT64", True), ("non_null_count", "INT64", True),
        ("null_pct", "FLOAT64", False), ("goal_count", "INT64", True), ("goal_rate", "FLOAT64", False),
        ("point_biserial_corr", "FLOAT64", False), ("abs_signal", "FLOAT64", False), ("materialized_at", "TIMESTAMP", False),
    ])
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_split_univariate_v1",
                                 job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
    print(f"[step1] inserted {len(rows)} univariate rows (6 features x 2 tracks x 3 splits)")
    return per_split_r


# --- Step 2: correlation / redundancy screen, scoped to pairs with >=1 new feature -------

def _min_abs_r(per_split_r: dict, track: str, col: str) -> float:
    vals = [abs(per_split_r[(track, col, s)]) for s in ("train", "validation", "test") if per_split_r.get((track, col, s)) is not None]
    return min(vals) if vals else 0.0


def materialize_correlation(client: bigquery.Client, track: str, df_train: pd.DataFrame, existing: tuple, per_split_r: dict, now: str) -> tuple[list[dict], set, set]:
    all_cols = list(existing) + list(NEW_FEATURES)
    corr = df_train[all_cols].corr(numeric_only=True)

    pairs = list(combinations(sorted(NEW_FEATURES), 2)) + [tuple(sorted((n, e))) for n in NEW_FEATURES for e in existing]
    rows = []
    dropped_new: set[str] = set()
    dropped_existing: set[str] = set()
    for a, b in pairs:
        r = corr.loc[a, b]
        n = int(df_train[[a, b]].dropna().shape[0])
        r_val = None if pd.isna(r) else float(r)
        both_new = a in NEW_FEATURES and b in NEW_FEATURES
        if r_val is not None and abs(r_val) >= REDUNDANCY_THRESHOLD:
            is_redundant = True
            if both_new:
                a_min, b_min = _min_abs_r(per_split_r, track, a), _min_abs_r(per_split_r, track, b)
                loser = b if a_min >= b_min else a
                resolution = "dropped_b" if loser == b else "dropped_a"
                reason = (f"NEW pair surfaced by Phase C's enlarged pool: |r_train|={abs(r_val):.4f} >= {REDUNDANCY_THRESHOLD}. "
                          f"Tie-break: kept the feature with the higher min(|point_biserial_corr| across train/val/test) "
                          f"({a}={a_min:.4f} vs {b}={b_min:.4f}); dropped {loser}.")
                dropped_new.add(loser)
            else:
                new_feat = a if a in NEW_FEATURES else b
                existing_feat = b if a in NEW_FEATURES else a
                resolution = "dropped_a" if a == new_feat else "dropped_b"
                reason = (f"NEW-vs-EXISTING redundant pair: |r_train|={abs(r_val):.4f} >= {REDUNDANCY_THRESHOLD}. "
                          f"Conservative default: {existing_feat} is an already-qualified/established pool member, "
                          f"so the late-arriving {new_feat} is dropped rather than disturbing the existing pool.")
                dropped_new.add(new_feat)
        else:
            is_redundant, resolution, reason = False, "kept_both_moderate", None
        rows.append({
            "track": track, "feature_a": a, "feature_b": b, "r_train": r_val, "n_train": n,
            "is_redundant": bool(is_redundant), "resolution": resolution, "resolution_reason": reason,
            "materialized_at": now,
        })

    client.query(
        f"DELETE FROM {q('cxg_feature_correlation_v1')} WHERE track = @track AND (feature_a IN UNNEST(@new) OR feature_b IN UNNEST(@new))",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("track", "STRING", track),
            bigquery.ArrayQueryParameter("new", "STRING", list(NEW_FEATURES)),
        ]),
        location=LOCATION,
    ).result()
    schema = _schema([
        ("track", "STRING", True), ("feature_a", "STRING", True), ("feature_b", "STRING", True),
        ("r_train", "FLOAT64", False), ("n_train", "INT64", True), ("is_redundant", "BOOL", True),
        ("resolution", "STRING", True), ("resolution_reason", "STRING", False), ("materialized_at", "STRING", True),
    ])
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_correlation_v1",
                                 job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
    print(f"[step2] {track}: {len(rows)} new pairs, {sum(r['is_redundant'] for r in rows)} redundant; "
          f"dropped_new={sorted(dropped_new)} dropped_existing={sorted(dropped_existing)}")
    return rows, dropped_new, dropped_existing


def sanity_check(df_train: pd.DataFrame) -> dict:
    a, b = SANITY_PAIR
    sub = df_train[[a, b]].dropna()
    r = float(sub[a].corr(sub[b])) if len(sub) > 2 else None
    return {"feature_a": a, "feature_b": b, "r_train": r, "n_train": int(len(sub))}


def materialize_pool_additions(client: bigquery.Client, track: str, new_survivors: tuple, dropped_existing: set, now: str) -> None:
    if dropped_existing:
        client.query(
            f"DELETE FROM {q('cxg_bivariate_candidate_pool_v1')} WHERE track = @track AND column_name IN UNNEST(@cols)",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("track", "STRING", track),
                bigquery.ArrayQueryParameter("cols", "STRING", sorted(dropped_existing)),
            ]),
            location=LOCATION,
        ).result()
        print(f"[step2] {track}: removed {sorted(dropped_existing)} from candidate pool (lost redundancy tie-break vs a new feature)")

    client.query(
        f"DELETE FROM {q('cxg_bivariate_candidate_pool_v1')} WHERE track = @track AND column_name IN UNNEST(@cols)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("track", "STRING", track),
            bigquery.ArrayQueryParameter("cols", "STRING", list(NEW_FEATURES)),
        ]),
        location=LOCATION,
    ).result()
    rows = [
        {"track": track, "feature_family": NEW_FEATURE_FAMILY, "column_name": col, "data_type": "FLOAT64",
         "qualification_reason": "new_phase_c_candidate_no_signal_gate_per_no_drop_for_weak_signal_principle", "materialized_at": now}
        for col in new_survivors
    ]
    schema = _schema([
        ("track", "STRING", True), ("feature_family", "STRING", True), ("column_name", "STRING", True),
        ("data_type", "STRING", True), ("qualification_reason", "STRING", True), ("materialized_at", "STRING", True),
    ])
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_candidate_pool_v1",
                                 job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION).result()
    print(f"[step2] {track}: added {len(rows)} new pool rows ({sorted(new_survivors)})")


# --- Step 3: PCA, full per-track rewrite over the enlarged pool --------------------------

def materialize_pca(client: bigquery.Client, track: str, df_train: pd.DataFrame, final_numeric: tuple, now: str) -> dict:
    for tbl in ("cxg_pca_components_v1", "cxg_pca_loadings_v1"):
        client.query(f"DELETE FROM {q(tbl)} WHERE track = @track",
                      job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("track", "STRING", track)]),
                      location=LOCATION).result()

    X_raw = df_train[list(final_numeric)].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(X_raw)
    scaler = StandardScaler().fit(imputer.transform(X_raw))
    X = scaler.transform(imputer.transform(X_raw))
    pca = PCA(n_components=min(len(final_numeric), X.shape[0])).fit(X)

    comp_rows, cum = [], 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_):
        cum += float(evr)
        comp_rows.append({"track": track, "component_number": i + 1, "explained_variance_ratio": float(evr), "cumulative_variance_ratio": cum, "materialized_at": now})
    schema1 = _schema([("track", "STRING", True), ("component_number", "INT64", True), ("explained_variance_ratio", "FLOAT64", True), ("cumulative_variance_ratio", "FLOAT64", True), ("materialized_at", "STRING", True)])
    client.load_table_from_json(comp_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_components_v1", job_config=bigquery.LoadJobConfig(schema=schema1, write_disposition="WRITE_APPEND"), location=LOCATION).result()

    n_80 = next((r["component_number"] for r in comp_rows if r["cumulative_variance_ratio"] >= 0.80), len(comp_rows))
    loading_rows = []
    for i in range(min(n_80, len(comp_rows))):
        for j, feat in enumerate(final_numeric):
            loading_rows.append({"track": track, "component_number": i + 1, "feature_name": feat, "loading": float(pca.components_[i, j]), "materialized_at": now})
    schema2 = _schema([("track", "STRING", True), ("component_number", "INT64", True), ("feature_name", "STRING", True), ("loading", "FLOAT64", True), ("materialized_at", "STRING", True)])
    client.load_table_from_json(loading_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_loadings_v1", job_config=bigquery.LoadJobConfig(schema=schema2, write_disposition="WRITE_APPEND"), location=LOCATION).result()

    print(f"[step3] {track}: {len(final_numeric)} features -> {len(comp_rows)} components, {n_80} needed for 80% cumulative variance")
    top_pc1 = sorted([r for r in loading_rows if r["component_number"] == 1], key=lambda r: -abs(r["loading"]))[:5]
    return {"n_components": len(comp_rows), "n_80": n_80, "scree": [r["explained_variance_ratio"] for r in comp_rows], "top_loadings_pc1": top_pc1}


# --- Step 4: Tier 1 bivariate, scoped to pairs with >=1 new feature -----------------------

def run_new_tier1(track: str, df_train: pd.DataFrame, df_val: pd.DataFrame, new_survivors: tuple, existing_survivors: tuple, categorical: tuple, now: str) -> list[dict]:
    pool_partners = existing_survivors + categorical
    pairs = list(combinations(sorted(new_survivors), 2)) + [tuple(sorted((n, p))) for n in new_survivors for p in pool_partners]
    rows = []
    for a, b in pairs:
        a_cat, b_cat = a in categorical, b in categorical
        result = fit_interaction(df_train, "is_goal", a, b, a_cat, b_cat)
        rows.append({
            "track": track, "tier": 1, "feature_a": a, "feature_b": b, "n_train": result.n_train,
            "interaction_coef": result.interaction_coef, "interaction_se": result.interaction_se,
            "interaction_p_raw": result.interaction_p_raw, "interaction_p_fdr": None, "lr_stat": result.lr_stat,
            "main_effect_a_coef": result.main_effect_a_coef, "main_effect_b_coef": result.main_effect_b_coef,
            "validated_on_val_split": None, "fit_status": result.fit_status, "materialized_at": now,
        })
    print(f"[step4] {track}: {len(rows)} new pairs to test ({len(new_survivors)} new features x {len(pool_partners)} pool partners + new-vs-new)")
    return rows


def calibrate_and_validate(client: bigquery.Client, track: str, new_rows: list[dict], df_val: pd.DataFrame, categorical: set) -> list[dict]:
    existing = _fetch_df(client, f"""
        SELECT feature_a, feature_b, interaction_p_raw
        FROM {q('cxg_bivariate_interaction_v1')}
        WHERE track = @track AND tier = 1 AND interaction_p_raw IS NOT NULL
    """, params=[bigquery.ScalarQueryParameter("track", "STRING", track)])
    all_p = list(existing.interaction_p_raw)
    pair_order = list(zip(existing.feature_a, existing.feature_b))
    for r in new_rows:
        if r["interaction_p_raw"] is not None:
            all_p.append(r["interaction_p_raw"])
            pair_order.append((r["feature_a"], r["feature_b"]))

    if all_p:
        _, p_fdr_all, _, _ = multipletests(all_p, method="fdr_bh")
        p_fdr_by_pair = dict(zip(pair_order, p_fdr_all))
        for r in new_rows:
            key = (r["feature_a"], r["feature_b"])
            if key in p_fdr_by_pair:
                r["interaction_p_fdr"] = float(p_fdr_by_pair[key])

    for r in new_rows:
        if r["interaction_p_fdr"] is not None and r["interaction_p_fdr"] < 0.10:
            a, b = r["feature_a"], r["feature_b"]
            r["validated_on_val_split"] = validates_on_split(df_val, "is_goal", a, b, r["interaction_coef"], a in categorical, b in categorical)

    confirmed = [r for r in new_rows if r["validated_on_val_split"] is True]
    print(f"[step4] {track}: fdr<0.10={sum(1 for r in new_rows if r['interaction_p_fdr'] is not None and r['interaction_p_fdr']<0.10)}; confirmed={len(confirmed)}")
    for r in confirmed:
        print(f"  CONFIRMED [{track}]: {r['feature_a']} x {r['feature_b']} p_fdr={r['interaction_p_fdr']:.4g}")
    return new_rows


def _schema_bivariate() -> list[bigquery.SchemaField]:
    return _schema([
        ("track", "STRING", True), ("tier", "INT64", True), ("feature_a", "STRING", True), ("feature_b", "STRING", True),
        ("n_train", "INT64", True), ("interaction_coef", "FLOAT64", False), ("interaction_se", "FLOAT64", False),
        ("interaction_p_raw", "FLOAT64", False), ("interaction_p_fdr", "FLOAT64", False), ("lr_stat", "FLOAT64", False),
        ("main_effect_a_coef", "FLOAT64", False), ("main_effect_b_coef", "FLOAT64", False),
        ("validated_on_val_split", "BOOL", False), ("fit_status", "STRING", True), ("materialized_at", "STRING", True),
    ])


def write_bivariate(client: bigquery.Client, track: str, new_rows: list[dict], dropped_existing: set) -> None:
    schema = _schema_bivariate()
    if dropped_existing:
        client.query(
            f"DELETE FROM {q('cxg_bivariate_interaction_v1')} WHERE track = @track AND tier = 1 AND (feature_a IN UNNEST(@cols) OR feature_b IN UNNEST(@cols))",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("track", "STRING", track),
                bigquery.ArrayQueryParameter("cols", "STRING", sorted(dropped_existing)),
            ]),
            location=LOCATION,
        ).result()
        print(f"[step4] {track}: removed stale tier-1 rows referencing dropped existing features {sorted(dropped_existing)}")
    client.load_table_from_json(
        [_coerce_row_to_schema(r, schema) for r in new_rows],
        f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_interaction_v1",
        job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"), location=LOCATION,
    ).result()
    print(f"[step4] {track}: wrote {len(new_rows)} new tier-1 rows")


def check_prior_confirmed(client: bigquery.Client) -> list[dict]:
    rows = _fetch_df(client, f"""
        SELECT feature_a, feature_b, interaction_p_fdr, validated_on_val_split
        FROM {q('cxg_bivariate_interaction_v1')}
        WHERE track = 'cxg_plus' AND tier = 1
    """)
    by_pair = {frozenset((r["feature_a"], r["feature_b"])): r for r in rows.to_dict("records")}
    out = []
    for a, b in PRIOR_CONFIRMED_CXG_PLUS_PAIRS:
        r = by_pair.get(frozenset((a, b)))
        if r is None:
            out.append({"pair": f"{a}x{b}", "status": "MISSING_removed_from_pool"})
        else:
            survives = r["interaction_p_fdr"] is not None and r["interaction_p_fdr"] < 0.10
            out.append({"pair": f"{a}x{b}", "status": "STILL_CONFIRMED" if (survives and r["validated_on_val_split"]) else "STATUS_CHANGED",
                        "p_fdr": r["interaction_p_fdr"], "validated": r["validated_on_val_split"]})
    return out


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()
    run_id = f"cxg-phase-c-requal-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    step0 = step0_confirm(client)

    df_event = fetch_cxg_event(client)
    df_plus = fetch_cxg_plus(client)
    dfs = {"cxg_event": df_event, "cxg_plus": df_plus}
    print(f"[fetch] cxg_event={len(df_event)} cxg_plus={len(df_plus)}")

    per_split_r = materialize_univariate(client, dfs, run_id, now)

    track_summaries = {}
    for track, spec in TRACKS.items():
        df = dfs[track]
        df_train = df[df.split == "train"]
        df_val = df[df.split == "validation"]
        existing = spec["existing_continuous"]
        categorical = spec["categorical"]

        corr_rows, dropped_new, dropped_existing = materialize_correlation(client, track, df_train, existing, per_split_r, now)
        new_survivors = tuple(f for f in NEW_FEATURES if f not in dropped_new)
        existing_survivors = tuple(f for f in existing if f not in dropped_existing)
        materialize_pool_additions(client, track, new_survivors, dropped_existing, now)

        final_numeric = existing_survivors + new_survivors
        pca_summary = materialize_pca(client, track, df_train, final_numeric, now)

        new_tier1 = run_new_tier1(track, df_train, df_val, new_survivors, existing_survivors, categorical, now)
        new_tier1 = calibrate_and_validate(client, track, new_tier1, df_val, set(categorical))
        write_bivariate(client, track, new_tier1, dropped_existing)

        track_summaries[track] = {
            "dropped_new": sorted(dropped_new), "dropped_existing": sorted(dropped_existing),
            "new_survivors": list(new_survivors), "final_pool": list(final_numeric) + list(categorical),
            "pca_summary": pca_summary,
            "confirmed_new_pairs": [{"a": r["feature_a"], "b": r["feature_b"], "p_fdr": r["interaction_p_fdr"]} for r in new_tier1 if r["validated_on_val_split"] is True],
            "n_new_pairs_tested": len(new_tier1),
        }

    sanity = sanity_check(dfs["cxg_event"][dfs["cxg_event"].split == "train"])
    sanity_plus = sanity_check(dfs["cxg_plus"][dfs["cxg_plus"].split == "train"])
    prior_check = check_prior_confirmed(client)

    summary = {
        "run_id": run_id, "step0": step0, "track_summaries": track_summaries,
        "sanity_pair_cxg_event": sanity, "sanity_pair_cxg_plus": sanity_plus,
        "prior_confirmed_cxg_plus_check": prior_check,
    }
    out_path = ROOT / "audit_outputs" / "cxg_analysis" / "phase_c_requalification" / "run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
