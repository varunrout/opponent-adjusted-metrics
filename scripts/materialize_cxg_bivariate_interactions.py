"""Materialize bivariate interaction testing (CxG + CxG+ tracks) -- final analysis stage:
Summary Statistics -> EDA -> Univariate -> Correlation/Redundancy Screen -> PCA -> **Bivariate**.

Reads the candidate pool from `oam_analysis.cxg_bivariate_candidate_pool_v1` (does not
re-derive it). Writes `cxg_bivariate_interaction_v1` (one row per tested pair, every tier,
including non-significant negatives -- never silently dropped) and
`cxg_bivariate_stratified_v1` (cross-tab goal-rate tables backing Tier 2/3 headline charts).

Tiers:
  1. Exhaustive within-pool pairwise (CxG: C(5,2)=10, CxG+: C(18,2)=153), BH-FDR corrected
     per track.
  2. CxG+ only, targeted: ODI trio x match-context features from cxg_event_context_features.
  3. CxG+ only, targeted: defensive_profile_cluster vs. shot geometry (shot_distance_sb /
     shot_angle_rad computed inline from shot_x_sb/shot_y_sb -- cxg_shot_geometry_features
     has no populated feature columns, confirmed by a prior investigation this session; see
     report). Also investigates the "suspiciously identical null-cluster coordinates"
     data-quality question directly.
  4. CxG+ only: cross-pool compounding -- reuses Tier 1's already-computed
     (last_action_interval_s, X) results for X = top CxG+-exclusive performers by
     abs_signal, re-tagged tier=4 (no re-fit; same data, same model, different narrative
     framing per the task's explicit instruction).
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
from statsmodels.stats.multitest import multipletests

from opponent_adjusted.analysis.bivariate.contracts import (
    CXG_BIVARIATE_INTERACTION_V1,
    CXG_BIVARIATE_STRATIFIED_V1,
)
from opponent_adjusted.analysis.bivariate.testing import fit_interaction, validates_on_split

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"

MATCH_CONTEXT_FEATURES = ["score_diff", "game_state", "late_game_leading", "late_game_trailing", "manpower_diff"]
MATCH_CONTEXT_CATEGORICAL = {"game_state"}


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
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
    table_ref = f"{PROJECT}.{ANALYSIS_DATASET}.{contract.name}"
    client.create_table(bigquery.Table(table_ref, schema=_schema_from_contract(contract)), exists_ok=True)


def shot_angle_rad(x: pd.Series, y: pd.Series) -> pd.Series:
    """Angle subtended by the goal mouth (posts at x=120, y=36/44) from (x, y), radians.
    Standard football-analytics shot-angle definition, via the law-of-cosines vector-angle
    formula (robust for shots behind/beside the goal line, unlike the atan2-ratio shortcut)."""
    v1x, v1y = 120 - x, 36 - y
    v2x, v2y = 120 - x, 44 - y
    dot = v1x * v2x + v1y * v2y
    n1 = np.sqrt(v1x**2 + v1y**2)
    n2 = np.sqrt(v2x**2 + v2y**2)
    cos_angle = (dot / (n1 * n2)).clip(-1.0, 1.0)
    return np.arccos(cos_angle)


def shot_distance_sb(x: pd.Series, y: pd.Series) -> pd.Series:
    return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)


# --- Step 0: candidate pool ---

CXG_EVENT_POOL = (
    "shot_x_sb",
    "first_box_entry_to_shot_s",
    "last_action_interval_s",
    "last_box_entry_to_shot_s",
    "regain_height_speed_interaction",
)

# The 18 CxG+ candidates, split by how each is sourced (all confirmed present on
# cxg_plus_360_model_matrix_v1 except the ODI trio (cxg_odi_features_v1, join on event_id) and
# defensive_profile_cluster (cxg_defensive_profile_clusters_v1, join on event_id).
CXG_PLUS_MATRIX_NUMERIC = (
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
CXG_PLUS_ODI_TRIO = ("gk_odi", "mean_backline_odi", "nearest_defender_odi")
CXG_PLUS_CATEGORICAL = ("defensive_profile_cluster",)
CXG_PLUS_POOL = CXG_PLUS_MATRIX_NUMERIC + CXG_PLUS_CATEGORICAL + CXG_PLUS_ODI_TRIO
assert len(CXG_EVENT_POOL) == 5
assert len(CXG_PLUS_POOL) == 18


def verify_step0(client: bigquery.Client) -> None:
    df = _fetch_df(client, f"SELECT track, column_name FROM {q('cxg_bivariate_candidate_pool_v1')}")
    live_event = set(df[df.track == "cxg_event"].column_name)
    live_plus = set(df[df.track == "cxg_plus"].column_name)
    expected_event, expected_plus = set(CXG_EVENT_POOL), set(CXG_PLUS_POOL)
    if live_event != expected_event or live_plus != expected_plus:
        raise SystemExit(
            "Step 0 FAILED -- live cxg_bivariate_candidate_pool_v1 does not match the task's given "
            f"5/18 split.\n  cxg_event live={sorted(live_event)}\n  cxg_event expected={sorted(expected_event)}\n"
            f"  cxg_plus live={sorted(live_plus)}\n  cxg_plus expected={sorted(expected_plus)}"
        )
    print(f"[step0] OK -- cxg_event={len(live_event)} cxg_plus={len(live_plus)}, exact match to task spec")


# --- data fetching ---

def fetch_cxg_event(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in CXG_EVENT_POOL)
    return _fetch_df(client, f"SELECT event_id, split, is_goal, {cols} FROM {q('cxg_event_model_matrix_v1')}")


def fetch_cxg_plus(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in CXG_PLUS_MATRIX_NUMERIC)
    df = _fetch_df(
        client,
        f"SELECT event_id, split, is_goal, shot_x_sb, shot_y_sb, {cols} FROM {q('cxg_plus_360_model_matrix_v1')}",
    )
    odi = _fetch_df(
        client,
        f"SELECT event_id, gk_odi, mean_backline_odi, nearest_defender_odi FROM {q('cxg_odi_features_v1')}",
    )
    cluster = _fetch_df(
        client,
        f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}",
    )
    df = df.merge(odi, on="event_id", how="left").merge(cluster, on="event_id", how="left")
    # Null cluster_label is itself a meaningful category (geometry-ineligible shots, Tier 3's
    # subject) -- not missing data to impute or drop, so it gets its own explicit string level.
    df["defensive_profile_cluster"] = df["cluster_label"].apply(
        lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}"
    )
    df["shot_distance_sb"] = shot_distance_sb(df["shot_x_sb"], df["shot_y_sb"])
    df["shot_angle_rad"] = shot_angle_rad(df["shot_x_sb"], df["shot_y_sb"])
    return df


def fetch_match_context(client: bigquery.Client) -> pd.DataFrame:
    cols = ", ".join(f"`{c}`" for c in MATCH_CONTEXT_FEATURES)
    df = _fetch_df(client, f"SELECT event_id, {cols} FROM {q('cxg_event_context_features', FEATURE_DATASET)}")
    df["late_game_leading"] = df["late_game_leading"].astype("float")
    df["late_game_trailing"] = df["late_game_trailing"].astype("float")
    return df


# --- row construction ---

def _row(track, tier, a, b, result, now, validated=None) -> dict:
    return {
        "track": track,
        "tier": tier,
        "feature_a": a,
        "feature_b": b,
        "n_train": result.n_train,
        "interaction_coef": result.interaction_coef,
        "interaction_se": result.interaction_se,
        "interaction_p_raw": result.interaction_p_raw,
        "interaction_p_fdr": None,
        "lr_stat": result.lr_stat,
        "main_effect_a_coef": result.main_effect_a_coef,
        "main_effect_b_coef": result.main_effect_b_coef,
        "validated_on_val_split": validated,
        "fit_status": result.fit_status,
        "materialized_at": now,
    }


def run_tier1(df_train: pd.DataFrame, df_val: pd.DataFrame, track: str, pool: tuple, categorical: set, tier: int, now: str) -> list[dict]:
    rows = []
    for a, b in combinations(sorted(pool), 2):
        a_cat, b_cat = a in categorical, b in categorical
        result = fit_interaction(df_train, "is_goal", a, b, a_cat, b_cat)
        rows.append(_row(track, tier, a, b, result, now))
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
            a_cat, b_cat = a in categorical, b in categorical
            ok = validates_on_split(df_val, "is_goal", a, b, r["interaction_coef"], a_cat, b_cat)
            r["validated_on_val_split"] = ok
    return rows


def run_tier2(df_train: pd.DataFrame, df_val: pd.DataFrame, now: str) -> tuple[list[dict], list[dict]]:
    rows = []
    for odi in CXG_PLUS_ODI_TRIO:
        for ctx in MATCH_CONTEXT_FEATURES:
            ctx_cat = ctx in MATCH_CONTEXT_CATEGORICAL
            result = fit_interaction(df_train, "is_goal", odi, ctx, False, ctx_cat)
            rows.append(_row("cxg_plus", 2, odi, ctx, result, now))
    for r in rows:
        if r["fit_status"] == "ok" and r["interaction_p_raw"] is not None and r["interaction_p_raw"] < 0.05:
            a, b = r["feature_a"], r["feature_b"]
            b_cat = b in MATCH_CONTEXT_CATEGORICAL
            ok = validates_on_split(df_val, "is_goal", a, b, r["interaction_coef"], False, b_cat)
            r["validated_on_val_split"] = ok

    ok_rows = [r for r in rows if r["fit_status"] == "ok" and r["interaction_p_raw"] is not None]
    headline = min(ok_rows, key=lambda r: r["interaction_p_raw"]) if ok_rows else None
    strat_rows = []
    if headline is not None:
        strat_rows = stratify(df_train, "cxg_plus", 2, headline["feature_a"], headline["feature_b"], now)
    return rows, strat_rows


def _tercile(series: pd.Series) -> pd.Series:
    """Rank-based tercile split (low/mid/high) that always produces exactly 3 groups, even
    under heavy value-ties (e.g. gk_odi is ~94% exactly 0.0 -- a plain `pd.qcut` on the raw
    values fails there with duplicate bin edges and silently falls through to one stratum per
    distinct float, producing dozens of near-singleton rows instead of 3 clean groups).
    Ties are broken by row order (`rank(method="first")`), so which side of a tercile boundary
    a tied value falls on is arbitrary -- documented in the report wherever this matters."""
    ranks = series.rank(method="first")
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"])


def stratify(df: pd.DataFrame, track: str, tier: int, a_col: str, b_col: str, now: str) -> list[dict]:
    sub = df[["is_goal", a_col, b_col]].dropna().copy()
    sub["_a"] = _tercile(sub[a_col]) if (a_col not in MATCH_CONTEXT_CATEGORICAL and a_col != "defensive_profile_cluster") else sub[a_col].astype(str)
    sub["_b"] = _tercile(sub[b_col]) if (b_col not in MATCH_CONTEXT_CATEGORICAL and b_col != "defensive_profile_cluster") else sub[b_col].astype(str)

    rows = []
    for (sa, sb), g in sub.groupby(["_a", "_b"], observed=True):
        n = len(g)
        goal_count = int(g["is_goal"].sum())
        rows.append(
            {
                "track": track,
                "tier": tier,
                "feature_a": a_col,
                "feature_b": b_col,
                "stratum_a": str(sa),
                "stratum_b": str(sb),
                "n": n,
                "goal_count": goal_count,
                "goal_rate": (goal_count / n) if n else None,
                "materialized_at": now,
            }
        )
    return rows


def fit_joint_significance(df: pd.DataFrame, y_col: str, covariate_cols: list[str], cat_col: str):
    """Nested-model LR test: does adding `cat_col`'s dummies to a model already containing
    `covariate_cols` significantly improve fit? Used by Tier 3 (cluster dummies vs. shot
    geometry controls) -- a joint-significance test, not a two-way interaction, so it does not
    reuse `fit_interaction`'s additive-vs-interaction shape."""
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.tools.tools import add_constant

    sub = df[[y_col, *covariate_cols, cat_col]].dropna().copy()
    n = len(sub)
    if n < 50 or sub[y_col].nunique() < 2 or sub[cat_col].nunique() < 2:
        return {"n": n, "lr_stat": None, "p_raw": None, "fit_status": "insufficient_data"}

    y = sub[y_col].astype(int)
    X_cov = sub[covariate_cols].astype(float)
    dummies = pd.get_dummies(sub[cat_col], prefix="cluster", drop_first=True).astype(float)

    X_reduced = add_constant(X_cov, has_constant="add")
    X_full = add_constant(pd.concat([X_cov, dummies], axis=1), has_constant="add")
    try:
        m_reduced = Logit(y, X_reduced).fit(disp=0, maxiter=100)
        m_full = Logit(y, X_full).fit(disp=0, maxiter=100)
    except Exception:
        return {"n": n, "lr_stat": None, "p_raw": None, "fit_status": "fit_failed"}
    if not (np.isfinite(m_reduced.llf) and np.isfinite(m_full.llf)):
        return {"n": n, "lr_stat": None, "p_raw": None, "fit_status": "fit_failed"}

    from scipy import stats as scipy_stats

    lr_stat = float(2 * (m_full.llf - m_reduced.llf))
    dof = X_full.shape[1] - X_reduced.shape[1]
    if lr_stat < 0 or dof <= 0:
        return {"n": n, "lr_stat": None, "p_raw": None, "fit_status": "fit_failed"}
    p_raw = float(scipy_stats.chi2.sf(lr_stat, dof))
    return {"n": n, "lr_stat": lr_stat, "p_raw": p_raw, "fit_status": "ok"}


def run_tier3(df_train: pd.DataFrame, now: str) -> tuple[list[dict], list[dict], dict]:
    # --- data-quality investigation: "suspiciously identical close-range coordinates" ---
    null_shots = df_train[df_train["defensive_profile_cluster"] == "null_cluster"]
    coord_counts = null_shots.groupby(["shot_x_sb", "shot_y_sb"]).size().sort_values(ascending=False)
    top_coord = coord_counts.index[0] if len(coord_counts) else None
    top_coord_n = int(coord_counts.iloc[0]) if len(coord_counts) else 0
    dq = {
        "n_null_cluster_shots": int(len(null_shots)),
        "n_distinct_coordinate_pairs": int(len(coord_counts)),
        "goal_rate": float(null_shots["is_goal"].mean()) if len(null_shots) else None,
        "top_coordinate": tuple(float(v) for v in top_coord) if top_coord is not None else None,
        "top_coordinate_share": (top_coord_n / len(null_shots)) if len(null_shots) else None,
        "mean_shot_distance_sb": float(null_shots["shot_distance_sb"].mean()) if len(null_shots) else None,
        "std_shot_distance_sb": float(null_shots["shot_distance_sb"].std()) if len(null_shots) else None,
    }

    # --- statistical test: cluster dummies jointly significant after controlling for geometry ---
    result = fit_joint_significance(df_train, "is_goal", ["shot_distance_sb", "shot_angle_rad"], "defensive_profile_cluster")
    row = {
        "track": "cxg_plus",
        "tier": 3,
        "feature_a": "defensive_profile_cluster",
        "feature_b": "shot_distance_sb+shot_angle_rad",
        "n_train": result["n"],
        "interaction_coef": None,
        "interaction_se": None,
        "interaction_p_raw": result["p_raw"],
        "interaction_p_fdr": None,
        "lr_stat": result["lr_stat"],
        "main_effect_a_coef": None,
        "main_effect_b_coef": None,
        "validated_on_val_split": None,
        "fit_status": result["fit_status"],
        "materialized_at": now,
    }

    strat_rows = stratify(df_train, "cxg_plus", 3, "defensive_profile_cluster", "shot_distance_sb", now)
    return [row], strat_rows, dq


CROSS_POOL_CXG_PLUS_EXCLUSIVE_TOP = ("visible_goal_angle_proxy", "gk_distance_to_shooter", "defensive_reset_index")


def run_tier4(tier1_rows: list[dict], now: str) -> list[dict]:
    """Cross-pool: last_action_interval_s (CxG's own top univariate performer restricted to
    features also present in the CxG+ pool -- shot_x_sb excluded per the task's explicit
    consistency rule, since it was redundancy-trimmed from the final CxG+ pool) against the
    top CxG+-exclusive performers by abs_signal with adequate non-null support. These exact
    pairs are already exhaustively covered by Tier 1 (both features are CxG+ pool members) --
    per the task's own framing this is a re-tagging/re-narrating of existing Tier-1 fits under
    the cross-pool lens, not a second independent fit (would be a pointless duplicate model)."""
    by_pair = {frozenset((r["feature_a"], r["feature_b"])): r for r in tier1_rows if r["track"] == "cxg_plus"}
    rows = []
    for other in CROSS_POOL_CXG_PLUS_EXCLUSIVE_TOP:
        key = frozenset(("last_action_interval_s", other))
        src = by_pair.get(key)
        if src is None:
            continue
        r = dict(src)
        r["tier"] = 4
        rows.append(r)
    return rows


# --- write ---

def _write(client: bigquery.Client, contract, rows: list[dict], track_filter: str | None = None) -> None:
    if not rows:
        return
    _ensure_table(client, contract)
    if track_filter is not None:
        client.query(
            f"DELETE FROM {q(contract.name)} WHERE track = @track",
            job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("track", "STRING", track_filter)]),
            location=LOCATION,
        ).result()
    job_config = bigquery.LoadJobConfig(schema=_schema_from_contract(contract), write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.{contract.name}", job_config=job_config, location=LOCATION).result()


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()

    verify_step0(client)

    df_event = fetch_cxg_event(client)
    df_plus = fetch_cxg_plus(client)
    df_ctx = fetch_match_context(client)
    df_plus = df_plus.merge(df_ctx, on="event_id", how="left")
    print(f"[fetch] cxg_event={len(df_event)} rows, cxg_plus={len(df_plus)} rows")

    ev_train, ev_val = df_event[df_event.split == "train"], df_event[df_event.split == "validation"]
    pl_train, pl_val = df_plus[df_plus.split == "train"], df_plus[df_plus.split == "validation"]
    print(f"[split] cxg_event train={len(ev_train)} val={len(ev_val)}; cxg_plus train={len(pl_train)} val={len(pl_val)}")

    tier1_event = run_tier1(ev_train, ev_val, "cxg_event", CXG_EVENT_POOL, set(), 1, now)
    tier1_plus = run_tier1(pl_train, pl_val, "cxg_plus", CXG_PLUS_POOL, set(CXG_PLUS_CATEGORICAL), 1, now)
    print(f"[tier1] cxg_event pairs={len(tier1_event)} survivors(fdr<0.10)={sum(1 for r in tier1_event if r['interaction_p_fdr'] is not None and r['interaction_p_fdr']<0.10)}")
    print(f"[tier1] cxg_plus pairs={len(tier1_plus)} survivors(fdr<0.10)={sum(1 for r in tier1_plus if r['interaction_p_fdr'] is not None and r['interaction_p_fdr']<0.10)}")

    tier2_rows, tier2_strat = run_tier2(pl_train, pl_val, now)
    print(f"[tier2] pairs={len(tier2_rows)} raw_p<0.05={sum(1 for r in tier2_rows if r['interaction_p_raw'] is not None and r['interaction_p_raw']<0.05)}")

    tier3_rows, tier3_strat, dq = run_tier3(pl_train, now)
    print(f"[tier3] data_quality={json.dumps(dq, default=str)}")
    print(f"[tier3] joint_significance_row={tier3_rows[0]}")

    tier4_rows = run_tier4(tier1_plus, now)
    print(f"[tier4] pairs={len(tier4_rows)} (reused from tier1, re-tagged)")

    all_interaction_rows = tier1_event + tier1_plus + tier2_rows + tier3_rows + tier4_rows
    all_strat_rows = tier2_strat + tier3_strat

    for track in ("cxg_event", "cxg_plus"):
        _write(client, CXG_BIVARIATE_INTERACTION_V1, [r for r in all_interaction_rows if r["track"] == track], track_filter=track)
        _write(client, CXG_BIVARIATE_STRATIFIED_V1, [r for r in all_strat_rows if r["track"] == track], track_filter=track)

    summary = {
        "cxg_event_tier1_pairs": len(tier1_event),
        "cxg_plus_tier1_pairs": len(tier1_plus),
        "tier2_pairs": len(tier2_rows),
        "tier3_rows": len(tier3_rows),
        "tier4_rows": len(tier4_rows),
        "total_interaction_rows": len(all_interaction_rows),
        "total_stratified_rows": len(all_strat_rows),
        "tier3_data_quality": dq,
    }
    print(json.dumps(summary, indent=2, default=str))
    out_path = ROOT / "audit_outputs" / "cxg_analysis" / "bivariate" / "bivariate_run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
