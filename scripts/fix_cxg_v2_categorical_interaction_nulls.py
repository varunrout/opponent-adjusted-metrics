"""Scoped fix for the 2 CxG+ Tier 1 rows with NULL `interaction_p_fdr`:
`defensive_profile_cluster x nearest_defender_role` and
`nearest_defender_role x second_nearest_defender_role`.

Root cause (confirmed via live cross-tab, not assumed -- see the report addendum): both are
categorical x categorical pairs whose standard interaction-model MLE fit fails on sparse/
structurally-empty cross-tab cells. `nearest_defender_role x second_nearest_defender_role`
has a literal zero-count GK x GK cell (a team has one goalkeeper, so it can never be both a
shot's nearest and second-nearest defender) -- a rank-deficient interaction design matrix,
not incidental sparsity. Several other cells (e.g. `cluster_0 x GK`, n=4, 0 goals;
`Fullback_WingBack x Fullback_WingBack`, n=16, 0 goals) drive complete/quasi-complete
separation. Reproduced directly: `overflow in exp` / `divide by zero in log` from
statsmodels' unregularized MLE.

Fix: `fit_categorical_interaction_saturated` (new function,
`src/opponent_adjusted/analysis/bivariate/testing.py`) -- a saturated-vs-additive deviance
test whose saturated half has a closed-form log-likelihood (empirical per-cell goal rate),
so it never needs to solve a system that could be singular. Applied ONLY to the 2 affected
rows via a scoped UPDATE (never touching any other Tier 1 row's already-valid p-value).

Checked for other silently-affected categorical x categorical pairs first (all 6 exist in
the pool): the other 4 (`defensive_profile_cluster x nearest_defender_style_archetype`,
`defensive_profile_cluster x second_nearest_defender_role`, `nearest_defender_role x
nearest_defender_style_archetype`, `nearest_defender_style_archetype x
second_nearest_defender_role`) all have `fit_status='ok'` with valid p-values already --
confirmed via direct query, not assumed -- so they are left untouched.

BH-FDR discipline: the 2 new raw p-values are inserted into the FULL 190-pair cxg_plus Tier
1 raw-p-value vector (the other 188 already have a p_fdr computed from that same batch) so
their own p_fdr reflects the correct correction stringency for a 190-test family -- but only
these 2 rows are written back; the other 188 rows are not touched, per the task's explicit
constraint, even though a bit-for-bit "recompute everything from scratch" would very slightly
re-rank a handful of them (an unavoidable, explicitly-accepted tradeoff, documented here).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests

from opponent_adjusted.analysis.bivariate.testing import (
    fit_categorical_interaction_saturated,
    validates_categorical_fallback_on_split,
)

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"

BROKEN_PAIRS = [
    ("defensive_profile_cluster", "nearest_defender_role"),
    ("nearest_defender_role", "second_nearest_defender_role"),
]


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _fetch_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    rows = list(client.query(sql, location=LOCATION).result())
    return pd.DataFrame([dict(r.items()) for r in rows])


def fetch_cxg_plus_categoricals(client: bigquery.Client) -> pd.DataFrame:
    df = _fetch_df(client, f"SELECT event_id, split, is_goal FROM {q('cxg_plus_360_model_matrix_v1')}")
    phase_ab = _fetch_df(
        client,
        f"SELECT event_id, nearest_defender_role, second_nearest_defender_role FROM {q('cxg_defensive_360_features', FEATURE_DATASET)}",
    )
    cluster = _fetch_df(client, f"SELECT event_id, cluster_label FROM {q('cxg_defensive_profile_clusters_v1')}")
    df = df.merge(phase_ab, on="event_id", how="left").merge(cluster, on="event_id", how="left")
    df["defensive_profile_cluster"] = df["cluster_label"].apply(lambda v: "null_cluster" if pd.isna(v) else f"cluster_{int(v)}")
    return df


def confirm_no_other_silent_failures(client: bigquery.Client) -> None:
    cats = {"defensive_profile_cluster", "nearest_defender_role", "second_nearest_defender_role", "nearest_defender_style_archetype"}
    df = _fetch_df(
        client,
        f"SELECT feature_a, feature_b, fit_status, interaction_p_fdr FROM {q('cxg_bivariate_interaction_v1')} WHERE track='cxg_plus' AND tier=1",
    )
    catcat = df[df.feature_a.isin(cats) & df.feature_b.isin(cats)]
    print(f"[check] {len(catcat)} categorical x categorical pairs in the pool:")
    for _, r in catcat.iterrows():
        print(f"  {r.feature_a} x {r.feature_b}: fit_status={r.fit_status} p_fdr={r.interaction_p_fdr}")
    broken = catcat[catcat.fit_status == "fit_failed"]
    broken_pairs_found = set(zip(broken.feature_a, broken.feature_b)) | set(zip(broken.feature_b, broken.feature_a))
    expected = set(BROKEN_PAIRS) | {(b, a) for a, b in BROKEN_PAIRS}
    if broken_pairs_found != expected:
        raise SystemExit(f"STOP: broken-pair set mismatch. Found {broken_pairs_found}, expected {expected}.")
    null_p_fdr = df[df.interaction_p_fdr.isna()]
    if len(null_p_fdr) != 2:
        raise SystemExit(f"STOP: expected exactly 2 NULL interaction_p_fdr rows, found {len(null_p_fdr)}.")
    print("[check] confirmed: exactly the 2 known pairs are broken, no other silent failures.")


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()

    confirm_no_other_silent_failures(client)

    df = fetch_cxg_plus_categoricals(client)
    df_train = df[df.split == "train"]
    df_val = df[df.split == "validation"]

    new_results = {}
    for a, b in BROKEN_PAIRS:
        result = fit_categorical_interaction_saturated(df_train, "is_goal", a, b)
        print(f"[fit] {a} x {b}: fit_status={result.fit_status} p_raw={result.interaction_p_raw} lr_stat={result.lr_stat} n_train={result.n_train}")
        new_results[(a, b)] = result

    if any(r.fit_status != "ok_saturated_fallback" for r in new_results.values()):
        raise SystemExit("STOP: saturated fallback also failed for at least one pair -- do not proceed silently.")

    existing = _fetch_df(
        client,
        f"SELECT feature_a, feature_b, interaction_p_raw FROM {q('cxg_bivariate_interaction_v1')} "
        f"WHERE track='cxg_plus' AND tier=1 AND interaction_p_raw IS NOT NULL",
    )
    all_p = list(existing.interaction_p_raw)
    pair_order = list(zip(existing.feature_a, existing.feature_b))
    for (a, b), r in new_results.items():
        all_p.append(r.interaction_p_raw)
        pair_order.append((a, b))

    _, p_fdr_all, _, _ = multipletests(all_p, method="fdr_bh")
    p_fdr_by_pair = dict(zip(pair_order, p_fdr_all))

    for a, b in BROKEN_PAIRS:
        result = new_results[(a, b)]
        p_fdr = float(p_fdr_by_pair[(a, b)])
        validated = None
        if p_fdr < 0.10:
            validated = validates_categorical_fallback_on_split(df_val, "is_goal", a, b)
        print(f"[result] {a} x {b}: p_raw={result.interaction_p_raw:.6f} p_fdr={p_fdr:.6f} validated={validated}")

        client.query(
            f"""
            UPDATE {q('cxg_bivariate_interaction_v1')}
            SET interaction_p_raw = @p_raw, interaction_p_fdr = @p_fdr, lr_stat = @lr_stat,
                fit_status = @fit_status, validated_on_val_split = @validated, materialized_at = @now
            WHERE track = 'cxg_plus' AND tier = 1 AND feature_a = @a AND feature_b = @b
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("p_raw", "FLOAT64", result.interaction_p_raw),
                    bigquery.ScalarQueryParameter("p_fdr", "FLOAT64", p_fdr),
                    bigquery.ScalarQueryParameter("lr_stat", "FLOAT64", result.lr_stat),
                    bigquery.ScalarQueryParameter("fit_status", "STRING", result.fit_status),
                    bigquery.ScalarQueryParameter("validated", "BOOL", validated),
                    bigquery.ScalarQueryParameter("now", "STRING", now),
                    bigquery.ScalarQueryParameter("a", "STRING", a),
                    bigquery.ScalarQueryParameter("b", "STRING", b),
                ]
            ),
            location=LOCATION,
        ).result()

    remaining_null = _fetch_df(
        client,
        f"SELECT COUNT(*) AS n FROM {q('cxg_bivariate_interaction_v1')} WHERE track='cxg_plus' AND interaction_p_fdr IS NULL",
    )
    print(f"[verify] remaining NULL interaction_p_fdr rows for cxg_plus: {remaining_null.iloc[0]['n']}")


if __name__ == "__main__":
    main()
