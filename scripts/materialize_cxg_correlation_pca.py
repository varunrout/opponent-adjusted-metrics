"""Materialize the correlation/redundancy screen + PCA analysis tables (CxG/CxG+).

Pipeline stage: Summary Statistics -> EDA -> Univariate -> **this step** -> Bivariate
(separate, later task -- not built here). See
`src/opponent_adjusted/analysis/corrpca/features.py` for the full candidate-pool and
redundancy-resolution reasoning this script encodes.

Writes 4 new tables via INSERT-only (never CREATE OR REPLACE, so existing analysis-table
rows for every other family/step are untouched):
  - cxg_feature_correlation_v1   (every train-split pair within each track's pre-trim pool)
  - cxg_bivariate_candidate_pool_v1  (the post-trim handoff artifact for the next, bivariate, task)
  - cxg_pca_components_v1 / cxg_pca_loadings_v1  (diagnostic only, does not filter the pool)
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from opponent_adjusted.analysis.corrpca.contracts import (
    CXG_BIVARIATE_CANDIDATE_POOL_V1,
    CXG_FEATURE_CORRELATION_V1,
    CXG_PCA_COMPONENTS_V1,
    CXG_PCA_LOADINGS_V1,
)
from opponent_adjusted.analysis.corrpca.features import (
    CXG_EVENT_QUALIFIED,
    CXG_PLUS_CATEGORICAL,
    CXG_PLUS_PRE_TRIM_NUMERIC,
    REDUNDANCY_THRESHOLD,
    REDUNDANT_PAIRS,
    final_cxg_event_pool,
    final_cxg_plus_numeric_pool,
)

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


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


def _resolution_for(track: str, a: str, b: str, r: float) -> tuple[bool, str, str | None]:
    for t, fa, fb, r_train, dropped, reason in REDUNDANT_PAIRS:
        if t == track and {fa, fb} == {a, b}:
            if len(dropped) == 2:
                res = "dropped_both"
            elif dropped[0] == a:
                res = "dropped_a"
            else:
                res = "dropped_b"
            return True, res, reason
    return False, "kept_both_moderate", None


def materialize_correlation(client: bigquery.Client, track: str, df: pd.DataFrame, candidates: tuple[str, ...]) -> None:
    _ensure_table(client, CXG_FEATURE_CORRELATION_V1)
    client.query(
        f"DELETE FROM {q('cxg_feature_correlation_v1')} WHERE track = @track",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("track", "STRING", track)]),
        location=LOCATION,
    ).result()

    corr = df[list(candidates)].corr(numeric_only=True)
    now = datetime.now(UTC).isoformat()
    rows = []
    uniq = sorted(set(candidates))
    for i, a in enumerate(uniq):
        for b in uniq[i + 1 :]:
            r = corr.loc[a, b]
            n = int(df[[a, b]].dropna().shape[0])
            r_val = None if pd.isna(r) else float(r)
            if r_val is None:
                is_redundant, resolution, reason = False, "kept_both_moderate", None
            else:
                is_redundant, resolution, reason = _resolution_for(track, a, b, r_val)
            rows.append(
                {
                    "track": track,
                    "feature_a": a,
                    "feature_b": b,
                    "r_train": r_val,
                    "n_train": n,
                    "is_redundant": bool(is_redundant),
                    "resolution": resolution,
                    "resolution_reason": reason,
                    "materialized_at": now,
                }
            )
    job_config = bigquery.LoadJobConfig(schema=_schema_from_contract(CXG_FEATURE_CORRELATION_V1), write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_correlation_v1", job_config=job_config, location=LOCATION).result()
    print(f"[correlation] {track}: {len(rows)} pairs, {sum(r['is_redundant'] for r in rows)} redundant (>= {REDUNDANCY_THRESHOLD})")


def materialize_pool(client: bigquery.Client) -> None:
    _ensure_table(client, CXG_BIVARIATE_CANDIDATE_POOL_V1)
    for track in ("cxg_event", "cxg_plus"):
        client.query(
            f"DELETE FROM {q('cxg_bivariate_candidate_pool_v1')} WHERE track = @track",
            job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("track", "STRING", track)]),
            location=LOCATION,
        ).result()

    family_lookup = {
        r["column_name"]: r["feature_family"]
        for r in _fetch_df(
            client,
            f"SELECT DISTINCT column_name, feature_family FROM {q('cxg_feature_inventory_v1')} WHERE column_role = 'feature'",
        ).to_dict("records")
    }

    now = datetime.now(UTC).isoformat()
    rows = []
    for col in final_cxg_event_pool():
        rows.append(
            {
                "track": "cxg_event",
                "feature_family": family_lookup[col],
                "column_name": col,
                "data_type": "FLOAT64",
                "qualification_reason": "univariate_stable",
                "materialized_at": now,
            }
        )
    origin_reverified = {"last_action_interval_s"}  # the only inherited-CxG survivor after redundancy trim
    for col in final_cxg_plus_numeric_pool():
        if col in ("nearest_defender_odi", "mean_backline_odi", "gk_odi"):
            reason = "deliberately_included_despite_weak_signal"
        elif col in origin_reverified:
            reason = "inherited_from_cxg_reverified"
        else:
            reason = "univariate_stable"
        rows.append(
            {
                "track": "cxg_plus",
                "feature_family": family_lookup[col],
                "column_name": col,
                "data_type": "FLOAT64",
                "qualification_reason": reason,
                "materialized_at": now,
            }
        )
    for col in CXG_PLUS_CATEGORICAL:
        rows.append(
            {
                "track": "cxg_plus",
                "feature_family": family_lookup[col],
                "column_name": col,
                "data_type": "INT64",
                "qualification_reason": "categorical_proven_stable",
                "materialized_at": now,
            }
        )

    job_config = bigquery.LoadJobConfig(schema=_schema_from_contract(CXG_BIVARIATE_CANDIDATE_POOL_V1), write_disposition="WRITE_APPEND")
    client.load_table_from_json(rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_candidate_pool_v1", job_config=job_config, location=LOCATION).result()
    print(f"[pool] cxg_event={sum(r['track']=='cxg_event' for r in rows)} cxg_plus={sum(r['track']=='cxg_plus' for r in rows)}")


def materialize_pca(client: bigquery.Client, track: str, df: pd.DataFrame, candidates: tuple[str, ...]) -> dict:
    _ensure_table(client, CXG_PCA_COMPONENTS_V1)
    _ensure_table(client, CXG_PCA_LOADINGS_V1)
    for tbl in ("cxg_pca_components_v1", "cxg_pca_loadings_v1"):
        client.query(
            f"DELETE FROM {q(tbl)} WHERE track = @track",
            job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("track", "STRING", track)]),
            location=LOCATION,
        ).result()

    X_raw = df[list(candidates)].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(X_raw)
    scaler = StandardScaler().fit(imputer.transform(X_raw))
    X = scaler.transform(imputer.transform(X_raw))

    pca = PCA(n_components=min(len(candidates), X.shape[0])).fit(X)
    now = datetime.now(UTC).isoformat()

    comp_rows = []
    cum = 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_):
        cum += float(evr)
        comp_rows.append(
            {
                "track": track,
                "component_number": i + 1,
                "explained_variance_ratio": float(evr),
                "cumulative_variance_ratio": cum,
                "materialized_at": now,
            }
        )
    job_config = bigquery.LoadJobConfig(schema=_schema_from_contract(CXG_PCA_COMPONENTS_V1), write_disposition="WRITE_APPEND")
    client.load_table_from_json(comp_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_components_v1", job_config=job_config, location=LOCATION).result()

    n_80 = next((r["component_number"] for r in comp_rows if r["cumulative_variance_ratio"] >= 0.80), len(comp_rows))
    loading_rows = []
    for i in range(min(n_80, len(comp_rows))):
        for j, feat in enumerate(candidates):
            loading_rows.append(
                {
                    "track": track,
                    "component_number": i + 1,
                    "feature_name": feat,
                    "loading": float(pca.components_[i, j]),
                    "materialized_at": now,
                }
            )
    job_config2 = bigquery.LoadJobConfig(schema=_schema_from_contract(CXG_PCA_LOADINGS_V1), write_disposition="WRITE_APPEND")
    client.load_table_from_json(loading_rows, f"{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_loadings_v1", job_config=job_config2, location=LOCATION).result()

    print(f"[pca] {track}: {len(candidates)} features -> {len(comp_rows)} components, {n_80} needed for 80% cumulative variance")
    return {"n_components": len(comp_rows), "n_80": n_80, "scree": [r["explained_variance_ratio"] for r in comp_rows]}


def main() -> None:
    client = _client()

    df_event = _fetch_df(
        client,
        f"SELECT {', '.join(f'`{c}`' for c in CXG_EVENT_QUALIFIED)} FROM {q('cxg_event_model_matrix_v1')} WHERE split = 'train'",
    )
    plus_cols = sorted(set(c for c in CXG_PLUS_PRE_TRIM_NUMERIC if c not in ("nearest_defender_odi", "mean_backline_odi", "gk_odi")))
    df_plus_m = _fetch_df(
        client,
        f"SELECT event_id, {', '.join(f'`{c}`' for c in plus_cols)} FROM {q('cxg_plus_360_model_matrix_v1')} WHERE split = 'train'",
    )
    df_plus_odi = _fetch_df(
        client,
        f"SELECT event_id, nearest_defender_odi, mean_backline_odi, gk_odi FROM {q('cxg_odi_features_v1')}",
    )
    df_plus = df_plus_m.merge(df_plus_odi, on="event_id", how="left")

    print(f"train rows: cxg_event={len(df_event)} cxg_plus={len(df_plus)}")

    materialize_correlation(client, "cxg_event", df_event, CXG_EVENT_QUALIFIED)
    materialize_correlation(client, "cxg_plus", df_plus, CXG_PLUS_PRE_TRIM_NUMERIC)
    materialize_pool(client)

    pca_event = materialize_pca(client, "cxg_event", df_event, final_cxg_event_pool())
    pca_plus = materialize_pca(client, "cxg_plus", df_plus, final_cxg_plus_numeric_pool())

    print(json.dumps({"cxg_event": pca_event, "cxg_plus": pca_plus}, indent=2))


if __name__ == "__main__":
    main()
