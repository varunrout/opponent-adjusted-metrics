"""Materialize `oam_analysis.cxg_defensive_profile_clusters_v1` (CxG+ Phase 2).

Defensive-shape archetype clustering, independent of the Phase 1 ODI
pipeline (no player identity used, only positional geometry already derived
in `oam_features.cxg_defensive_360_features` / `cxg_line_shape_360_features`
-- no geometry is recomputed).

Pipeline:
  1. Pull the 13-column feature set (`defprofile.features.CLUSTER_FEATURES`,
     selection reasoning documented there) for the full 3,960-shot
     360-eligible cohort, joined to `cxg_plus_360_model_matrix_v1` for
     `split` and `is_goal`.
  2. Fit a median `SimpleImputer` + `StandardScaler` on the TRAIN split
     only; apply the same fitted transform to validation/test (never
     refit on them -- this is the only place missingness is touched, and
     only for the ~2.4% of rows with a PARTIAL null; rows with ALL 13
     features null are never imputed, see step 3).
  3. Rows with every one of the 13 features null (no visible-area-gated
     geometry available at all -- 101/3,960 shots) are never assigned a
     cluster: `cluster_label` is NULL with an explicit reason, rather than
     force-fitting a cluster onto an entirely fabricated (all-imputed)
     feature vector.
  4. Fit K-Means for k in 4..8 on the TRAIN split only; report silhouette
     score and cluster-size balance per k (`audit_outputs` raw JSON).
  5. Assign every eligible shot (train, validation, test) to the nearest
     TRAIN-fit centroid. Validation/test never influence the fit.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from google.cloud import bigquery
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from opponent_adjusted.analysis.defprofile.features import (
    CLUSTER_FEATURES,
    DEFENSIVE_360_CANDIDATES,
    LINE_SHAPE_360_CANDIDATES,
)

PROJECT = "oam-varun-260819"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
SEED = 260821  # matches run_cxg_split_analysis.py's split seed for repo-wide reproducibility convention

FEATURE_SET_VERSION = "cxg_defprofile_features_v1_f13a"
K_CHOSEN = 4  # see audit_outputs/cxg_analysis/defprofile_phase2 report for k-selection justification
CLUSTER_MODEL_VERSION = f"cxg_defprofile_kmeans_v1_k{K_CHOSEN}_{FEATURE_SET_VERSION}"

TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_defensive_profile_clusters_v1"
K_SELECTION_PATH = ROOT / "audit_outputs" / "cxg_analysis" / "defprofile_phase2" / "raw" / "k_selection.json"
CLUSTER_MEANS_PATH = ROOT / "audit_outputs" / "cxg_analysis" / "defprofile_phase2" / "raw" / "cluster_means.json"

DEF360_SELECTED = [c for c in DEFENSIVE_360_CANDIDATES if c in CLUSTER_FEATURES]
LINESHAPE_SELECTED = [c for c in LINE_SHAPE_360_CANDIDATES if c in CLUSTER_FEATURES]

FEATURE_QUERY = f"""
SELECT
  m.event_id, m.match_id, m.split, m.is_goal,
  {', '.join('d.' + c for c in DEF360_SELECTED)},
  {', '.join('l.' + c for c in LINESHAPE_SELECTED)}
FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_plus_360_model_matrix_v1` m
JOIN `{PROJECT}.{FEATURE_DATASET}.cxg_defensive_360_features` d
  ON m.event_id = d.event_id AND d.data_version = @data_version
JOIN `{PROJECT}.{FEATURE_DATASET}.cxg_line_shape_360_features` l
  ON m.event_id = l.event_id AND l.data_version = @data_version
"""


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def main() -> None:
    client = _client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION)]
    )
    rows = [dict(r.items()) for r in client.query(FEATURE_QUERY, job_config=job_config, location=LOCATION).result()]
    print(f"fetched {len(rows)} shots")
    assert len(rows) == 3960, f"expected 3960 360-eligible shots, got {len(rows)}"

    event_ids = [r["event_id"] for r in rows]
    match_ids = [r["match_id"] for r in rows]
    splits = [r["split"] for r in rows]
    is_goal = [bool(r["is_goal"]) if r["is_goal"] is not None else False for r in rows]
    X_raw = np.array([[r[c] for c in CLUSTER_FEATURES] for r in rows], dtype=float)
    all_null_mask = np.all(np.isnan(X_raw), axis=1)
    print(f"rows with ALL {len(CLUSTER_FEATURES)} features null (geometry-ineligible): {all_null_mask.sum()}")

    train_mask = np.array([s == "train" for s in splits])
    train_eligible_mask = train_mask & ~all_null_mask
    X_train_raw = X_raw[train_eligible_mask]

    imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    scaler = StandardScaler().fit(imputer.transform(X_train_raw))
    X_train = scaler.transform(imputer.transform(X_train_raw))

    # --- k selection (train only) ---
    k_selection = []
    fitted_models: dict[int, KMeans] = {}
    for k in range(4, 9):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_train)
        sil = silhouette_score(X_train, km.labels_)
        sizes = np.bincount(km.labels_, minlength=k)
        k_selection.append(
            {
                "k": k,
                "silhouette_train": float(sil),
                "cluster_sizes_train": sizes.tolist(),
                "min_cluster_fraction_train": float(sizes.min() / sizes.sum()),
            }
        )
        fitted_models[k] = km
        print(f"k={k} silhouette={sil:.4f} sizes={sizes.tolist()}")

    K_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    K_SELECTION_PATH.write_text(json.dumps(k_selection, indent=2), encoding="utf8")

    final_model = fitted_models[K_CHOSEN]

    # --- assign every eligible shot (train + validation + test) to the nearest train-fit centroid ---
    eligible_mask = ~all_null_mask
    X_eligible_raw = X_raw[eligible_mask]
    X_eligible = scaler.transform(imputer.transform(X_eligible_raw))
    labels_eligible = final_model.predict(X_eligible)

    cluster_label = np.full(len(rows), -1, dtype=int)
    cluster_label[eligible_mask] = labels_eligible

    # --- cluster interpretation: real mean/median of RAW (unimputed) values per cluster ---
    cluster_stats = []
    for c in range(K_CHOSEN):
        member_mask = eligible_mask & (cluster_label == c)
        n = int(member_mask.sum())
        member_raw = X_raw[member_mask]
        member_goal = np.array(is_goal)[member_mask]
        member_split = np.array(splits)[member_mask]
        stats = {
            "cluster": c,
            "n": n,
            "fraction_of_cohort": n / len(rows),
            "goal_rate": float(member_goal.mean()) if n else None,
            "split_counts": {s: int((member_split == s).sum()) for s in ("train", "validation", "test")},
            "feature_means": {
                feat: (float(np.nanmean(member_raw[:, i])) if n else None)
                for i, feat in enumerate(CLUSTER_FEATURES)
            },
            "feature_medians": {
                feat: (float(np.nanmedian(member_raw[:, i])) if n else None)
                for i, feat in enumerate(CLUSTER_FEATURES)
            },
        }
        cluster_stats.append(stats)
        print(f"cluster {c}: n={n} ({stats['fraction_of_cohort']:.1%}) goal_rate={stats['goal_rate']}")

    CLUSTER_MEANS_PATH.write_text(json.dumps(cluster_stats, indent=2), encoding="utf8")

    # --- assemble output rows ---
    materialized_at = datetime.now(UTC).isoformat()
    rows_out = []
    for i, r in enumerate(rows):
        label = int(cluster_label[i]) if cluster_label[i] >= 0 else None
        rows_out.append(
            {
                "event_id": event_ids[i],
                "match_id": match_ids[i],
                "split": splits[i],
                "cluster_label": label,
                "cluster_label_null_reason": None if label is not None else "geometry_ineligible_all_features_null",
                "cluster_model_version": CLUSTER_MODEL_VERSION,
                "k": K_CHOSEN,
                "feature_set_version": FEATURE_SET_VERSION,
                "data_version": DATA_VERSION,
                "silver_schema_version": SCHEMA_VERSION,
                "materialized_at": materialized_at,
            }
        )

    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("match_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("split", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cluster_label", "INT64"),
        bigquery.SchemaField("cluster_label_null_reason", "STRING"),
        bigquery.SchemaField("cluster_model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("k", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("feature_set_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("data_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("silver_schema_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    job_config_load = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_json(rows_out, TABLE, job_config=job_config_load, location=LOCATION)
    job.result()
    print(json.dumps({"table": TABLE, "rows": len(rows_out), "cluster_model_version": CLUSTER_MODEL_VERSION}))


if __name__ == "__main__":
    main()
