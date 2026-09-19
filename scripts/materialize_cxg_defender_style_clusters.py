"""Materialize CxG+ Phase B defender playing-STYLE clusters.

Outputs
-------
  1. `oam_analysis.cxg_defender_style_clusters_v1`         (one row per player)
  2. `oam_analysis.cxg_defender_style_cluster_profile_v1`  (one row per cluster)
  3. `oam_features.cxg_defensive_360_features.nearest_defender_style_archetype`
     -- an ADDITIVE column merge (ALTER TABLE ADD COLUMN IF NOT EXISTS +
     targeted UPDATE). The Gold table is never CREATE OR REPLACE'd here, so
     this cannot clobber concurrent work on the same table.

Pipeline
--------
  1. Aggregate every player's seven defensive-action counts from
     `oam_core.events` (their FULL event history -- not shot-facing
     involvement rows), broken down by the existing match-level split from
     `oam_analysis.cxg_match_splits_v1`. `statsbomb_xg` is never read.
  2. Build the compositional action-share vector per player
     (`defstyle.features.action_shares`); players below `MIN_ACTIONS` are
     excluded from clustering with an explicit null reason.
  3. Fit median-imputer + StandardScaler + K-Means on the TRAIN split only.
  4. Choose k by silhouette + elbow + cluster-size balance (train only).
  5. Validate: seed-perturbation ARI, 80% subsample ARI, and an independent
     validation-split refit ARI.
  6. Derive archetype labels mechanically from the scaled centroids
     (`defstyle.labels`) -- muddy clusters stay muddy.
  7. Assign every above-threshold player, then join each CxG+ 360-eligible
     shot to its nearest defender's archetype via the EXISTING nearest-defender
     definition in `oam_analysis.cxg_defensive_involvement_v1`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from google.cloud import bigquery

from opponent_adjusted.analysis.defstyle.clustering import (
    K_RANGE,
    build_preprocessor,
    fit_kmeans,
    holdout_refit_ari,
    k_diagnostics,
    stability_report,
)
from opponent_adjusted.analysis.defstyle.contracts import (
    GOLD_STYLE_ARCHETYPE_COLUMN,
    GOLD_TARGET_DATASET,
    GOLD_TARGET_TABLE,
)
from opponent_adjusted.analysis.defstyle.features import (
    ACTION_TYPES,
    FEATURE_SET_VERSION,
    MIN_ACTIONS,
    NULL_REASON_BELOW_THRESHOLD,
    STYLE_FEATURES,
    action_shares,
)
from opponent_adjusted.analysis.defstyle.labels import derive_archetype_labels, is_muddy
from opponent_adjusted.analysis.defstyle.shot_join import (
    coverage_summary,
    resolve_shot_archetype,
)

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

K_CHOSEN = 4  # see audit_outputs/.../phase_b_report.md for the k-selection evidence
CLUSTER_MODEL_VERSION = f"cxg_defstyle_kmeans_v1_k{K_CHOSEN}_{FEATURE_SET_VERSION}"

CLUSTERS_TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_defender_style_clusters_v1"
PROFILE_TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_defender_style_cluster_profile_v1"
GOLD_TABLE = f"{PROJECT}.{GOLD_TARGET_DATASET}.{GOLD_TARGET_TABLE}"
STAGING_TABLE = f"{PROJECT}.{ANALYSIS_DATASET}._cxg_defstyle_shot_archetype_staging"

AUDIT_DIR = ROOT / "audit_outputs" / "cxg_analysis" / "phase_b_defender_style_clustering" / "raw"

ACTION_LIST_SQL = ", ".join(f'"{action}"' for action in ACTION_TYPES)

# Per-player, per-split defensive-action counts over the player's whole event
# history. `oam_core.events` holds three silver_schema_version copies of the
# same corpus, so both lineage columns must be pinned or every count triples.
COUNTS_QUERY = f"""
SELECT
  e.player_id,
  s.split,
  e.event_type_name,
  COUNT(*) AS n
FROM `{PROJECT}.{CORE_DATASET}.events` e
JOIN `{PROJECT}.{ANALYSIS_DATASET}.cxg_match_splits_v1` s
  USING (match_id)
WHERE e.data_version = @data_version
  AND e.silver_schema_version = @schema_version
  AND e.player_id IS NOT NULL
  AND e.event_type_name IN ({ACTION_LIST_SQL})
GROUP BY 1, 2, 3
"""

# Nearest defender per CxG+ 360-eligible shot, from the EXISTING definition.
SHOT_QUERY = f"""
SELECT
  m.event_id,
  m.split,
  i.player_id AS nearest_defender_player_id
FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_plus_360_model_matrix_v1` m
LEFT JOIN `{PROJECT}.{ANALYSIS_DATASET}.cxg_defensive_involvement_v1` i
  ON m.event_id = i.shot_event_id
"""

CLUSTERS_SCHEMA = [
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("total_actions", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("total_actions_train", "INT64", mode="REQUIRED"),
    *[bigquery.SchemaField(f, "FLOAT64") for f in STYLE_FEATURES],
    bigquery.SchemaField("cluster_label", "INT64"),
    bigquery.SchemaField("style_archetype", "STRING"),
    bigquery.SchemaField("style_archetype_null_reason", "STRING"),
    bigquery.SchemaField("used_in_fit", "BOOL", mode="REQUIRED"),
    bigquery.SchemaField("cluster_model_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("k", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("feature_set_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("data_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("silver_schema_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("materialized_at", "TIMESTAMP", mode="REQUIRED"),
]

PROFILE_SCHEMA = [
    bigquery.SchemaField("cluster_label", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("style_archetype", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("is_muddy", "BOOL", mode="REQUIRED"),
    bigquery.SchemaField("cluster_size", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("cluster_size_train", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("cluster_fraction", "FLOAT64", mode="REQUIRED"),
    *[bigquery.SchemaField(f"{f}_centroid", "FLOAT64") for f in STYLE_FEATURES],
    *[bigquery.SchemaField(f"{f}_z", "FLOAT64") for f in STYLE_FEATURES],
    bigquery.SchemaField("dominant_feature", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("dominant_feature_z", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("cluster_model_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("k", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("feature_set_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("data_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("silver_schema_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("materialized_at", "TIMESTAMP", mode="REQUIRED"),
]


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params() -> list[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]


def fetch_counts(client: bigquery.Client) -> dict[int, dict[str, dict[str, int]]]:
    """player_id -> split -> event_type_name -> count."""
    job_config = bigquery.QueryJobConfig(query_parameters=_params())
    counts: dict[int, dict[str, dict[str, int]]] = {}
    for row in client.query(COUNTS_QUERY, job_config=job_config, location=LOCATION).result():
        counts.setdefault(row.player_id, {}).setdefault(row.split, {})[row.event_type_name] = row.n
    return counts


def _merge_splits(by_split: dict[str, dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for split_counts in by_split.values():
        for action, n in split_counts.items():
            merged[action] = merged.get(action, 0) + n
    return merged


def main(skip_gold_merge: bool = False) -> None:
    client = _client()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    materialized_at = datetime.now(UTC).isoformat()

    counts = fetch_counts(client)
    print(f"players with >=1 qualifying defensive action: {len(counts)}")

    player_ids = sorted(counts)
    deploy_shares: dict[int, dict[str, float] | None] = {}
    train_shares: dict[int, dict[str, float] | None] = {}
    total_all: dict[int, int] = {}
    total_train: dict[int, int] = {}
    for player_id in player_ids:
        by_split = counts[player_id]
        all_counts = _merge_splits(by_split)
        tr_counts = by_split.get("train", {})
        total_all[player_id] = sum(all_counts.values())
        total_train[player_id] = sum(tr_counts.values())
        deploy_shares[player_id] = action_shares(all_counts)
        train_shares[player_id] = action_shares(tr_counts)

    fit_ids = [p for p in player_ids if train_shares[p] is not None]
    assign_ids = [p for p in player_ids if deploy_shares[p] is not None]
    print(f"threshold={MIN_ACTIONS}: {len(fit_ids)} players in fit, {len(assign_ids)} assignable")

    x_train_raw = np.array([[train_shares[p][f] for f in STYLE_FEATURES] for p in fit_ids])
    pre = build_preprocessor().fit(x_train_raw)
    x_train = pre.transform(x_train_raw)

    diagnostics = k_diagnostics(x_train, K_RANGE)
    for d in diagnostics:
        print(
            f"k={d.k} silhouette={d.silhouette_train:.4f} inertia={d.inertia_train:.1f} "
            f"sizes={d.cluster_sizes_train}"
        )
    (AUDIT_DIR / "k_selection.json").write_text(
        json.dumps([d.__dict__ for d in diagnostics], indent=2), encoding="utf8"
    )

    model = fit_kmeans(x_train, K_CHOSEN)

    # --- validation: stability under re-seeding, subsampling, and an independent refit ---
    stability = stability_report(x_train, K_CHOSEN, reference=model)
    validation_ids = [
        p
        for p in player_ids
        if action_shares(counts[p].get("validation", {})) is not None
    ]
    x_validation = pre.transform(
        np.array(
            [
                [action_shares(counts[p]["validation"])[f] for f in STYLE_FEATURES]
                for p in validation_ids
            ]
        )
    )
    x_assign_raw = np.array([[deploy_shares[p][f] for f in STYLE_FEATURES] for p in assign_ids])
    x_assign = pre.transform(x_assign_raw)
    validation_ari = holdout_refit_ari(x_train, x_validation, x_assign, K_CHOSEN)
    stability_payload = {
        "k": K_CHOSEN,
        "seed_ari": stability.seed_ari,
        "min_seed_ari": stability.min_seed_ari,
        "bootstrap_ari_mean": stability.bootstrap_ari_mean,
        "bootstrap_ari_min": stability.bootstrap_ari_min,
        "validation_split_refit_ari": validation_ari,
        "n_validation_players": len(validation_ids),
    }
    (AUDIT_DIR / "stability.json").write_text(
        json.dumps(stability_payload, indent=2), encoding="utf8"
    )
    print(json.dumps(stability_payload, indent=2))

    # --- labels derived from scaled centroids, never hand-assigned ---
    scaled_centroids = model.cluster_centers_
    raw_centroids = pre.named_steps["scale"].inverse_transform(scaled_centroids)
    archetypes = derive_archetype_labels([list(c) for c in scaled_centroids])
    print("archetypes:", archetypes)

    labels_assign = model.predict(x_assign)
    labels_train = model.predict(x_train)
    label_by_player = dict(zip(assign_ids, (int(v) for v in labels_assign), strict=True))
    fit_id_set = set(fit_ids)

    # --- clusters table ---
    cluster_rows = []
    for player_id in player_ids:
        shares = deploy_shares[player_id]
        label = label_by_player.get(player_id)
        cluster_rows.append(
            {
                "player_id": int(player_id),
                "total_actions": int(total_all[player_id]),
                "total_actions_train": int(total_train[player_id]),
                **{f: (float(shares[f]) if shares else None) for f in STYLE_FEATURES},
                "cluster_label": label,
                "style_archetype": archetypes[label] if label is not None else None,
                "style_archetype_null_reason": (
                    None if label is not None else NULL_REASON_BELOW_THRESHOLD
                ),
                "used_in_fit": player_id in fit_id_set,
                "cluster_model_version": CLUSTER_MODEL_VERSION,
                "k": K_CHOSEN,
                "feature_set_version": FEATURE_SET_VERSION,
                "data_version": DATA_VERSION,
                "silver_schema_version": SCHEMA_VERSION,
                "materialized_at": materialized_at,
            }
        )

    # --- cluster profile table ---
    profile_rows = []
    for cluster in range(K_CHOSEN):
        size = int((labels_assign == cluster).sum())
        z_values = dict(zip(STYLE_FEATURES, scaled_centroids[cluster], strict=True))
        dominant = max(z_values, key=lambda f: z_values[f])
        profile_rows.append(
            {
                "cluster_label": cluster,
                "style_archetype": archetypes[cluster],
                "is_muddy": is_muddy(archetypes[cluster]),
                "cluster_size": size,
                "cluster_size_train": int((labels_train == cluster).sum()),
                "cluster_fraction": size / len(assign_ids),
                **{
                    f"{f}_centroid": float(raw_centroids[cluster][i])
                    for i, f in enumerate(STYLE_FEATURES)
                },
                **{f"{f}_z": float(z_values[f]) for f in STYLE_FEATURES},
                "dominant_feature": dominant,
                "dominant_feature_z": float(z_values[dominant]),
                "cluster_model_version": CLUSTER_MODEL_VERSION,
                "k": K_CHOSEN,
                "feature_set_version": FEATURE_SET_VERSION,
                "data_version": DATA_VERSION,
                "silver_schema_version": SCHEMA_VERSION,
                "materialized_at": materialized_at,
            }
        )
    (AUDIT_DIR / "cluster_profile.json").write_text(
        json.dumps(profile_rows, indent=2), encoding="utf8"
    )

    _load(client, cluster_rows, CLUSTERS_TABLE, CLUSTERS_SCHEMA)
    _load(client, profile_rows, PROFILE_TABLE, PROFILE_SCHEMA)
    print(f"wrote {len(cluster_rows)} cluster rows, {len(profile_rows)} profile rows")

    # --- shot-level attachment ---
    archetype_by_player = {
        int(p): (archetypes[label_by_player[p]] if p in label_by_player else None)
        for p in player_ids
    }
    shot_rows = [
        dict(row.items()) for row in client.query(SHOT_QUERY, location=LOCATION).result()
    ]
    resolutions = [
        resolve_shot_archetype(r["nearest_defender_player_id"], archetype_by_player)
        for r in shot_rows
    ]
    coverage = coverage_summary(resolutions)
    coverage["by_archetype"] = {
        a: sum(1 for r in resolutions if r.archetype == a) for a in sorted(set(archetypes))
    }
    (AUDIT_DIR / "shot_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf8")
    print(json.dumps(coverage, indent=2))

    if skip_gold_merge:
        print("skip-gold-merge set; Gold column not written")
        return

    assignments = [
        {"event_id": row["event_id"], "nearest_defender_style_archetype": resolution.archetype}
        for row, resolution in zip(shot_rows, resolutions, strict=True)
        if resolution.archetype is not None
    ]
    merge_gold_column(client, assignments)
    print(f"merged {len(assignments)} archetype assignments into {GOLD_TABLE}")


def _load(
    client: bigquery.Client, rows: list[dict], table: str, schema: list[bigquery.SchemaField]
) -> None:
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_json(rows, table, job_config=job_config, location=LOCATION).result()


def merge_gold_column(client: bigquery.Client, assignments: list[dict]) -> None:
    """Additively add + populate the archetype column on the Gold table.

    Never CREATE OR REPLACE: the Gold defensive feature family is under
    concurrent development, so this only ever widens the schema by one
    nullable STRING and updates that one column. Any other column, and any
    row not present in `assignments`, is left exactly as found (a shot with
    no resolvable archetype stays NULL, with the reason recorded in the
    local audit JSON rather than in the model-facing table).
    """
    column = GOLD_STYLE_ARCHETYPE_COLUMN.name
    client.query(
        f"ALTER TABLE `{GOLD_TABLE}` ADD COLUMN IF NOT EXISTS `{column}` STRING",
        location=LOCATION,
    ).result()

    staging_schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField(column, "STRING"),
    ]
    client.load_table_from_json(
        assignments,
        STAGING_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=staging_schema, write_disposition="WRITE_TRUNCATE"
        ),
        location=LOCATION,
    ).result()
    try:
        # Reset first so a re-run cannot leave a stale archetype on a shot that
        # no longer resolves; then apply the current assignments.
        client.query(
            f"UPDATE `{GOLD_TABLE}` SET `{column}` = NULL WHERE TRUE", location=LOCATION
        ).result()
        client.query(
            f"""
            UPDATE `{GOLD_TABLE}` t
            SET t.`{column}` = s.`{column}`
            FROM `{STAGING_TABLE}` s
            WHERE t.event_id = s.event_id
            """,
            location=LOCATION,
        ).result()
    finally:
        client.query(f"DROP TABLE IF EXISTS `{STAGING_TABLE}`", location=LOCATION).result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gold-merge", action="store_true")
    args = parser.parse_args()
    main(skip_gold_merge=args.skip_gold_merge)
