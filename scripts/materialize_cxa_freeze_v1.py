"""CxA P_create Step 9: freeze model family, feature set, and hyperparameters, for
BOTH tracks -- the first and only script in this repo permitted to write
`oam_ml.cxa_{track}_frozen_v1_config`.

Frozen per docs/analysis/cxa_p_create_baseline_and_candidate_v1.md's own comparison
and recommendation (not re-derived here):
- Model family: LightGBM (tree) for both tracks -- beat logistic on every metric in
  both tracks, and CxA+'s logistic candidate has a real, reported quasi-complete
  separation problem on `reception_geometry_missing` (coefficient -21.82,
  std_error 470,161, p=1.0; 134 affected rows, 0 positives in either split) that
  makes it unreliable as a frozen candidate. Logistic is not carried forward here.
- Feature set: exactly the locked lists in docs/analysis/cxa_event_p_create_feature_
  lock_v1.md (10 features) and docs/analysis/cxa_plus_p_create_feature_lock_v1.md
  (9 features, `pass_body_part_name=No Touch` and `pass_technique_name=Straight`
  both stay excluded). No new features, no silent additions -- reuses
  `encode_event_candidate`/`encode_plus_candidate` from `_cxa_modeling_common.py`
  unchanged.

This script's only new work is the capped hyperparameter search (<=8 LightGBM
configurations per track, train-fit / validation-scored, see `SEARCH_GRID` below) and
writing the frozen config tables. It does not touch `split='test'` anywhere -- the
data loader only ever requests `('train', 'validation')`, same as every other CxA
modelling script in this project.

Selection rule (stated, not left implicit): the configuration with the LOWEST
validation `log_loss` is chosen; validation `roc_auc` is the tie-breaker if two
configurations are within 0.0005 log_loss of each other. log_loss is primary because
it is the only one of the two metrics sensitive to calibration as well as ranking,
and this model's eventual use (a probability, not just a rank) needs calibration to be
right, not just discrimination.

Writes (WRITE_TRUNCATE, one row per track):
  oam_ml.cxa_event_frozen_v1_config
  oam_ml.cxa_plus_frozen_v1_config
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cxa_modeling_common import (  # noqa: E402
    ML_DATASET,
    PROJECT,
    compute_metrics,
    encode_event_candidate,
    encode_plus_candidate,
    load_track,
    write_table,
)

FROZEN_AT = datetime.now(UTC).isoformat()

EVENT_FEATURE_LIST = [
    "is_cross",
    "is_through_ball",
    "pass_type_name (Corner, Free Kick levels)",
    "start_x",
    "end_x",
    "play_pattern_name (Counter, Corner levels)",
    "pass_technique_name (Outswinging level)",
    "is_switch",
    "is_cut_back",
    "pass_body_part_name (No Touch level)",
]
PLUS_FEATURE_LIST = [
    "is_cross",
    "is_through_ball",
    "pass_type_name (Corner, Free Kick levels)",
    "play_pattern_name (Counter, Corner levels)",
    "pass_technique_name (Outswinging level)",
    "is_cut_back",
    "is_switch",
    "reception_nearest_opponent_distance_m",
    "reception_opponents_within_5m",
    "start_x (shared base)",
    "end_x (shared base)",
]

SOURCE_DOCS = [
    "docs/analysis/cxa_p_create_baseline_and_candidate_v1.md",
    "docs/analysis/cxa_event_p_create_feature_lock_v1.md",
    "docs/analysis/cxa_plus_p_create_feature_lock_v1.md",
    "docs/analysis/cxa_p_create_locked_feature_eda_v1.md",
]

# Baseline (config 0, the untuned config already fit in materialize_cxa_candidate_v1.py)
# plus 7 reasoned single/paired variations -- learning-rate/tree-count trade-off,
# depth/complexity in both directions, stronger regularization, and no subsampling.
# Capped at 8 per the task -- not an open-ended search.
SEARCH_GRID: list[dict] = [
    {
        "label": "baseline (untuned, from candidate comparison)",
        "n_estimators": 200, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.05,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    {
        "label": "more trees, lower learning rate",
        "n_estimators": 400, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.03,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    {
        "label": "fewer trees, higher learning rate",
        "n_estimators": 100, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.10,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    {
        "label": "deeper trees",
        "n_estimators": 200, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.05,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    {
        "label": "shallower trees",
        "n_estimators": 200, "max_depth": 3, "num_leaves": 7, "learning_rate": 0.05,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    {
        "label": "stronger regularization (higher min_child_samples)",
        "n_estimators": 200, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.05,
        "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    {
        "label": "no subsampling",
        "n_estimators": 200, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.05,
        "min_child_samples": 50, "subsample": 1.0, "colsample_bytree": 1.0,
    },
    {
        "label": "more trees + deeper (combined)",
        "n_estimators": 300, "max_depth": 5, "num_leaves": 25, "learning_rate": 0.04,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    },
]


def run_search(track: str, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(SEARCH_GRID):
        params = {k: v for k, v in cfg.items() if k != "label"}
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        train_pred = model.predict_proba(X_train)[:, 1]
        val_metrics = compute_metrics(y_val.to_numpy().astype(bool), val_pred, with_auc=True)
        train_metrics = compute_metrics(y_train.to_numpy().astype(bool), train_pred, with_auc=True)
        rows.append(
            {
                "track": track,
                "config_index": i,
                "label": cfg["label"],
                **params,
                "train_log_loss": train_metrics["log_loss"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_log_loss": val_metrics["log_loss"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )
        print(
            f"  [{track}] cfg {i} ({cfg['label']}): "
            f"val_log_loss={val_metrics['log_loss']:.5f} val_roc_auc={val_metrics['roc_auc']:.5f}"
        )
    return pd.DataFrame(rows)


def select_config(search_df: pd.DataFrame) -> dict:
    """Lowest validation log_loss wins; roc_auc breaks ties within 0.0005 log_loss."""
    best_log_loss = search_df["val_log_loss"].min()
    within_tie = search_df[search_df["val_log_loss"] <= best_log_loss + 0.0005]
    chosen = within_tie.sort_values("val_roc_auc", ascending=False).iloc[0]
    return chosen.to_dict()


def run_track(client: bigquery.Client, track: str, feature_list: list[str]) -> None:
    print(f"=== {track}: hyperparameter search (train/validation only) ===")
    df = load_track(client, track)
    train_raw = df[df["split"] == "train"]
    val_raw = df[df["split"] == "validation"]

    if track == "event":
        X_train = encode_event_candidate(train_raw)
        X_val = encode_event_candidate(val_raw)
    else:
        train_median_dist = float(train_raw["reception_nearest_opponent_distance_m"].median())
        X_train = encode_plus_candidate(train_raw, train_median_dist)
        X_val = encode_plus_candidate(val_raw, train_median_dist)

    y_train = train_raw["y_create"].astype(int)
    y_val = val_raw["y_create"].astype(int)

    search_df = run_search(track, X_train, y_train, X_val, y_val)
    chosen = select_config(search_df)
    print(f"  -> chosen: cfg {chosen['config_index']} ({chosen['label']})")

    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_freeze_v1_search", search_df)

    frozen_row = pd.DataFrame(
        [
            {
                "track": track,
                "model_family": "lightgbm_tree",
                "feature_list": feature_list,
                "n_estimators": int(chosen["n_estimators"]),
                "max_depth": int(chosen["max_depth"]),
                "num_leaves": int(chosen["num_leaves"]),
                "learning_rate": float(chosen["learning_rate"]),
                "min_child_samples": int(chosen["min_child_samples"]),
                "subsample": float(chosen["subsample"]),
                "colsample_bytree": float(chosen["colsample_bytree"]),
                "selection_rule": "lowest validation log_loss; roc_auc tie-break within 0.0005 log_loss",
                "chosen_config_label": chosen["label"],
                "chosen_config_index": int(chosen["config_index"]),
                "train_log_loss": chosen["train_log_loss"],
                "train_roc_auc": chosen["train_roc_auc"],
                "val_log_loss": chosen["val_log_loss"],
                "val_roc_auc": chosen["val_roc_auc"],
                "source_docs": SOURCE_DOCS,
                "frozen_at": FROZEN_AT,
            }
        ]
    )
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_frozen_v1_config", frozen_row)
    print()


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    run_track(client, "event", EVENT_FEATURE_LIST)
    run_track(client, "plus", PLUS_FEATURE_LIST)


if __name__ == "__main__":
    main()
