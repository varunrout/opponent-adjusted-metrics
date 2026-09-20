"""CxA detail-page explainability data: feature importances (tree-family sub-models)
and coefficients (the one logistic-family sub-model, CxA+'s P_convert).

**Why this script exists -- a gap found during investigation, not assumed away.**
Before writing this, every prior materialize script in this project was grepped for
`feature_importances_` and every `_coefficients` table was checked against which
model it actually belongs to. Result: **no BigQuery table anywhere holds explainability
data for any of CxA's four actual FROZEN sub-models.**
- P_create (`lightgbm_tree`, both tracks): `oam_ml.cxa_{track}_candidate_v1_
  coefficients` exists, but it belongs to the *logistic candidate* -- P_create froze
  *tree*, not logistic, for both tracks, so that table describes a model that was
  never chosen. No tree feature-importance table for P_create exists anywhere, not
  even in a local audit JSON file -- confirmed via `grep -c feature_importances_`
  across `materialize_cxa_candidate_v1.py`/`_freeze_v1.py`/`_test_eval_v1.py`: zero
  hits in all three.
- P_convert: `analyze_cxconvert_baseline_and_candidate.py` DID capture both
  coefficients and tree feature_importances_ -- but only to a local JSON file
  (`audit_outputs/cxconvert_analysis/baseline_and_candidate/{track}_result.json`),
  never to BigQuery, and from the CANDIDATE-stage fit (untuned hyperparameters for
  the tree, pre-freeze), not the actual frozen configuration selected afterward.

**What this script does about it, and why that's still "read/serve, not
recompute":** each of the four frozen sub-models is refit EXACTLY as
`materialize_cxa_test_eval_v1.py` / `materialize_cxconvert_test_eval_v1.py` already
legitimately refit them (same model family, same feature list, same hyperparameters,
all read live from the already-frozen config tables; same train+validation fit pool)
-- this script makes no new modelling decision, chooses nothing, and tunes nothing.
It only adds one extraction step neither test-eval script happened to include:
reading `.feature_importances_` (trees) or `.params`/`.bse`/`.pvalues` (the logistic
fit) off the exact same fitted object test-eval already produces, and persisting that
as a new, clearly-versioned artifact. This is the same "materialize once, serve
read-only" precedent every prior step in this project already established -- applied
to a byproduct of an existing, already-sanctioned refit, not a new one.

Writes (WRITE_TRUNCATE):
  oam_ml.cxa_event_explainability_v1 / cxa_plus_explainability_v1
    (P_create, both tracks: tree feature importances -- feature, importance_split,
    importance_gain)
  oam_ml.cxconvert_event_explainability_v1
    (P_convert event-only: tree feature importances, same shape)
  oam_ml.cxconvert_plus_explainability_v1
    (P_convert CxA+: logistic coefficients -- feature, coefficient, std_error, p_value)

Does not touch, modify, or re-freeze any `oam_ml.*_frozen_v1_config` table -- read
only. Does not change any locked feature set, model family, or hyperparameter.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from google.cloud import bigquery
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

import _cxa_modeling_common as pcreate
import _cxconvert_modeling_common as pconvert

PROJECT = "oam-varun-260819"
ML_DATASET = "oam_ml"
LOCATION = "europe-west2"


def read_frozen_config(client: bigquery.Client, table: str) -> dict:
    rows = list(client.query(f"SELECT * FROM `{PROJECT}.{ML_DATASET}.{table}`").result())
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly 1 row in {table}, found {len(rows)}")
    return dict(rows[0].items())


def lgb_params_from_frozen(frozen: dict) -> dict:
    return dict(
        n_estimators=int(frozen["n_estimators"]), max_depth=int(frozen["max_depth"]),
        num_leaves=int(frozen["num_leaves"]), learning_rate=float(frozen["learning_rate"]),
        min_child_samples=int(frozen["min_child_samples"]), subsample=float(frozen["subsample"]),
        colsample_bytree=float(frozen["colsample_bytree"]), random_state=42, verbosity=-1,
    )


def write_table(client: bigquery.Client, table: str, df: pd.DataFrame) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table, job_config=job_config, location=LOCATION)
    job.result()
    print(f"wrote {len(df)} row(s) -> {table}")


def tree_importances(model: lgb.LGBMClassifier, columns: list[str]) -> pd.DataFrame:
    split_imp = model.booster_.feature_importance(importance_type="split")
    gain_imp = model.booster_.feature_importance(importance_type="gain")
    return pd.DataFrame({
        "feature": columns,
        "importance_split": [int(v) for v in split_imp],
        "importance_gain": [float(v) for v in gain_imp],
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)


def run_p_create(client: bigquery.Client, track: str) -> None:
    print(f"=== P_create/{track} explainability ===")
    frozen = read_frozen_config(client, f"cxa_{track}_frozen_v1_config")
    assert frozen["model_family"] == "lightgbm_tree", f"unexpected P_create family: {frozen['model_family']}"

    df = pcreate.load_track(client, track, splits=("train", "validation"))
    if track == "event":
        X = pcreate.encode_event_candidate(df)
    else:
        median_dist = float(df["reception_nearest_opponent_distance_m"].median())
        X = pcreate.encode_plus_candidate(df, median_dist)
    y = df["y_create"].astype(int)

    params = lgb_params_from_frozen(frozen)
    print(f"  refitting frozen lightgbm_tree (n={len(X)}): {params}")
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    out = tree_importances(model, list(X.columns))
    write_table(client, f"{PROJECT}.{ML_DATASET}.cxa_{track}_explainability_v1", out)


def run_p_convert(client: bigquery.Client, track: str) -> None:
    print(f"=== P_convert/{track} explainability ===")
    frozen = read_frozen_config(client, f"cxconvert_{track}_frozen_v1_config")
    model_family = frozen["model_family"]

    df = pconvert.load_track(client, track, splits=("train", "validation"))
    y = df["y_goal"].astype(int)
    train_median_gk = float(df["shot_gk_distance_m"].median())
    n_null_gk = int(df["shot_gk_distance_m"].isna().sum())
    gk_flag_informative = n_null_gk > 0

    encode_fn = pconvert.encode_event_candidate if track == "event" else pconvert.encode_plus_candidate
    is_tree = model_family == "lightgbm_tree"
    X = encode_fn(df, train_median_gk, for_tree=is_tree, gk_flag_informative=gk_flag_informative)

    if is_tree:
        params = lgb_params_from_frozen(frozen)
        print(f"  refitting frozen lightgbm_tree (n={len(X)}): {params}")
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        out = tree_importances(model, list(X.columns))
        write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_explainability_v1", out)
    else:
        print(f"  refitting frozen logistic_mle (statsmodels Logit) (n={len(X)})")
        Xc = add_constant(X, has_constant="add")
        logit_model = Logit(y, Xc).fit(disp=0, maxiter=200)
        out = pd.DataFrame({
            "feature": logit_model.params.index,
            "coefficient": logit_model.params.values,
            "std_error": logit_model.bse.values,
            "p_value": logit_model.pvalues.values,
        })
        write_table(client, f"{PROJECT}.{ML_DATASET}.cxconvert_{track}_explainability_v1", out)


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    for track in ("event", "plus"):
        run_p_create(client, track)
        run_p_convert(client, track)


if __name__ == "__main__":
    main()
