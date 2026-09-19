# CxA P_create Freeze v1 (Step 9)

Date: 2026-09-19
Freezes model family, feature set, and hyperparameters for **both** CxA P_create
tracks (event-only, CxA+), per split policy step 9
([`docs/cxa_split_policy_and_parallel_plan.md`](cxa_split_policy_and_parallel_plan.md)).
Built on, not re-derived from:
[`docs/analysis/cxa_p_create_baseline_and_candidate_v1.md`](cxa_p_create_baseline_and_candidate_v1.md)
(the baseline/candidate comparison),
[`docs/analysis/cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md),
[`docs/analysis/cxa_plus_p_create_feature_lock_v1.md`](cxa_plus_p_create_feature_lock_v1.md),
and
[`docs/analysis/cxa_p_create_locked_feature_eda_v1.md`](cxa_p_create_locked_feature_eda_v1.md).

Reproducible via
[`scripts/materialize_cxa_freeze_v1.py`](../../scripts/materialize_cxa_freeze_v1.py);
raw search output under
[`audit_outputs/cxa_analysis/freeze_v1/`](../../audit_outputs/cxa_analysis/freeze_v1/).

**Test split was not touched.** See section 5 for the query and counts confirming
this.

## 1. Final decision: LightGBM (tree) for both tracks

**Frozen model family: `lightgbm_tree`, both event-only and CxA+.** Logistic
regression is **not** carried forward as a frozen candidate. Justification, from the
comparison doc's own findings (not re-derived here):

- Tree beat logistic on every metric (`log_loss`, `brier_score`, `roc_auc`) in both
  tracks on validation: event AUC 0.9116 (tree) vs 0.9009 (logistic); CxA+ AUC
  0.9482 (tree) vs 0.9418 (logistic).
- CxA+'s logistic candidate has a specific, confirmed reliability problem: the
  `reception_geometry_missing` coefficient is degenerate (quasi-complete separation --
  coefficient -21.82, std_error 470,161, p=1.0; all 134 affected rows across train and
  validation, 99 + 35, have `y_create = FALSE`). This is not a reason to distrust
  CxA+'s logistic *metrics* (they were sound), but it is a concrete, specific reason
  not to freeze that fit as a production model when a family with no such problem
  (tree) outperforms it anyway.
- The comparison doc's own section 6 recommendation was tree for both tracks, more
  strongly for CxA+ given the above.

**Logistic regression remains available, but only as a coefficient-interpretability
side-model if ever needed later** (e.g. for a stakeholder-facing explanation of
feature direction/magnitude, or a sanity-check reference) -- it is not the frozen
predictor for either track, and if it is ever used for CxA+, the separation issue
above must be fixed first (drop the 134 affected rows from that fit or refit with an
L2 penalty), per the comparison doc's own recommendation.

## 2. Frozen feature sets (exact, no additions)

**Event-only (10 features)**, per
[`cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md):

1. `is_cross`
2. `is_through_ball`
3. `pass_type_name` (Corner, Free Kick levels)
4. `start_x`
5. `end_x`
6. `play_pattern_name` (Counter, Corner levels)
7. `pass_technique_name` (Outswinging level)
8. `is_switch`
9. `is_cut_back`
10. `pass_body_part_name` (No Touch level)

**CxA+ (9 features + shared `start_x`/`end_x` base)**, per
[`cxa_plus_p_create_feature_lock_v1.md`](cxa_plus_p_create_feature_lock_v1.md):

1. `is_cross`
2. `is_through_ball`
3. `pass_type_name` (Corner, Free Kick levels)
4. `play_pattern_name` (Counter, Corner levels)
5. `pass_technique_name` (Outswinging level)
6. `is_cut_back`
7. `is_switch`
8. `reception_nearest_opponent_distance_m`
9. `reception_opponents_within_5m`
10. `start_x` (shared base, reference per the lock doc)
11. `end_x` (shared base, reference per the lock doc)

`pass_body_part_name = No Touch` stays **excluded** (held out per the lock doc's own
recommendation -- thin support, 22 validation rows, only 1 positive). `pass_technique_
name = Straight` stays **excluded** (25 total rows population-wide, per the EDA doc's
rare-level finding -- pooled into the reference category, never its own feature). No
feature outside either locked list was added; the encoding (one-hot per confirmed
locked level only, `start_x` clipped to `[0,120]`, CxA+'s 154 missing-360-geometry rows
flagged + train-median-imputed) is unchanged from
[`scripts/_cxa_modeling_common.py`](../../scripts/_cxa_modeling_common.py), reused
as-is by `materialize_cxa_freeze_v1.py`.

## 3. Hyperparameter search (capped, train/validation only)

8 configurations per track (the baseline untuned config already fit in
`materialize_cxa_candidate_v1.py`, plus 7 reasoned single/paired variations covering
the learning-rate/tree-count trade-off, depth/complexity in both directions, stronger
regularization, and no subsampling). Selection rule, stated before results were
examined: **lowest validation `log_loss` wins; `roc_auc` breaks ties within 0.0005
log_loss** (log_loss chosen as primary because it is sensitive to calibration as well
as ranking, which matters for a model whose output is used as a probability).

### Event-only

| cfg | label | n_estimators | max_depth | num_leaves | learning_rate | min_child_samples | subsample | colsample_bytree | val log_loss | val roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline (untuned) | 200 | 4 | 15 | 0.05 | 50 | 0.8 | 0.8 | 0.06570 | 0.91158 |
| 1 | more trees, lower LR | 400 | 4 | 15 | 0.03 | 50 | 0.8 | 0.8 | 0.06563 | 0.91196 |
| 2 | fewer trees, higher LR | 100 | 4 | 15 | 0.10 | 50 | 0.8 | 0.8 | 0.06581 | 0.91119 |
| **3** | **deeper trees** | **200** | **6** | **31** | **0.05** | **50** | **0.8** | **0.8** | **0.06551** | **0.91270** |
| 4 | shallower trees | 200 | 3 | 7 | 0.05 | 50 | 0.8 | 0.8 | 0.06583 | 0.91099 |
| 5 | stronger regularization | 200 | 4 | 15 | 0.05 | 100 | 0.8 | 0.8 | 0.06568 | 0.91182 |
| 6 | no subsampling | 200 | 4 | 15 | 0.05 | 50 | 1.0 | 1.0 | 0.06566 | 0.91185 |
| 7 | more trees + deeper | 300 | 5 | 25 | 0.04 | 50 | 0.8 | 0.8 | 0.06553 | 0.91264 |

**Chosen: cfg 3 (deeper trees)** -- lowest validation log_loss (0.06551), also the
highest validation AUC of the 8. Improvement over the untuned baseline (cfg 0) is
small: log_loss 0.06570 -> 0.06551 (**0.29% relative improvement**), AUC +0.00112.
This is a real, if modest, improvement, not noise -- cfg 3 and cfg 7 (the two
deeper-tree variants) are the clear top two on both metrics, while every shallower or
higher-learning-rate variant underperforms the baseline. **Frozen for event-only:**
`n_estimators=200, max_depth=6, num_leaves=31, learning_rate=0.05,
min_child_samples=50, subsample=0.8, colsample_bytree=0.8`.

### CxA+

| cfg | label | n_estimators | max_depth | num_leaves | learning_rate | min_child_samples | subsample | colsample_bytree | val log_loss | val roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline (untuned) | 200 | 4 | 15 | 0.05 | 50 | 0.8 | 0.8 | 0.05892 | 0.94816 |
| 1 | more trees, lower LR | 400 | 4 | 15 | 0.03 | 50 | 0.8 | 0.8 | 0.05897 | 0.94828 |
| 2 | fewer trees, higher LR | 100 | 4 | 15 | 0.10 | 50 | 0.8 | 0.8 | 0.05954 | 0.94717 |
| 3 | deeper trees | 200 | 6 | 31 | 0.05 | 50 | 0.8 | 0.8 | 0.05941 | 0.94631 |
| 4 | shallower trees | 200 | 3 | 7 | 0.05 | 50 | 0.8 | 0.8 | 0.05914 | 0.94779 |
| **5** | **stronger regularization** | **200** | **4** | **15** | **0.05** | **100** | **0.8** | **0.8** | **0.05879** | **0.94860** |
| 6 | no subsampling | 200 | 4 | 15 | 0.05 | 50 | 1.0 | 1.0 | 0.05903 | 0.94826 |
| 7 | more trees + deeper | 300 | 5 | 25 | 0.04 | 50 | 0.8 | 0.8 | 0.05930 | 0.94645 |

**Chosen: cfg 5 (stronger regularization, `min_child_samples=100`)** -- lowest
validation log_loss (0.05879), also the highest validation AUC of the 8. Improvement
over the untuned baseline is small: log_loss 0.05892 -> 0.05879 (**0.22% relative
improvement**), AUC +0.00044. Notably, CxA+'s *deeper*-tree variants (cfg 3, cfg 7)
**underperform** the baseline here, the opposite pattern from event-only -- consistent
with CxA+'s much smaller training population (95,083 rows vs event-only's 422,946):
more complex trees overfit faster on less data, and stronger regularization
(`min_child_samples=100`, i.e. requiring more support per leaf) helps rather than
hurts. **Frozen for CxA+:** `n_estimators=200, max_depth=4, num_leaves=15,
learning_rate=0.05, min_child_samples=100, subsample=0.8, colsample_bytree=0.8`.

**Overall reading of the search:** in both tracks the untuned baseline was already
close to optimal -- no configuration in either 8-config grid beats it by more than
~0.3% relative log_loss. This is expected for a first candidate that was already a
"reasonable, conservative default" (per `materialize_cxa_candidate_v1.py`'s own
docstring) rather than an arbitrary guess, and confirms the light tuning pass did not
need to (and did not) find a dramatically different configuration -- it is being
reported honestly as a marginal, not a transformative, improvement.

## 4. Frozen artifacts written

- `oam_ml.cxa_event_frozen_v1_config` (1 row): `model_family`, `feature_list`
  (array, the 10 features above), `n_estimators`, `max_depth`, `num_leaves`,
  `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`,
  `selection_rule`, `chosen_config_label`, `chosen_config_index`, `train_log_loss`,
  `train_roc_auc`, `val_log_loss`, `val_roc_auc`, `source_docs` (array), `frozen_at`.
- `oam_ml.cxa_plus_frozen_v1_config` (1 row): same schema, the 11-entry CxA+ feature
  list above.
- `oam_ml.cxa_event_freeze_v1_search` / `oam_ml.cxa_plus_freeze_v1_search` (8 rows
  each): the full search grid results tabulated in section 3, for audit/reproduction.

## 5. Confirmation: test split was not touched

`materialize_cxa_freeze_v1.py`'s data loader
(`_cxa_modeling_common.load_track`) only ever queries
`WHERE m.split IN UNNEST(@splits)` with `@splits = ('train', 'validation')` -- `test`
is never passed as a parameter anywhere in this script, so it was structurally
impossible for the search or the frozen fit to read a single test row. Independently,
a read-only row-count query (counting only -- no feature or label value from test was
read for any modelling purpose) confirms the test split's size is unchanged from every
prior step in this project:

| track | split | matches | rows |
|---|---|---|---|
| event | train | 426 | 422,946 |
| event | validation | 92 | 92,977 |
| event | **test** | **92** | **92,799** |
| plus | train | 119 | 95,083 |
| plus | validation | 24 | 19,490 |
| plus | **test** | **23** | **18,570** |

These match every prior step's reported split sizes exactly (feature-lock docs,
baseline/candidate doc) -- test has not been touched, resized, or read for anything
beyond this row count in this task.

## 6. Ready for step 10

Step 9 (this document) is complete: model family, feature set, and hyperparameters are
frozen for both tracks, backed by written BigQuery config tables. **Step 10 (the
sealed test run for final unbiased model reporting) is a separate, subsequent task and
requires explicit go-ahead before it runs** -- per the split policy, test is run
exactly once, for final reporting only, after everything upstream of it is settled and
reviewed. This document does not initiate that run.
