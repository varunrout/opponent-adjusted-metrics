# CxA P_create Baseline and Candidate Model Comparison v1

Date: 2026-09-19
Rungs 1-3 of the model ladder (dumb baseline -> XY baseline -> candidate), mirroring
CxG's exact ladder shape (`oam_ml.cxg_baseline_v1_metrics` / `cxg_event_v3_metrics`:
model names `dumb_baseline`, `v1`, `v3`; metrics `log_loss`, `roc_auc`,
`brier_score` by split). Two parallel tracks, same as CxG/CxG+: **CxA event-only**
and **CxA+**.

Built on:
[`docs/cxa_split_policy_and_parallel_plan.md`](cxa_split_policy_and_parallel_plan.md)
(baseline spec, split table), 
[`docs/analysis/cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md)
(10 locked event-only features),
[`docs/analysis/cxa_plus_p_create_feature_lock_v1.md`](cxa_plus_p_create_feature_lock_v1.md)
(9 locked + 1 held-out CxA+ features), and
[`docs/analysis/cxa_p_create_locked_feature_eda_v1.md`](cxa_p_create_locked_feature_eda_v1.md)
(encoding fixes, applied below and restated in
[`scripts/_cxa_modeling_common.py`](../../scripts/_cxa_modeling_common.py)'s module
docstring).

**Test stays sealed.** Every number below is train or validation only; the loader in
`_cxa_modeling_common.py` only ever queries `split IN ('train', 'validation')`.
**No feature set or hyperparameters are frozen by this document** -- that is split
policy step 9, a separate, later, reviewed decision. This document is a comparison for
review, not a freeze.

Split sizes (both tracks, restated from the feature-lock docs): event-only train
422,946 rows / 426 matches, validation 92,977 rows / 92 matches; CxA+ train 95,083
rows / 119 matches, validation 19,490 rows / 24 matches.

## Encoding fixes applied (per the EDA doc, stated not skipped)

1. `pass_technique_name = Straight` (CxA+, 25 total rows) -- pooled into reference; no
   separate dummy created, only `Outswinging` is its own column.
2. `start_x` -- clipped to `[0, 120]` everywhere it's used (baseline and candidate,
   both tracks).
3. CxA+'s 154 rows (99 train / 35 validation, confirmed live -- see "A model-fit issue
   worth reporting honestly" below) with no computed 360 geometry: **flagged, not
   dropped.** An explicit `reception_geometry_missing` indicator is added, and
   `reception_nearest_opponent_distance_m` is imputed with the TRAIN split's median
   (6.562m, fit train-only) while `reception_opponents_within_5m` is imputed with 0.
4. `pass_body_part_name = No Touch` -- excluded from the CxA+ candidate feature set
   (stays held out per the lock doc); included in the event-only candidate set (it IS
   one of the 10 locked event-only features).

## 1. Dumb baseline

Predicts the TRAIN split's creation rate as a constant for every row. `log_loss` only
-- a constant predictor has no rank information, so `roc_auc` is undefined, not just
poor, and is reported as `NULL` in `oam_ml.cxa_{track}_baseline_v1_metrics` rather than
a placeholder value.

| track | split | n | log_loss |
|---|---|---|---|
| event | train | 422,946 | 0.09235 |
| event | validation | 92,977 | 0.09214 |
| plus | train | 95,083 | 0.10167 |
| plus | validation | 19,490 | 0.10402 |

## 2. XY baseline (model name `v1`)

Logistic regression on `start_x`, `start_y` **only** -- per the split policy's
baseline spec, deliberately not the locked feature set.

| track | split | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|---|
| event | train | 422,946 | 0.07446 | 0.01728 | 0.8628 |
| event | validation | 92,977 | 0.07386 | 0.01718 | 0.8658 |
| plus | train | 95,083 | 0.07316 | 0.01799 | 0.8987 |
| plus | validation | 19,490 | 0.07500 | 0.01835 | 0.8944 |

Already a large jump over the dumb baseline in both tracks -- pass origin/destination
alone recovers most of the "is this near goal" signal, as expected (position is the
single strongest univariate signal in both feature-lock docs).

## 3. Candidate models -- full locked feature set, two families

### 3a. Logistic (`v_candidate_logistic`) -- statsmodels `Logit`, same fitting
approach as CxG's `v3`, so a coefficients table with `std_error`/`p_value` could be
produced (`oam_ml.cxa_{track}_candidate_v1_coefficients`, mirrors
`cxg_event_v3_coefficients`).

### 3b. Tree (`v_candidate_tree`) -- LightGBM gradient boosting, untuned first-pass
hyperparameters (`n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
min_child_samples=50, subsample=0.8, colsample_bytree=0.8`) -- reasonable defaults for
a shallow, regularized first candidate given the ~2% positive rate, explicitly not
tuned (tuning is out of scope for this task).

| track | model | split | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|---|---|
| event | logistic | train | 422,946 | 0.06965 | 0.01693 | 0.8981 |
| event | logistic | validation | 92,977 | 0.06906 | 0.01685 | 0.9009 |
| event | tree | train | 422,946 | 0.06562 | 0.01629 | 0.9121 |
| event | tree | validation | 92,977 | 0.06571 | 0.01629 | 0.9116 |
| plus | logistic | train | 95,083 | 0.05895 | 0.01486 | 0.9434 |
| plus | logistic | validation | 19,490 | 0.06126 | 0.01558 | 0.9418 |
| plus | tree | train | 95,083 | 0.05269 | 0.01370 | 0.9597 |
| plus | tree | validation | 19,490 | 0.05892 | 0.01518 | 0.9482 |

Both families comfortably beat the XY baseline in both tracks (see section 4). Train
vs validation gaps are small for both families in both tracks (largest gap:
CxA+ tree, log_loss 0.0527 train -> 0.0589 validation, roc_auc 0.9597 -> 0.9482) --
some train/validation drift as expected for a tree model on a ~2% positive-rate
target, not alarming overfitting.

### Calibration (validation, decile-binned predicted-vs-actual)

Both families are well-calibrated across deciles on validation in both tracks --
predicted and actual rates track closely, including in the top (highest-risk) decile,
which is the part of the calibration curve that matters most for a chance-creation
model used to rank passes:

| track | model | decile 9 (top) mean predicted | decile 9 mean actual |
|---|---|---|---|
| event | logistic | 0.1159 | 0.1140 |
| event | tree | 0.1171 | 0.1229 |
| plus | logistic | 0.1658 | 0.1688 |
| plus | tree | 0.1665 | 0.1714 |

No family shows the kind of systematic over/under-confidence in the top decile that
would make its ranking untrustworthy. Full 10-decile tables for both splits and both
model families are in `oam_ml.cxa_{track}_candidate_v1_calibration`.

## 4. Comparison: baseline (v1) vs both candidate families, validation only

Mirroring CxG's v1->v3 reporting style (plain log_loss/AUC delta statement):

**Event-only:**

| comparison | log_loss delta | relative log_loss improvement | roc_auc delta |
|---|---|---|---|
| v1 -> logistic candidate | 0.07386 -> 0.06906 | **6.5% lower** | 0.8658 -> 0.9009 (**+0.0351**) |
| v1 -> tree candidate | 0.07386 -> 0.06571 | **11.0% lower** | 0.8658 -> 0.9116 (**+0.0458**) |

**CxA+:**

| comparison | log_loss delta | relative log_loss improvement | roc_auc delta |
|---|---|---|---|
| v1 -> logistic candidate | 0.07500 -> 0.06126 | **18.3% lower** | 0.8944 -> 0.9418 (**+0.0475**) |
| v1 -> tree candidate | 0.07500 -> 0.05892 | **21.4% lower** | 0.8944 -> 0.9482 (**+0.0538**) |

**Both candidate families beat the XY baseline by a wide margin in both tracks.** The
locked feature set is carrying real, substantial signal beyond raw pass position --
expected given the feature-lock docs' own split-confirmed lifts (up to 35.9x for
`is_cross` on CxA+), but this is the first time that signal has been shown to hold up
inside an actual fitted model rather than univariate lift tables. The tree family
outperforms the logistic family on every metric in both tracks, by a modest but
consistent margin (event: +0.0107 AUC; CxA+: +0.0064 AUC).

## 5. A model-fit issue worth reporting honestly

The CxA+ logistic candidate's `reception_geometry_missing` coefficient is
**degenerate**: `coefficient = -21.82`, `std_error = 470,161`, `p_value = 1.0`. This is
the textbook signature of quasi-complete separation, and it is real, not a
computation bug -- confirmed live: **all 99 train rows and all 35 validation rows**
with `reception_geometry_missing = 1` have `y_create = FALSE`. With zero positive
examples in that flag's TRUE group, maximum-likelihood logistic regression drives the
coefficient toward negative infinity trying to perfectly separate them, and the
reported standard error is meaningless at that scale.

**Practical impact is small but not zero.** This affects one coefficient out of 14,
covering 134/114,573 rows (0.12%) -- the overall CxA+ logistic metrics and calibration
above are not materially compromised (log_loss/AUC/calibration all look sound), but
this specific coefficient cannot be interpreted, and separation of this kind can subtly
destabilize a maximum-likelihood fit's *other* coefficients too, not just the separated
one. Plausibly relevant: `reception_nearest_opponent_distance_m`'s fitted coefficient
is **positive** (`+0.0565`, p<0.001) -- the opposite sign from its own univariate
relationship (lower distance -> higher creation, confirmed repeatedly in the pre-model
analysis and both feature-lock docs). This is not necessarily wrong on its own (a
sign flip under multivariate control is a known, legitimate suppression effect once a
correlated feature -- here `reception_opponents_within_5m`, r=-0.64 with distance per
the EDA doc -- is also in the model), but it should not be trusted as a clean,
standalone reading without first fixing the separation issue and re-fitting, since the
two are related by construction (same near-100%-empty TRUE group problem touching
adjacent coefficients in a small-sample corner of the design matrix).

**Recommendation for the freeze stage (not decided here):** either (a) drop the 134
affected rows from the logistic fit specifically (they contribute no positive
examples and are demonstrably destabilizing one coefficient), or (b) refit the
logistic candidate with an L2 penalty, which resolves separation by construction. The
tree candidate has no equivalent problem -- gradient boosting isolates the
`reception_geometry_missing = 1` rows into their own leaf/split with a finite,
well-behaved probability estimate, which is itself a point in favor of the tree
family for this track (see recommendation below).

Full coefficient tables (all features, both tracks) are in
`oam_ml.cxa_{track}_candidate_v1_coefficients`. Every other coefficient in both
tracks has a normal, finite standard error and a sign consistent with its
feature-lock-confirmed direction (e.g. event-only `is_cross` +0.564, `is_cut_back`
+0.511, both well-determined with p<0.001 despite their EDA-documented ~65% overlap --
no wild coefficient inflation from that correlation, a good sign the rest of the
event-only fit is well-behaved).

## 6. Recommendation (input to the freeze decision, not the freeze decision itself)

**Event-only:** the tree candidate shows a consistent, if modest, edge over the
logistic candidate on every metric (log_loss 0.0657 vs 0.0691, AUC 0.9116 vs 0.9009,
both validation) with no fit-stability issue of any kind. **Tree is the more promising
family for this track**, with logistic retained as the interpretable comparison point
(the same role CxG's `v1` plays relative to `v3`) -- not because logistic performed
badly (it beat the XY baseline by a wide margin too), but because tree's edge is real
and comes with no caveats.

**CxA+:** the tree candidate again outperforms logistic on every metric (log_loss
0.0589 vs 0.0613, AUC 0.9482 vs 0.9418, both validation), and additionally avoids the
separation problem described in section 5 entirely. **Tree is the more promising
family for this track as well**, more clearly than in the event-only case given the
added fit-stability concern on the logistic side. If an interpretable logistic
candidate is still wanted for CxA+ at freeze time (e.g. for coefficient-based
reporting or a simpler production path), it should be refit with the separation fix
from section 5 first -- the current logistic fit's headline metrics are usable, but
its coefficient table should not be cited as-is.

Neither recommendation freezes anything. Both tracks' final feature set, model family,
and hyperparameters remain open questions for split policy step 9, to be decided after
this comparison is reviewed.
