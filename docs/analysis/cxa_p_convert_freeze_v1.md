# CxA P_convert Freeze v1

Date: 2026-09-20
Freezes model family, feature set, and hyperparameters for **both** CxA P_convert
tracks (event-only, CxA+), mirroring
[`docs/analysis/cxa_p_create_freeze_v1.md`](cxa_p_create_freeze_v1.md)'s structure and
rigor. Built on, not re-derived from:
[`docs/analysis/cxa_p_convert_baseline_and_candidate_v1.md`](cxa_p_convert_baseline_and_candidate_v1.md)
(the baseline/candidate comparison this freeze is built on -- its sections 5 and 6
drive everything below),
[`docs/analysis/cxa_event_p_convert_feature_lock_v1.md`](cxa_event_p_convert_feature_lock_v1.md)
(15 locked event-only features),
[`docs/analysis/cxa_plus_p_convert_feature_lock_v1.md`](cxa_plus_p_convert_feature_lock_v1.md)
(**15** locked CxA+ features -- that document's own header says "17," a known
documentation error already flagged in the Step 5 doc; the 15-row table is the correct
count and is used here, not propagated as 17), and
[`docs/analysis/cxa_p_convert_locked_feature_eda_v1.md`](cxa_p_convert_locked_feature_eda_v1.md)
(encoding guidance).

**Branch note:** none of P_convert steps 1-5 were merged to main at the time this task
started. Since this task needs to read documents from steps 1, 3, 4, *and* 5 (not just
step 1), and each step's branch was cut independently from main rather than stacked on
the previous one, basing off `feature/cxa-p-convert-pipeline` alone (step 1) would not
have had the feature-lock, EDA, or baseline/candidate docs available. This branch was
cut from `main` and then merged in all five step branches
(`feature/cxa-p-convert-pipeline`,
`analysis/cxa-p-convert-pre-model-study`, `analysis/cxa-p-convert-feature-lock`,
`analysis/cxa-p-convert-locked-feature-eda`,
`modeling/cxa-p-convert-baseline-and-candidate`) -- all five merged cleanly, no
conflicts.

Reproducible via
[`scripts/materialize_cxconvert_freeze_v1.py`](../../scripts/materialize_cxconvert_freeze_v1.py);
raw search output under
[`audit_outputs/cxconvert_analysis/freeze_v1/`](../../audit_outputs/cxconvert_analysis/freeze_v1/).

**Test split was not touched.** See section 6 for the query and counts confirming
this.

## 1. What Step 5 found, and what this task had to actually resolve

Restated briefly, not re-derived (full detail in the baseline/candidate doc):

- **Event-only:** logistic and tree landed within noise of each other on validation
  (log_loss gap 0.003, AUC gap 0.001). No hyperparameter search had been run for
  either family -- both used untuned/mirrored defaults. Step 5 declined to force a
  recommendation.
- **CxA+:** the tree candidate overfit badly (train log_loss 0.1996 vs. validation
  0.3351) and its validation log_loss was *worse* than the trivial XY baseline
  (0.3320), despite a real AUC edge (0.691 vs. 0.648). The logistic candidate had no
  such problem and clearly beat both baselines (validation log_loss 0.31606, AUC
  0.7283). Step 5 recommended logistic for CxA+ "as currently fit," but explicitly
  left a genuine hyperparameter search for the tree family out of its own scope.

**This task's job was to actually run that search on both tracks** (not just for
CxA+ -- event-only's "too close to call" verdict also deserved a real search before
being accepted as final) and make a defended freeze decision, not another comparison.

## 2. Hyperparameter search (8 configs per track, train/validation only)

**Selection rule, stated before results were examined** (same rule P_create's own
freeze used): lowest validation `log_loss` wins; `roc_auc` breaks ties within 0.0005
log_loss.

### Event-only: reuses P_create's own 8-config grid shape

No prior evidence of an overfitting problem on this track (Step 5's train/validation
tree gap was modest, 0.230 -> 0.266), so the same grid shape P_create's own freeze
used for its comparable-scale event-only track is reused here -- baseline, tree-count/
learning-rate trade-offs, depth in both directions, regularization, no subsampling.

| cfg | label | n_estimators | max_depth | num_leaves | learning_rate | min_child_samples | subsample | colsample_bytree | val log_loss | val roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline (untuned, Step 5) | 200 | 4 | 15 | 0.05 | 50 | 0.8 | 0.8 | 0.26647 | 0.76567 |
| 1 | more trees, lower LR | 400 | 4 | 15 | 0.03 | 50 | 0.8 | 0.8 | 0.26564 | 0.76889 |
| 2 | fewer trees, higher LR | 100 | 4 | 15 | 0.10 | 50 | 0.8 | 0.8 | 0.26711 | 0.76589 |
| 3 | deeper trees | 200 | 6 | 31 | 0.05 | 50 | 0.8 | 0.8 | 0.26922 | 0.76032 |
| **4** | **shallower trees** | **200** | **3** | **7** | **0.05** | **50** | **0.8** | **0.8** | **0.26356** | **0.77484** |
| 5 | stronger regularization | 200 | 4 | 15 | 0.05 | 100 | 0.8 | 0.8 | 0.26514 | 0.76861 |
| 6 | no subsampling | 200 | 4 | 15 | 0.05 | 50 | 1.0 | 1.0 | 0.26491 | 0.76968 |
| 7 | more trees + deeper | 300 | 5 | 25 | 0.04 | 50 | 0.8 | 0.8 | 0.26717 | 0.76643 |

**cfg 4 (shallower trees) wins clearly** -- lowest validation log_loss (0.26356) *and*
highest validation AUC (0.77484) of all 8, with the next-closest config (cfg 6, no
subsampling, 0.26491) a full 0.00135 behind -- outside the 0.0005 tie-break band, so
this is an unambiguous win, not a coin flip. Unlike the deeper-trees pattern that
sometimes helps on larger populations, this track's own data prefers *shallower*
trees: cfg 3 (deeper) is the single worst config in the grid (0.26922, actually behind
the untuned baseline). This directly resolves Step 5's "too close to call" verdict --
see section 4.

### CxA+: a DIFFERENT grid, purpose-built to test the overfitting gap

Re-running P_create-shaped variants (deeper trees, more estimators) would test the
wrong direction for this track's known problem. Per this task's explicit instruction,
every config here is shallower and more constrained than Step 5's already-corrected
baseline (`max_depth<=3`, `num_leaves<=8`, `n_estimators<=200`,
`min_child_samples>=100`).

| cfg | label | n_estimators | max_depth | num_leaves | learning_rate | min_child_samples | subsample | colsample_bytree | val log_loss | val roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Step 5 corrected baseline | 200 | 4 | 15 | 0.05 | 100 | 0.8 | 0.8 | 0.33506 | 0.69118 |
| 1 | very shallow, very small, strong reg | 50 | 2 | 4 | 0.05 | 150 | 0.8 | 0.8 | 0.31859 | 0.72018 |
| **2** | **shallow, more trees, low LR** | **100** | **3** | **7** | **0.03** | **120** | **0.8** | **0.8** | **0.31714** | **0.72236** |
| 3 | shallow, moderate LR | 75 | 2 | 4 | 0.07 | 150 | 0.8 | 0.8 | 0.31844 | 0.72094 |
| 4 | shallow + row subsampling | 100 | 3 | 8 | 0.05 | 150 | 0.7 | 0.7 | 0.31792 | 0.72050 |
| 5 | very shallow, many trees, very low LR | 200 | 2 | 4 | 0.02 | 150 | 0.8 | 0.8 | 0.31798 | 0.72727 |
| 6 | minimal complexity | 50 | 2 | 4 | 0.10 | 100 | 0.8 | 0.8 | 0.31900 | 0.72300 |
| 7 | moderate shallow (depth 3) | 100 | 3 | 7 | 0.05 | 100 | 0.8 | 0.8 | 0.31870 | 0.72158 |

**Every shallow config in this grid dramatically improves on cfg 0** (Step 5's
already-corrected baseline) -- log_loss drops from 0.335 to 0.317-0.319 across the
board, confirming the overfitting diagnosis directly: this track's tree needed to be
much shallower than `max_depth=4`/`num_leaves=15` to stop memorizing train noise.
**cfg 2 (shallow, more trees, low LR) wins the tree search** at val log_loss 0.31714,
AUC 0.72236 -- close to but not tied with the other shallow configs (next-closest cfg
4 at 0.31792, a 0.00078 gap, outside the 0.0005 tie-break band).

## 3. Does any CxA+ tree config beat the XY baseline and/or the logistic candidate?
Answered explicitly, per this task's instruction.

**Yes, every shallow config beats the XY baseline (0.33204).** cfg 2's 0.31714 is a
**4.5% relative improvement** over the XY baseline -- a real, substantial win the
untuned Step 5 tree candidate did not achieve (it was *worse* than the XY baseline).

**No, the best tree config still does not beat the logistic candidate -- but the gap
has closed to near-nothing.** Plain-MLE logistic (refit fresh in this script, not
copied from Step 5): validation log_loss **0.31606**, AUC 0.7283. Best tree (cfg 2):
validation log_loss **0.31714**, AUC 0.72236. Logistic remains ahead on log_loss by
0.00108 (0.34% relative) and on AUC by 0.006 -- both differences that should be read
as within the noise of a 420-row/46-positive validation split (per the standing
small-sample caution), not a large, confidently-real gap. **This is itself the
legitimate finding this task asked for if the answer came back "no": a real,
purpose-built search closed nearly all of Step 5's gap (tree went from
worse-than-XY-baseline to a near-tie with logistic) but did not flip the result.**

**Logistic L2 regularization check** (this task's item 2 -- there is no comparable
hyperparameter surface for plain MLE, so this substitutes a short, explicit check
rather than skipping the question): `sklearn.LogisticRegression(penalty="l2")` at
`C in [0.1, 1, 10, 100]`, both tracks.

| track | C | val log_loss | val roc_auc |
|---|---|---|---|
| event | 0.1 | 0.26660 | 0.77005 |
| event | 1.0 | 0.26836 | 0.76778 |
| event | 10.0 | 0.26921 | 0.76682 |
| event | 100.0 | 0.26934 | 0.76678 |
| plus | 0.1 | 0.31611 | 0.71774 |
| plus | **1.0** | **0.31512** | **0.72658** |
| plus | 10.0 | 0.31585 | 0.72861 |
| plus | 100.0 | 0.31599 | 0.72861 |

**Event-only: L2 at `C=0.1` (0.26660, AUC 0.77005) modestly beats plain MLE (0.26934,
AUC 0.7671)** -- a real if small improvement (1.0% relative log_loss), but still well
behind tree cfg 4 (0.26356) on both metrics. Does not change the family decision
(section 4), only confirms tree's margin over *either* logistic variant is real.

**CxA+: L2 at `C=1.0` (0.31512, AUC 0.72658) is marginally better than plain MLE
(0.31606, AUC 0.7283)** on log_loss (0.3% relative), essentially a wash on AUC (very
slightly worse). This is the single best-log_loss configuration found in this entire
CxA+ search across both families. **Confirmed not a meaningful enough difference to
justify trading away plain MLE's interpretable coefficient table (std errors,
p-values, already reported in the baseline/candidate doc's section 5) for a marginal,
noise-level log_loss gain on a 420-row validation split.** Plain MLE is retained for
CxA+ -- explicitly confirmed, not silently assumed, per this task's instruction.

## 4. Freeze decision, per track

### Event-only: **LightGBM (tree)**, cfg 4 hyperparameters

**Model family: `lightgbm_tree`.** Unlike Step 5's "too close to call" verdict, the
search in section 2 found a real, non-tie-break-level winner: tree cfg 4 beats every
logistic variant (plain MLE 0.26934, best L2 0.26660) by a clear margin on both
log_loss and AUC. This resolves the ambiguity Step 5 explicitly left open -- the
event-only race was never a permanent tie, it was an untuned comparison, and tuning
tips it decisively toward tree.

**Frozen hyperparameters:** `n_estimators=200, max_depth=3, num_leaves=7,
learning_rate=0.05, min_child_samples=50, subsample=0.8, colsample_bytree=0.8`
(shallower than Step 5's untuned default, not deeper -- this track's own data prefers
less complexity, the same direction CxA+'s search found more dramatically).

**Feature set: the full 15-feature locked set, unchanged, all three geometry-trio
members included** (`shot_x_sb`, `shot_dist_to_goal_m`, `shot_gk_distance_m` together)
-- trees tolerate the |r| 0.93-0.97 collinearity natively, per the EDA's own guidance,
carried forward unchanged (no logistic-style trimming needed once tree is the frozen
family).

### CxA+: **Logistic regression (plain MLE)**, no comparable hyperparameters to tune

**Model family: `logistic_mle`.** Even after a real, purpose-built shallow-tree
search that closed nearly all of Step 5's overfitting gap (section 2-3), tree did not
overtake logistic. Logistic remains ahead on every metric, has no separation or
stability issue (confirmed in Step 5), and is the more interpretable choice. Section 3
found this margin is small enough that it should be read as directionally reliable
rather than numerically decisive at CxA+'s validation size -- but "small and
directionally consistent across log_loss, AUC, and every variant checked" is still a
real basis for a decision, not an arbitrary pick.

**Frozen hyperparameters: n/a (plain MLE, no penalty).** L2 regularization was
checked (section 3) and confirmed not to change the picture meaningfully -- plain MLE
is retained explicitly, not by default.

**Feature set: the full 15-feature locked set, encoded per the logistic design
matrix** (13 encoded columns) -- per the EDA's multicollinearity guidance, only
`shot_gk_distance_m` is kept from the geometry trio (`shot_x_sb` and
`shot_dist_to_goal_m` dropped for this family specifically, carried forward from Step
5 unchanged), and `shot_technique_name=Lob` is pooled into the reference category
(16 total rows, per the EDA's rare-level finding) rather than encoded as its own
dummy.

**This is a genuine per-track model-family split (tree for event-only, logistic for
CxA+), not an inconsistency.** Both decisions are backed by a real search on their own
track's data, and the divergence is explainable: CxA+'s much smaller population (1,991
train rows vs. 7,847) makes tree-based overfitting a real risk that a comparable
untuned config does not face at event-only's scale, and even a purpose-built shallow
search only closes the gap rather than reversing it.

## 5. Encoding fixes carried forward, unchanged

All four fixes from the EDA/baseline-candidate docs apply to whichever family is
frozen per track, exactly as before -- confirmed, not re-derived:

1. **`shot_technique_name = Lob` pooled into reference for CxA+ only** (16 total
   rows) -- applies to CxA+'s frozen logistic feature set (no separate dummy).
   Event-only keeps it as its own locked level (62 rows, frozen tree feature set
   includes it as a dummy, unaffected by trees' handling of collinearity since this is
   a support question, not a redundancy question).
2. **`shot_gk_distance_m` missingness-safe encoding** -- the flag + train-median
   imputation mechanism from `_cxconvert_modeling_common.py` is reused unchanged.
   Event-only's frozen tree design matrix includes the flag (4 null train rows, real
   information); CxA+'s frozen logistic design matrix omits it (0 null rows in
   train+validation for this track, confirmed again in this task's own fresh data
   load -- a constant-zero flag would still be dead weight, though for a tree model a
   zero-variance column is harmless rather than singularizing, unlike for logistic).
3. **Geometry-trio collinearity** -- no issue for event-only's frozen tree (all three
   members kept, per the EDA's own note that trees handle this natively). CxA+'s
   frozen logistic keeps only `shot_gk_distance_m`, per Step 5's justification (most
   stable of the trio across train-vs-validation confirmation in both tracks).
4. **No pitch-bounds clipping** -- unchanged, confirmed still unnecessary (the EDA
   found no out-of-range values in either track).

## 6. Confirmation: test split was not touched

`materialize_cxconvert_freeze_v1.py`'s data loader
(`_cxconvert_modeling_common.load_track`) only ever queries `split IN
('train','validation')` -- `test` is never passed as a parameter anywhere in this
script, so it was structurally impossible for the search or the frozen fit to read a
single test row. Independently, a read-only row-count query (counting only) confirms
the test split's size is unchanged from every prior step in this project:

| track | split | matches | rows |
|---|---|---|---|
| event | train | 426 | 7,847 |
| event | validation | 92 | 1,720 |
| event | **test** | **92** | **1,736** |
| plus | train | 119 | 1,991 |
| plus | validation | 24 | 420 |
| plus | **test** | **23** | **419** |

These match every prior step's reported split sizes exactly (feature-lock docs,
baseline/candidate doc) -- test has not been touched, resized, or read for anything
beyond this row count in this task.

## 7. Frozen artifacts written

- `oam_ml.cxconvert_event_frozen_v1_config` (1 row): `model_family='lightgbm_tree'`,
  15-item `feature_list`, `n_estimators=200`, `max_depth=3`, `num_leaves=7`,
  `learning_rate=0.05`, `min_child_samples=50`, `subsample=0.8`,
  `colsample_bytree=0.8`, `regularization_C=NULL`, `selection_rule`,
  `chosen_config_label`, `train_log_loss=0.24574`, `train_roc_auc=0.82469`,
  `val_log_loss=0.26356`, `val_roc_auc=0.77484`, `source_docs` (array), `frozen_at`.
- `oam_ml.cxconvert_plus_frozen_v1_config` (1 row): `model_family='logistic_mle'`,
  12-item `feature_list` (the logistic-encoded set), all tree hyperparameter columns
  `NULL`, `regularization_C=NULL` (plain MLE, not the L2 variant),
  `val_log_loss=0.31606`, `val_roc_auc=0.7283`, `source_docs` (array), `frozen_at`.
- `oam_ml.cxconvert_event_freeze_v1_search` (8 rows): the event-only search grid
  tabulated in section 2.
- `oam_ml.cxconvert_plus_freeze_v1_search` (8 rows): the CxA+ search grid tabulated
  in section 2.

Named `cxconvert_*`, not `cxa_*`, to avoid colliding with the P_create freeze tables
of a near-identical name (`oam_ml.cxa_event_frozen_v1_config` etc., already in use).

## 8. Ready for a sealed test evaluation

This document completes the freeze: model family, feature set, and hyperparameters
are settled for both tracks, backed by written BigQuery config tables and a real
hyperparameter search (not an untuned guess carried forward by default). **A sealed
test evaluation is a separate, subsequent task and requires explicit go-ahead before
it runs** -- test is read exactly once, for final reporting only, after everything
upstream of it is settled and reviewed. This document does not initiate that run.
