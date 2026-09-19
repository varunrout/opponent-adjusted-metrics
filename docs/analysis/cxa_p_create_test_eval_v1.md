# CxA P_create Sealed Test Evaluation v1 (Step 10)

Date: 2026-09-19
The ONE-TIME sealed test run for both CxA P_create tracks, per
[`docs/cxa_split_policy_and_parallel_plan.md`](cxa_split_policy_and_parallel_plan.md)
step 10. Built on
[`docs/analysis/cxa_p_create_freeze_v1.md`](cxa_p_create_freeze_v1.md) (frozen model
family, feature lists, hyperparameters -- read directly from
`oam_ml.cxa_{track}_frozen_v1_config` by
[`scripts/materialize_cxa_test_eval_v1.py`](../../scripts/materialize_cxa_test_eval_v1.py),
not re-typed, to eliminate any risk of drift from what was actually frozen) and
[`docs/analysis/cxa_p_create_baseline_and_candidate_v1.md`](cxa_p_create_baseline_and_candidate_v1.md)
(the validation-stage numbers this document compares against).

**Test was read exactly once per track, for this run only.** No alternative was
evaluated on test and compared -- see section 1 for the one refit decision made
before test was touched, and section 5 for confirmation this was a single read.

## 1. Refit decision (stated up front, as required)

Each track's frozen model was **refit on train+validation combined**, using the exact
feature list and exact hyperparameters read live from
`oam_ml.cxa_{track}_frozen_v1_config`. This is standard practice once hyperparameters
are chosen: validation's only remaining job after model/hyperparameter selection is
to serve as additional training signal for the final model, since it can no longer
leak into a decision that has already been made (the decision -- model family,
feature set, hyperparameters -- was finalized in the freeze step, before this task
began). The XY baseline (`v1`) was refit the same way, for a fair, consistent
comparison across the whole ladder. The dumb baseline's constant is the
train+validation creation rate, same logic.

**This script did not also evaluate the validation-fit models on test for
comparison.** Doing that and picking whichever looked better would be exactly the
test-leakage-through-model-selection this step exists to avoid -- the task was explicit
that this decision, once made, is not something to go back and second-guess based on
results.

## 2. Test metrics -- full ladder, both tracks

Fit pool (train+validation combined): event 515,923 rows (422,946 + 92,977), CxA+
114,573 rows (95,083 + 19,490). Test (read once): event 92,799 rows / 92 matches,
CxA+ 18,570 rows / 23 matches.

### Event-only

| model | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|
| dumb_baseline | 92,799 | 0.09296 | -- | -- (undefined, constant predictor) |
| v1 (XY logistic) | 92,799 | 0.07483 | 0.01739 | 0.8635 |
| **frozen_tree** | 92,799 | **0.06536** | **0.01634** | **0.9153** |

### CxA+

| model | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|
| dumb_baseline | 18,570 | 0.10791 | -- | -- (undefined, constant predictor) |
| v1 (XY logistic) | 18,570 | 0.07610 | 0.01897 | 0.9026 |
| **frozen_tree** | 18,570 | **0.05673** | **0.01481** | **0.9565** |

The frozen tree model beats both baselines by a wide margin in both tracks, matching
the pattern established at validation -- the locked feature set's signal, and the tree
family's edge over logistic, both hold up on genuinely unseen matches.

## 3. Calibration (test, decile-binned predicted vs actual)

Full 10-decile tables in `oam_ml.cxa_{track}_test_v1_calibration`. Top (highest-risk)
decile, the part of the curve that matters most for a ranking/chance-creation model:

| track | model | decile 9 mean predicted | decile 9 mean actual |
|---|---|---|---|
| event | v1 | 0.1010 | 0.1016 |
| event | frozen_tree | 0.1188 | 0.1218 |
| plus | v1 | 0.1458 | 0.1513 |
| plus | frozen_tree | 0.1714 | 0.1858 |

Event-only calibration is essentially exact in the top decile for both models. CxA+
shows a modest **under-confidence** in the top decile for both models (predicted
somewhat lower than actual, most visible for `frozen_tree`: 0.1714 predicted vs 0.1858
actual) -- the model is not overconfident, if anything slightly the opposite, on
CxA+'s highest-risk passes. Given CxA+ test is only 18,570 rows across 1,857-row
deciles, a gap of this size is consistent with the sample-size-driven variance
discussed in section 4, not a systematic calibration failure.

## 4. Test vs validation: honest comparison, gaps named and explained

**Caveat stated before the numbers:** the validation-stage numbers below were produced
by a model fit on **train only**; the test-stage numbers were produced by a model fit
on **train+validation combined** (section 1's refit decision). This is standard
practice and the reason a same-population comparison isn't possible here, but it means
part of any test-vs-validation difference is genuinely attributable to the final fit
having ~22% more training data (event) or ~20% more (CxA+), not only to
generalization. Both effects are named below rather than only attributing everything
to one cause.

### Event-only

| model | metric | validation (train-only fit) | test (train+val fit) | delta |
|---|---|---|---|---|
| v1 | log_loss | 0.07386 | 0.07483 | +0.00097 (1.3% higher) |
| v1 | roc_auc | 0.8658 | 0.8635 | -0.0023 |
| frozen_tree | log_loss | 0.06551 | 0.06536 | -0.00015 (slightly lower) |
| frozen_tree | roc_auc | 0.9127 | 0.9153 | **+0.0026** |

**Small, expected-direction gaps, nothing to flag.** `v1`'s slight degradation
(log_loss up 1.3%, AUC down 0.0023) is the ordinary, small generalization gap expected
from validation to an unseen match set -- both splits are the same size (92 matches
each), so this isn't a small-sample artifact, just normal variance for a 2-feature
linear baseline. `frozen_tree` actually reads *slightly better* on test than
validation -- plausible and unremarkable given the final fit had the additional
92,977 validation rows to train on, and not a large enough move to warrant suspicion
of anything unusual.

### CxA+

| model | metric | validation (train-only fit) | test (train+val fit) | delta |
|---|---|---|---|---|
| v1 | log_loss | 0.07500 | 0.07610 | +0.00110 (1.5% higher) |
| v1 | roc_auc | 0.8944 | 0.9026 | +0.0082 |
| frozen_tree | log_loss | 0.05879 | 0.05673 | -0.00206 (3.5% lower) |
| frozen_tree | roc_auc | 0.9486 | 0.9565 | **+0.0079** |

**A real gap, named and explained, not explained away.** Both models read noticeably
*better* on test than validation for CxA+ -- `frozen_tree`'s AUC moves from 0.9486 to
0.9565, a larger jump than anything seen in the event-only comparison. Two honest
contributing factors, not one tidy story:

1. **More training data.** The train+validation refit added 19,490 rows (~20% more
   than train alone) -- some real improvement from this is expected, same as
   event-only's smaller version of the same effect.
2. **Small test-set variance.** CxA+'s test split is only **23 matches / 18,570
   rows** -- roughly a quarter the size of the event-only test set. A metric computed
   on 23 matches has meaningfully more sampling variance than one computed on 92, and
   a move of this size (+0.008 AUC) is well within what a different 23-match draw
   from the same underlying population could produce by chance alone. This is not a
   reason to distrust the result -- CxA+'s test performance is genuinely strong -- but
   it is a reason not to treat "CxA+ generalizes even better than validation
   suggested" as a confirmed finding. The honest statement is: CxA+'s frozen model
   performs at least as well on held-out tournament matches as validation indicated,
   and plausibly better, but the exact margin should not be over-read given the
   test population's size.

**No gap in either track is large enough to suggest overfitting to validation during
feature selection or hyperparameter search** -- every test metric is close to or
better than its validation counterpart, which is the outcome a properly-sealed test
split should produce when the upstream process (feature lock, model comparison,
freeze) was done correctly.

## 5. Confirmation: single sealed read

`materialize_cxa_test_eval_v1.py` is the only script in this project's history that
ever passes `splits=("train", "validation", "test")` to `load_track` -- every other
CxA modelling script (`materialize_cxa_baseline_v1.py`,
`materialize_cxa_candidate_v1.py`, `materialize_cxa_freeze_v1.py`) uses the
`("train", "validation")` default and never queries test. This script was run once,
end to end, no re-runs, no alternative configurations tried against test, no
comparison of the train-only-fit model against the train+validation-fit model on test
(section 1). `oam_ml.cxa_event_frozen_v1_config` and
`oam_ml.cxa_plus_frozen_v1_config` were read-only inputs, never modified.

## 6. P_create is complete

Both CxA P_create tracks now have a frozen, test-evaluated model:

- **Event-only:** LightGBM tree, 10 locked features, test AUC 0.9153, test log_loss
  0.06536.
- **CxA+:** LightGBM tree, 9 locked features + shared position base, test AUC 0.9565,
  test log_loss 0.05673 (tournament-only population, per every prior CxA+ document's
  generalization caveat -- restated here since it still applies to this final result).

This closes out P_create. **P_convert (Y_goal | Y_create = TRUE) is the next, separate
stage**, per the CxA methodology (P_create x P_convert = CxA) and explicitly out of
scope for this task. It will need its own feasibility check, split-aware feature
work, and model ladder, conditioned on the `Y_create = TRUE` population this stage
has now finished characterizing and modelling -- not started here.
