# CxA P_convert Sealed Test Evaluation v1

Date: 2026-09-20
The ONE-TIME sealed test run for both CxA P_convert tracks, mirroring
[`docs/analysis/cxa_p_create_test_eval_v1.md`](cxa_p_create_test_eval_v1.md)'s
structure and discipline. Built on
[`docs/analysis/cxa_p_convert_freeze_v1.md`](cxa_p_convert_freeze_v1.md) (frozen model
family, feature lists, hyperparameters -- read directly from
`oam_ml.cxconvert_{track}_frozen_v1_config` by
[`scripts/materialize_cxconvert_test_eval_v1.py`](../../scripts/materialize_cxconvert_test_eval_v1.py),
not re-typed, to eliminate any risk of drift from what was actually frozen) and
[`docs/analysis/cxa_p_convert_baseline_and_candidate_v1.md`](cxa_p_convert_baseline_and_candidate_v1.md)
(the validation-stage numbers this document compares against).

**Branch note:** same reasoning as the freeze branch. None of P_convert steps 1-5 or
the freeze step were on `main` yet, and each was an independent branch rather than
stacked, so this branch was cut from `main` and had all six prior branches merged in
(`feature/cxa-p-convert-pipeline`, `analysis/cxa-p-convert-pre-model-study`,
`analysis/cxa-p-convert-feature-lock`, `analysis/cxa-p-convert-locked-feature-eda`,
`modeling/cxa-p-convert-baseline-and-candidate`, `modeling/cxa-p-convert-freeze-v1`) --
all six merged cleanly, no conflicts.

**Test was read exactly once per track, for this run only.** No alternative was
evaluated on test and compared -- see section 1 for the one refit decision made before
test was touched, and section 5 for confirmation this was a single read.

## 1. Refit decision (stated up front, as required)

Each track's frozen model was **refit on train+validation combined**, using the exact
model family, feature list, and hyperparameters read live from
`oam_ml.cxconvert_{track}_frozen_v1_config` -- **not the same family for both tracks**,
unlike P_create: event-only's frozen family is `lightgbm_tree`, CxA+'s is
`logistic_mle`, and this script branches on whichever value the config table actually
contains rather than assuming tree. This is standard practice once
model/hyperparameters are chosen: validation's only remaining job after selection is to
serve as additional training signal for the final model, since it can no longer leak
into a decision that has already been made (the decision was finalized in the freeze
step, before this task began). The XY baseline (`v1`) was refit the same way (on
`shot_x_sb`/`shot_y_sb`, per the baseline/candidate doc's definition), for a fair,
consistent comparison across the whole ladder. The dumb baseline's constant is the
train+validation goal-conversion rate, same logic.

**This script did not also evaluate the validation-fit models on test for
comparison.** Doing that and picking whichever looked better would be exactly the
test-leakage-through-model-selection this step exists to avoid.

## 2. Test metrics -- full ladder, both tracks

Fit pool (train+validation combined): event 9,567 rows (7,847 + 1,720), CxA+ 2,411
rows (1,991 + 420). Test (read once): event 1,736 rows / 92 matches / **141 goals**,
CxA+ **419 rows / 23 matches / 36 goals**.

### Event-only

| model | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|
| dumb_baseline | 1,736 | 0.28271 | -- | -- (undefined, constant predictor) |
| v1 (XY logistic) | 1,736 | 0.26477 | 0.07219 | 0.6949 |
| **frozen_candidate (lightgbm_tree)** | 1,736 | **0.24646** | **0.06841** | **0.7646** |

### CxA+

| model | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|
| dumb_baseline | 419 | 0.29340 | -- | -- (undefined, constant predictor) |
| v1 (XY logistic) | 419 | 0.27776 | 0.07698 | 0.6665 |
| **frozen_candidate (logistic_mle)** | 419 | **0.24964** | **0.07042** | **0.7939** |

The frozen candidate beats both baselines by a real margin in both tracks, matching the
pattern established at validation -- the locked feature set's signal holds up on
genuinely unseen matches in both tracks, regardless of which family was frozen for
each.

## 3. Calibration (test, decile-binned predicted vs actual)

Full 10-decile tables in `oam_ml.cxconvert_{track}_test_v1_calibration`. Top
(highest-risk) decile, the part of the curve that matters most for a model whose
output ranks or scores chance quality:

| track | model | decile 9 mean predicted | decile 9 mean actual | decile n |
|---|---|---|---|---|
| event | v1 | 0.1997 | 0.1839 | 174 |
| event | frozen_candidate | 0.3182 | 0.2586 | 174 |
| plus | v1 | 0.1885 | 0.1190 | 42 |
| plus | frozen_candidate | 0.2742 | 0.2857 | 42 |

**Event-only's frozen candidate is somewhat over-confident in the top decile**
(predicted 0.3182 vs actual 0.2586, a real gap on a 174-row bin, ~45 actual positives
-- large enough that this is not pure noise, though not severe). `v1` is closer
(0.1997 vs 0.1839). Worth noting as a genuine, if modest, calibration softness in the
frozen tree's highest-risk predictions on test -- reported plainly, not something this
task is positioned to fix (no re-tuning based on test, per the task's own constraint).

**CxA+'s frozen candidate is essentially exactly calibrated in the top decile**
(predicted 0.2742 vs actual 0.2857) -- but on a **42-row decile with roughly 12 actual
positives**, where one or two outcomes shifting would move the actual rate by
2.4-4.8 percentage points on their own. This good-looking number should be read with
the same caution as every other CxA+ test statistic in this document, not cited as
proof of excellent calibration.

## 4. Test vs validation: honest comparison, gaps named and explained

**Caveat stated before the numbers:** the validation-stage numbers below were produced
by a model fit on **train only**; the test-stage numbers were produced by a model fit
on **train+validation combined** (section 1's refit decision). Part of any
test-vs-validation difference is genuinely attributable to the final fit having ~22%
more training data (event, 9,567 vs 7,847) or ~21% more (CxA+, 2,411 vs 1,991), not
only to generalization. Both effects are named below rather than attributing everything
to one cause.

### Event-only

| model | metric | validation (train-only fit) | test (train+val fit) | delta |
|---|---|---|---|---|
| v1 | log_loss | 0.28879 | 0.26477 | -0.02402 (8.3% lower) |
| v1 | roc_auc | 0.6856 | 0.6949 | +0.0093 |
| frozen_candidate | log_loss | 0.26356 | 0.24646 | -0.01710 (6.5% lower) |
| frozen_candidate | roc_auc | 0.77484 | 0.76463 | -0.01021 |

**Log_loss improves on test for both models (expected direction, given more training
data); `frozen_candidate`'s AUC dips modestly (-0.0102).** This AUC dip is worth
naming plainly rather than glossing over: it is a real, if small, move in the
"worse" direction on the metric most sensitive to ranking quality. Two honest
candidate explanations, not one tidy story: (1) ordinary generalization variance
moving from a 92-match validation set to a different 92-match test set (both splits
are the same size, so this isn't a small-sample artifact on its own); (2) the
`frozen_candidate`'s validation-stage selection (section 2 of the freeze doc) was
itself chosen by lowest validation log_loss on this exact split, so some of
validation's AUC advantage may reflect a mild selection effect specific to that split
rather than a property that transfers perfectly to a new one. **Not large enough to
call a problem** -- log_loss, the primary metric this whole ladder was selected on,
still improves on test -- but stated honestly as a real, not hand-waved, AUC move.

### CxA+

| model | metric | validation (train-only fit) | test (train+val fit) | delta |
|---|---|---|---|---|
| v1 | log_loss | 0.33204 | 0.27776 | -0.05428 (16.3% lower) |
| v1 | roc_auc | 0.6475 | 0.6665 | +0.0190 |
| frozen_candidate | log_loss | 0.31606 | 0.24964 | **-0.06642 (21.0% lower)** |
| frozen_candidate | roc_auc | 0.7283 | 0.7939 | **+0.0656** |

**A large, real gap -- named and explained, not explained away, per this task's
explicit instruction.** Both models read substantially *better* on test than
validation for CxA+, more dramatically than anything in the event-only comparison.
Two honest contributing factors:

1. **More training data.** The train+validation refit added 420 rows (~21% more than
   train alone) -- some real improvement is expected, the same effect seen at
   event-only's smaller scale, but CxA+'s relative increase is comparable in
   proportion (21% vs event's 22%), so this alone does not explain why CxA+'s
   improvement is so much larger in absolute terms.
2. **Small test-set variance -- the dominant explanation here.** CxA+'s test split is
   only **23 matches / 419 rows / 36 goals** -- roughly a quarter the size of the
   event-only test set (92 matches / 1,736 rows / 141 goals), and CxA+'s validation
   split that the "before" numbers came from is itself only 420 rows / 46 positives.
   A metric computed on 23 matches and 36 positive outcomes has substantially more
   sampling variance than one computed on 92 matches and 141 positives. A log_loss
   move of this size (-0.066) and an AUC move of this size (+0.066) are both well
   within what a different 23-match draw from the same underlying tournament
   population could produce by chance alone. **This is not a reason to distrust the
   result** -- CxA+'s test performance is genuinely strong, and nothing in this
   document found a reason to doubt it -- **but it is a reason not to treat "CxA+
   generalizes dramatically better than validation suggested" as a confirmed
   finding.** The honest statement, consistent with the small-sample caution
   maintained throughout Steps 5 and 6: CxA+'s frozen logistic model performs at
   least as well on held-out tournament matches as validation indicated, and
   plausibly quite a bit better, but the exact margin should not be over-read given
   how few matches and goals this test split actually contains.

**No gap in either track is large enough to suggest the upstream process (feature
lock, model comparison, freeze) went wrong.** Every test log_loss is better than its
validation counterpart in both tracks; the one metric moving in the "worse" direction
(event-only `frozen_candidate` AUC, by 0.0102) is small and has a named, plausible
explanation, not an alarming one.

## 5. CxA+ logistic separation/stability check on the combined refit (checked, per
this task's item 6)

**Worth checking explicitly, since the fit population just grew** (1,991 -> 2,411
rows) **from what was fit at the freeze stage -- a thin group's behavior on train
alone doesn't guarantee its behavior once validation rows are added in.** Checked
directly, not assumed:

- **Every locked boolean/dummy column has at least one positive example in both its
  TRUE and FALSE groups on the combined fit pool** -- the thinnest, `pass_type_Corner`,
  has 9 positives among 214 TRUE rows; `shot_open_goal` has 14 among 21. No zero-
  positive group anywhere (the precondition for quasi-complete separation).
- **Every fitted coefficient's standard error is small and reasonable** -- max
  standard error across all 12 features + intercept is 1.354 (the intercept itself);
  every feature coefficient's own standard error is under 0.5. The largest
  `|coefficient| / std_error` ratio across the fit is 6.16 -- nowhere close to the
  hundreds-or-thousands-scale signature P_create's own CxA+ finding showed for a
  genuinely separated coefficient.
- **Result: no separation or stability issue found on the combined refit.** This is a
  checked non-finding, reported as such, not a silent assumption that "it worked at
  freeze time so it must still work now."

## 6. Confirmation: single sealed read

`materialize_cxconvert_test_eval_v1.py` is the only script in this project's P_convert
history that ever passes `splits=("train", "validation", "test")` to `load_track` --
every other P_convert modelling script (`analyze_cxconvert_baseline_and_candidate.py`,
`materialize_cxconvert_freeze_v1.py`) uses the `("train", "validation")` default and
never queries test. This script was run once, end to end, no re-runs, no alternative
configurations tried against test, no comparison of the train-only-fit model against
the train+validation-fit model on test (section 1). `oam_ml.cxconvert_event_frozen_v1_config`
and `oam_ml.cxconvert_plus_frozen_v1_config` were read-only inputs, never modified.

**Independently confirmed the written output tables contain `split='test'` rows only**
(a follow-up read-only query, not just the script's own internal assertion):

| table | distinct split values | row count |
|---|---|---|
| `cxconvert_event_test_v1_metrics` | `{test}` | 3 |
| `cxconvert_event_test_v1_calibration` | `{test}` | 20 |
| `cxconvert_event_test_v1_predictions` | `{test}` | 1,736 |
| `cxconvert_plus_test_v1_metrics` | `{test}` | 3 |
| `cxconvert_plus_test_v1_calibration` | `{test}` | 20 |
| `cxconvert_plus_test_v1_predictions` | `{test}` | 419 |

No `train` or `validation` rows in any of the six written tables.

## 7. Summary: is P_convert ready to be considered complete?

**Yes, both CxA P_convert tracks now have a frozen, test-evaluated model that beats
its baselines by a real margin and shows no fit-stability problem:**

- **Event-only:** LightGBM tree, 15 locked features, test log_loss 0.24646, test AUC
  0.7646 -- a real, if modest, calibration softness in the top decile (section 3) is
  the only concern worth flagging for a downstream consumer, not severe enough to
  block use but worth knowing if this model's probabilities are read at face value
  for the highest-risk passes specifically.
- **CxA+:** Logistic regression (plain MLE), 12 locked/encoded features, test
  log_loss 0.24964, test AUC 0.7939 -- no separation or stability issue on the
  combined refit (section 5), and a large, real test-vs-validation improvement whose
  magnitude should not be over-read given the sample size (section 4).

**The one standing concern for anyone consuming these models downstream: CxA+'s test
split is very small (419 rows, 36 goals, 23 matches) and its population remains
tournament-only (zero Premier League rows, inherited from every prior P_create/
P_convert document's own caveat) -- restated here honestly, one more time, because it
is the property that most affects how much confidence a downstream user should place
in CxA+'s exact numbers versus its general direction.** Both tracks' models are ready
for use; CxA+'s specific metric values should be treated as directionally reliable
rather than numerically precise, the same standard applied to every CxA+ number in
this project since the split policy doc first flagged the track's smaller scale.

This closes out P_convert. Combined with the already-complete P_create stage, CxA
(P_create x P_convert) now has a full, frozen, test-evaluated model ladder for both
tracks at both stages.
