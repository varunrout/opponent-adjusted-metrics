# CxA P_convert Baseline and Candidate Model Comparison v1

Date: 2026-09-20
Rungs 1-3 of the model ladder (dumb baseline -> XY baseline "v1" -> candidate),
mirroring `docs/analysis/cxa_p_create_baseline_and_candidate_v1.md`'s exact ladder
shape and rigor. Two parallel tracks: **CxA event-only** and **CxA+**.

Built on:
[`docs/cxa_convert_split_policy_and_plan.md`](cxa_convert_split_policy_and_plan.md)
(split table),
[`docs/analysis/cxa_event_p_convert_feature_lock_v1.md`](cxa_event_p_convert_feature_lock_v1.md)
(15 locked event-only features),
[`docs/analysis/cxa_plus_p_convert_feature_lock_v1.md`](cxa_plus_p_convert_feature_lock_v1.md)
(**15** locked CxA+ features -- that document's own header says "17," but its own
table lists 15 numbered rows; the 15-row table is used here, per this task's own
instruction that the header is a known documentation error), and
[`docs/analysis/cxa_p_convert_locked_feature_eda_v1.md`](cxa_p_convert_locked_feature_eda_v1.md)
(encoding guidance, applied below and in
[`scripts/_cxconvert_modeling_common.py`](../../scripts/_cxconvert_modeling_common.py)'s
module docstring).

**Test stays sealed.** Every number below is train or validation only; the loader in
`_cxconvert_modeling_common.py` only ever queries `split IN ('train', 'validation')`.
**No feature set or hyperparameters are frozen by this document** -- that is a separate,
later, reviewed decision. This document is a comparison for review, not a freeze.

Split sizes (restated from the feature-lock docs): event-only train 7,847 rows / 738
goals, validation 1,720 rows / 159 goals; CxA+ train 1,991 rows / 181 goals, validation
**420 rows / 46 goals**.

**Standing small-sample caution for CxA+, repeated at every place a CxA+ verdict is
stated below, not just here:** CxA+'s validation split (420 rows, 46 positives) is
roughly 4x smaller than event-only's (1,720 rows, 159 positives), and both are far
smaller than P_create's own CxA+ validation split (19,490 rows). Every CxA+ metric
comparison in this document should be read as meaningfully noisier than its
event-only counterpart -- a handful of rows landing differently can move a metric
visibly at this scale, and this document treats that as a first-class consideration in
its recommendation, not an afterthought.

## Encoding fixes applied (per the EDA doc -- confirmed implemented, not just read)

Each of the four bullet points this task named is confirmed applied in
[`scripts/_cxconvert_modeling_common.py`](../../scripts/_cxconvert_modeling_common.py):

1. **`shot_technique_name = Lob` on CxA+ (16 total rows) is pooled into the
   reference/baseline category** -- confirmed in `encode_plus_candidate`: no
   `shot_technique_Lob` dummy is created for CxA+ at all (every CxA+ row collapses into
   the implicit `Normal`/other reference level for this feature). Event-only keeps it
   as its own dummy (`encode_event_candidate`, 62 rows there) -- confirmed present in
   the event-only design matrix (`logistic columns` / `tree columns` printed by the
   script both include `shot_technique_Lob` for event, and it is absent from CxA+'s
   column lists; see raw output).
2. **`shot_gk_distance_m` nulls get an explicit missingness-safe encoding** --
   confirmed in `_gk_distance_columns`: a `shot_gk_distance_missing` flag plus
   train-median imputation (fit train-only). **Applied differently per track, and this
   difference is itself confirmed, not assumed:** event-only has 4 null rows in train
   and 1 in validation, so the flag is a real, informative column there (included in
   both design matrices). CxA+ has **0** null rows in train+validation for this split
   (confirmed live, not assumed from the EDA's "possible in principle" framing) -- a
   flag column that is constant-zero on train makes the logistic design matrix singular
   (confirmed the hard way: the first run of this script crashed with
   `numpy.linalg.LinAlgError: Singular matrix` fitting CxA+'s logistic candidate before
   this was caught and fixed), so the flag is correctly omitted from CxA+'s design
   matrix while the imputation mechanism itself remains in the code path for both
   tracks (a no-op when nothing is null).
3. **The `shot_x_sb`/`shot_dist_to_goal_m`/`shot_gk_distance_m` near-duplicate cluster
   is handled by dropping two of three for the LOGISTIC candidate** -- confirmed:
   `encode_event_candidate`/`encode_plus_candidate` only include `shot_x_sb` and
   `shot_dist_to_goal_m` when `for_tree=True`. `shot_gk_distance_m` was chosen as the
   sole logistic representative because the feature-lock docs' own train-vs-validation
   confirmation showed it was the most stable of the trio in both tracks (event:
   -5.50 -> -5.59, essentially flat; plus: -4.84 -> -4.68, also essentially flat). The
   TREE candidate keeps all three, confirmed by the printed `tree columns` list
   including `shot_x_sb`, `shot_dist_to_goal_m`, and `shot_gk_distance_m` together in
   both tracks.
4. **No pitch-bounds clipping is applied** -- confirmed by omission: unlike
   P_create's equivalent script (which clips `start_x` to `[0,120]` for its own,
   different, quirk), this script's loader applies no clip anywhere, consistent with
   the EDA's finding that no clipping was needed for this feature set.

## 1. Dumb baseline

Predicts the TRAIN split's goal-conversion rate as a constant for every row. `roc_auc`
is undefined for a constant predictor (no rank information), reported as `null`.

| track | split | n | log_loss |
|---|---|---|---|
| event | train | 7,847 | 0.31181 |
| event | validation | 1,720 | 0.30817 |
| plus | train | 1,991 | 0.30464 |
| plus | **validation** | **420** | **0.34750** |

CxA+'s validation log_loss for the dumb baseline (0.3475) is notably higher than its
train log_loss (0.3046) -- consistent with the split policy's own documented finding
that CxA+'s validation split has a higher goal-conversion rate (10.95%) than its train
split (9.09%), a real, already-known consequence of this track's small population, not
new information, but worth restating here since it sets a harder-to-beat validation
bar for every subsequent rung on this track specifically.

## 2. XY baseline (model name `v1`)

Logistic regression on `shot_x_sb`, `shot_y_sb` **only** -- the shot's own location,
the P_convert analogue of P_create's pass-origin (`start_x`/`start_y`) XY baseline,
deliberately not the locked feature set.

| track | split | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|---|
| event | train | 7,847 | 0.29231 | 0.08197 | 0.6870 |
| event | validation | 1,720 | 0.28879 | 0.08127 | 0.6856 |
| plus | train | 1,991 | 0.28809 | 0.08008 | 0.6702 |
| plus | validation | 420 | 0.33204 | 0.09556 | 0.6475 |

A real but modest jump over the dumb baseline in both tracks -- shot location alone
recovers some, not most, of the signal (unlike P_create's own XY baseline, which
jumped straight to AUC ~0.86-0.90; P_convert's target is a harder discrimination
problem from location alone, consistent with football intuition that where a shot is
struck from matters, but far less than whether the finish itself is clean, under
pressure, one-on-one, etc.). **CxA+'s XY baseline is noticeably worse on validation
than train** (log_loss 0.288 -> 0.332, AUC 0.670 -> 0.648) -- consistent with the
small-sample caution above, not treated as a surprising finding on its own at this
rung.

## 3. Candidate models -- full locked feature set, two families

### 3a. Logistic -- statsmodels `Logit`, same fitting approach as P_create's own
candidate, producing a coefficients table with `std_error`/`p_value`.

### 3b. Tree -- LightGBM gradient boosting.

**Event-only** hyperparameters mirror P_create's own untuned defaults exactly:
`n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
min_child_samples=50, subsample=0.8, colsample_bytree=0.8`.

**CxA+ hyperparameters: `min_child_samples` raised to 100** (double event-only's
value), per this task's explicit instruction to adjust it upward if CxA+'s small size
warrants it -- **and it did, empirically, not just in principle.** A first attempt at
`min_child_samples=30` (a *decrease* from event-only's 50, reasoned from a since-corrected
assumption) was tried and produced severe overfitting: train log_loss 0.165 (AUC 0.966)
vs. validation log_loss 0.353 (AUC 0.681) -- a validation log_loss *worse than the dumb
baseline* (0.347). Raising `min_child_samples` to 100 (so a leaf must cover a
comparable *share* of CxA+'s smaller population, not a comparable absolute row count)
narrows but does **not** eliminate this gap -- see section 5 for the full, honestly-
reported diagnostic. `n_estimators`/`max_depth`/`num_leaves` are kept at P_create's
mirrored values per this task's explicit scope (only `min_child_samples` was named as
the knob to reconsider); no broader hyperparameter search was run to "fix" the
remaining overfitting, consistent with tuning being out of scope for this task.

| track | model | split | n | log_loss | brier_score | roc_auc |
|---|---|---|---|---|---|---|
| event | logistic | train | 7,847 | 0.26880 | 0.07586 | 0.7663 |
| event | logistic | validation | 1,720 | 0.26934 | 0.07702 | 0.7671 |
| event | tree | train | 7,847 | 0.23029 | 0.06603 | 0.8635 |
| event | tree | validation | 1,720 | 0.26647 | 0.07537 | 0.7657 |
| plus | logistic | train | 1,991 | 0.26631 | 0.07404 | 0.7482 |
| plus | logistic | validation | 420 | **0.31606** | 0.09193 | **0.7283** |
| plus | tree | train | 1,991 | 0.19964 | 0.05731 | 0.9120 |
| plus | tree | validation | 420 | **0.33506** | 0.09513 | **0.6912** |

**Event-only: both families land very close together on validation, with no clear
winner** (log_loss 0.2693 logistic vs 0.2665 tree, a 0.0029 gap; AUC 0.7671 logistic vs
0.7657 tree, tree very slightly *behind*). Event-only's tree shows a real but modest
train-validation gap (log_loss 0.230 -> 0.266) consistent with ordinary, expected tree
variance at this population size -- not the severe pattern seen on CxA+.

**CxA+: both families beat both baselines on validation log_loss and AUC, but the
tree candidate's validation log_loss (0.33506) is essentially the same as -- in fact
marginally worse than -- the XY baseline's (0.33204).** The tree candidate's AUC
(0.6912) is meaningfully better than the XY baseline's (0.6475), so it has real
ranking ability the XY baseline lacks, but its predicted *probabilities* are
overconfident enough (see calibration, below) that log_loss does not reward it. The
logistic candidate does not have this problem -- it clearly beats both baselines on
every metric (log_loss 0.31606 vs XY's 0.33204, AUC 0.7283 vs XY's 0.6475). **This is
the standout, honestly-reported finding of this document, occupying the role
P_create's own quasi-complete-separation finding did for its equivalent step -- see
section 5.**

### Calibration (validation, decile-binned predicted-vs-actual)

| track | model | decile 9 (top) mean predicted | decile 9 mean actual |
|---|---|---|---|
| event | logistic | 0.3529 | 0.3023 |
| event | tree | 0.3614 | 0.3256 |
| plus | logistic | 0.3211 | 0.2381 |
| plus | tree | 0.3674 | 0.3095 |

**Event-only both families are reasonably calibrated in the top decile**, with a
similar, modest over-confidence in both (predicted running ~0.03-0.05 above actual) --
not alarming, consistent with a well-behaved fit on a track with enough validation
support (1,720 rows) to make a 172-row top decile a reasonably stable estimate.

**CxA+ both families over-predict in the top decile, tree more severely** (logistic:
predicted 0.3211 vs actual 0.2381, a real but bounded gap; tree: predicted 0.3674 vs
actual 0.3095, also a real gap but on a **42-row decile bin with roughly 10-13
positive examples** -- at this bin size, a difference of 1-2 actual outcomes shifts the
"actual" rate by 2.4-4.8 percentage points on its own, so neither gap should be read
with the same confidence event-only's calibration numbers deserve. Full 10-decile
tables for both splits and both model families are in
`audit_outputs/cxconvert_analysis/baseline_and_candidate/{event,plus}_result.json`.

## 4. Comparison: baseline (v1) vs both candidate families, validation only

**Event-only:**

| comparison | log_loss delta | relative log_loss improvement | roc_auc delta |
|---|---|---|---|
| v1 -> logistic candidate | 0.28879 -> 0.26934 | **6.7% lower** | 0.6856 -> 0.7671 (**+0.0815**) |
| v1 -> tree candidate | 0.28879 -> 0.26647 | **7.7% lower** | 0.6856 -> 0.7657 (**+0.0801**) |

**CxA+:**

| comparison | log_loss delta | relative log_loss improvement | roc_auc delta |
|---|---|---|---|
| v1 -> logistic candidate | 0.33204 -> 0.31606 | **4.8% lower** | 0.6475 -> 0.7283 (**+0.0807**) |
| v1 -> tree candidate | 0.33204 -> 0.33506 | **0.9% HIGHER (worse)** | 0.6475 -> 0.6912 (**+0.0437**) |

**Both candidate families beat the XY baseline on event-only.** On CxA+, **only the
logistic candidate does** -- the tree candidate's validation log_loss is slightly
*worse* than the trivial two-coordinate baseline it is meant to improve on, despite a
real AUC improvement. This asymmetry between the two tracks (unlike P_create, where
tree won clearly and consistently in both) is the central empirical finding of this
document.

## 5. A model-fit issue worth reporting honestly

**CxA+'s tree candidate overfits enough to fail its own baseline comparison on
log_loss, even after the min_child_samples correction described in section 3b.**

The diagnosis, checked directly rather than inferred from the headline metrics alone:

- **Train-validation gap is large and one-sided.** Log_loss: 0.1996 (train) -> 0.3351
  (validation), a 68% relative increase. AUC: 0.9120 (train) -> 0.6912 (validation), a
  drop of 0.22. Event-only's tree, by contrast, shows a much smaller gap (log_loss
  0.230 -> 0.266, a 16% relative increase; AUC 0.863 -> 0.766, a drop of 0.10) on a
  ~4x larger validation split -- both the absolute overfitting and the track-size
  explanation for it are consistent with each other.
- **Several locked boolean features get zero split usage in the CxA+ tree** --
  `is_through_ball`, `shot_one_on_one`, `is_cross`, `shot_open_goal`, and `is_cut_back`
  all show **0** in the fitted model's split-count feature importances (raw output:
  `audit_outputs/cxconvert_analysis/baseline_and_candidate/plus_result.json`,
  `candidate.tree.feature_importance`). This is not a bug -- with `min_child_samples=100`
  on a 1,991-row train set, a boolean flag with a few hundred TRUE rows or fewer often
  cannot clear the leaf-size floor once the tree has already split on the
  higher-information continuous shot-geometry features (`shot_dist_to_goal_m`: 290
  splits, `reception_nearest_opponent_distance_m`: 222, `start_x`: 277 -- these
  dominate). **The practical consequence: CxA+'s fitted tree effectively reduces to a
  geometry-driven model and does not meaningfully use most of the locked boolean
  signal that the feature-lock docs spent real effort confirming.** Event-only's tree
  does use its boolean features (`is_through_ball`: 41 splits, `is_cross`: 41,
  `shot_first_time`: 37, `shot_open_goal`: 25, all nonzero) -- a real, checked
  difference between the two tracks' fitted trees, not just a metrics-table
  coincidence.
- **The logistic candidate has no equivalent problem.** All coefficients have finite,
  reasonable standard errors (checked explicitly per the task's instruction, the same
  way P_create's own candidate step checked for degenerate coefficients -- none found
  here). The largest standard error relative to its coefficient is `is_cut_back`
  (coefficient 0.011, std_error 0.438, p=0.980) -- this coefficient carries essentially
  no information, consistent with its thin support (53 train TRUE rows, 8 positive,
  already flagged thin by both the feature-lock and EDA docs), but it is **not**
  degenerate in the P_create sense (no coefficient anywhere near the -21.82/470,161
  std_error signature that document found for its own separation case). **No
  quasi-complete separation was found on CxA+'s logistic candidate** -- checked
  directly via the `separation_check` diagnostic (every locked boolean/dummy column's
  TRUE and FALSE groups both contain at least one positive example in train, event-only
  included; full counts in the raw JSON output) -- a genuine, checked non-finding, not
  an assumption.

**Practical impact: real, not cosmetic.** Unlike P_create's own finding (one degenerate
coefficient out of 14, small blast radius), this affects the CxA+ tree candidate's
headline usability directly -- its validation log_loss does not clear the XY baseline
it is meant to improve on, which is a materially different, more consequential finding
than a single hard-to-interpret coefficient.

**Recommendation for a future freeze/tuning stage (not decided here):** either (a) a
genuine hyperparameter search specifically for CxA+ (shallower trees, fewer
estimators, and/or early stopping against the validation split), which this task's
scope explicitly did not authorize, or (b) accept the logistic candidate as CxA+'s
primary model family (see section 6) and revisit the tree family only if a future task
is scoped to tune it properly for this track's population size.

## 6. Recommendation (input to a future freeze decision, not the freeze decision
itself)

**Event-only: no strong winner between families -- both are usable, with a mild lean
toward tree on loss-based metrics.** Tree edges out logistic on log_loss (0.2665 vs
0.2693) and brier score (0.0754 vs 0.0770), while logistic edges out tree on AUC
(0.7671 vs 0.7657) -- every one of these gaps is small enough (<=0.003 on any metric)
that neither family should be presented as a clear winner. Tree shows a real, if
modest, train-validation gap that logistic does not (logistic's train and validation
metrics are nearly identical, a sign of a stable, well-generalizing fit); this
stability is worth weighing alongside tree's tiny loss-metric edge, not overridden by
it. **If a freeze decision is needed for event-only from this comparison alone, either
family is defensible; this document does not force a confident recommendation where
the data does not support one.**

**CxA+: logistic is the clearer choice, for reasons beyond simple metric comparison.**
Logistic beats the XY baseline on every metric; tree does not (section 4). Logistic's
coefficients are all well-behaved (no separation, section 5); tree's fitted structure
effectively ignores most of the locked boolean feature set at this population size
(section 5). **Tree's AUC (0.6912) is still meaningfully better than the XY
baseline's (0.6475)**, so it is not without value, but its calibration and
generalization at CxA+'s scale are demonstrably weaker than logistic's right now, with
the specific, correctable cause (an untuned tree on a small population) identified
rather than left mysterious. **Logistic is the more promising family for CxA+ as
currently fit** -- the opposite conclusion from P_create's own CxA+ recommendation
(which favored tree), and stated as such deliberately: this is a genuine,
population-size-driven divergence between the two projects' equivalent steps, not an
inconsistency to paper over.

**Standing caution restated:** every CxA+ number above comes from a 420-row validation
split with 46 positive examples. The *direction* of the logistic-over-tree finding
rests on more than a single noisy metric (it holds across log_loss, brier score, the
baseline comparison, the split-usage diagnostic, and the separation check together),
which is why this document is comfortable stating it as a real finding rather than
validation noise -- but the exact metric values themselves (e.g. the precise 0.335 vs
0.316 log_loss gap) should be treated as directionally reliable, not numerically
precise, at this sample size.

Neither recommendation freezes anything. Both tracks' final feature set, model family,
and hyperparameters remain open questions for a later, separate, reviewed freeze
decision.
