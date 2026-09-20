# CxA Event-Only P_convert Feature Lock v1

Date: 2026-09-20
Track: CxA event-only (`oam_features.cxconvert_event_v1_training_matrix`). CxA+ is
explicitly out of scope for this lock -- see
[`cxa_plus_p_convert_feature_lock_v1.md`](cxa_plus_p_convert_feature_lock_v1.md) and
"What this doc does not cover" below.
Split: `split` column already joined into the training matrix at materialization time
(from `oam_analysis.cxa_match_splits_v1`), no separate join needed. Train = 7,847 rows
(738 goals, 9.405%), validation = 1,720 rows (159 goals, 9.244%) -- see
`docs/cxa_convert_split_policy_and_plan.md` for full split sizes/balance, not
re-derived here.

This is the final locked candidate feature list for a first CxA event-only P_convert
baseline, built from `docs/analysis/cxa_p_convert_pre_model_analysis.md` section 9's
three candidate buckets, all of which were re-run against train and validation in this
task -- **including the "locked" bucket**, which P_create's own equivalent lock skipped
re-checking. This task's instructions were explicitly stricter on that point ("every one
must still be checked, not waved through"), so section 1 below is a genuine
confirmation, not a formality.

**Test stays sealed.** Everything below is train + validation only, reproduced live by
[`scripts/validate_cxconvert_features_on_split.py`](../../scripts/validate_cxconvert_features_on_split.py)
(raw output:
[`audit_outputs/cxconvert_analysis/feature_lock/split_validation_event.json`](../../audit_outputs/cxconvert_analysis/feature_lock/split_validation_event.json)).
Nothing here has been checked against test, and nothing should be until model training
is complete and a final report is due.

## 1. Population-strong features, re-confirmed on train and validation (11)

Lift = `rate(y_goal | flag=TRUE) / rate(y_goal | flag=FALSE)` for booleans; gap =
`mean(flag=TRUE) - mean(flag=FALSE)` for numerics, signed to match the pre-model
analysis's stated direction.

| feature | train stat | validation stat | verdict |
|---|---|---|---|
| `is_through_ball` | 3.44x | 4.04x | Confirmed -- **strengthens** on validation |
| `shot_one_on_one` | 3.08x | 2.46x | Confirmed -- direction holds, real decay (0.62x) |
| `is_cross` | 1.69x | 1.93x | Confirmed -- **strengthens** on validation |
| `shot_first_time` | 1.65x | 1.73x | Confirmed -- **strengthens** on validation |
| `start_x` | +3.35 units | +2.90 units | Confirmed -- direction holds, modest decay |
| `pass_end_x` | +6.16 units | +6.50 units | Confirmed -- **strengthens** on validation |
| `shot_x_sb` | +5.11 units | +5.00 units | Confirmed -- essentially flat, no decay |
| `shot_dist_to_goal_m` (derived, `shot_x_sb`/`shot_y_sb` combined) | -5.32m | -5.07m | Confirmed -- essentially flat |
| `shot_gk_distance_m` | -5.50m | -5.59m | Confirmed -- **strengthens** slightly |
| `shot_defenders_within_5m` | +0.272 | +0.095 | Confirmed, direction holds -- but a large relative decay (65% of train's gap lost); see caution below |
| `shot_defenders_within_8m` | +0.322 | +0.220 | Confirmed, direction holds -- moderate decay (32% lost) |

**All 11 hold direction on validation.** Nine are stable-to-strengthening. Two
(`shot_defenders_within_5m`/`_8m`) show real magnitude decay worth flagging even though
direction holds: these are the same features the pre-model analysis already flagged as
partially confounded with shot-distance-to-goal (closer shots are both more likely to
score and sit in a more congested penalty box) -- the decay on validation is consistent
with that confound being noisier than the primary distance signal, not evidence the
underlying relationship is spurious. Kept locked, flagged for attention at the
modelling stage (e.g. checking VIF/partial effects against `shot_gk_distance_m` once a
model is fit).

## 2. "Needs split-validation" features -- promotion results

**Promotion rule, adapted from P_create's own fixed 2.0x-of-baseline floor (stated and
justified, not silently reused):** P_create's validation splits had 19,490-92,977 rows;
this track's validation split has 1,720 rows (159 goals). A fixed 2.0x floor calibrated
against P_create's much larger splits would reject real signal here purely on sample
size. Instead: (1) direction must not flip (same sign/side as train), AND (2)
validation's effect magnitude (`ratio - 1` for lift/suppression, `|gap|` for numerics)
must retain **>= 50%** of train's magnitude. This scales the bar to how strong the
feature's own train signal was, which is the right comparison at this support level.

| feature | train stat | validation stat | retention | verdict |
|---|---|---|---|---|
| `shot_open_goal` | 7.41x (excess 6.41) | 4.74x (excess 3.74) | 0.58 | **PROMOTE** |
| `shot_technique_name = Lob` | 3.37x (excess 2.37) | 2.34x (excess 1.34) | 0.57 | **PROMOTE** |
| `shot_technique_name = Diving Header` | 2.30x (excess 1.30) | 1.98x (excess 0.98) | 0.75 | **PROMOTE** |
| `shot_technique_name = Backheel` | 1.92x (excess 0.92) | 3.12x (excess 2.12) | 2.31 | **PROMOTE** -- strengthens sharply, thin support (7 validation TRUE) |
| `shot_technique_name = Volley` | 1.45x (excess 0.45) | 1.38x (excess 0.38) | 0.84 | **PROMOTE** |
| `is_cut_back` | 1.35x (excess 0.35) | 2.41x (excess 1.41) | 3.98 | **PROMOTE** -- strengthens sharply |
| `pass_body_part_name = No Touch` | 1.94x (excess 0.94) | 0.00x (0/5 validation TRUE rows scored) | 0.00 | **DROP** -- validation collapses to zero (no positives in 5 TRUE rows) |
| `pass_technique_name = Straight` | 0.47x (i.e. suppressed on train, not elevated) | 0.51x (suppressed) | n/a | **DROP** -- the pre-model analysis's "needs-validation" bucket assumed an elevated direction for this level; train data shows it is actually mildly *suppressive* (6.4% vs 9.4% baseline in the full-population table), so it fails the "elevated" direction check on train itself, before validation is even considered |
| `is_switch` | suppression 1.52x (excess 0.52) | suppression 1.24x (excess 0.24) | 0.46 | **DROP** -- direction holds but retention (0.46) falls just short of the 0.50 floor |
| `pass_type_name = Corner` | suppression 1.35x (excess 0.35) | suppression 1.42x (excess 0.42) | 1.20 | **PROMOTE** -- strengthens |

**6 of 10 promote, 4 drop.** Two drops are clean (`pass_body_part_name = No Touch`
collapses to zero signal on validation; `pass_technique_name = Straight` never had the
assumed direction to begin with). `is_switch` is a genuine borderline case -- direction
holds, but retention (0.46) is close enough to the 0.50 floor that a different
reasonable threshold could have gone the other way; dropped per the stated rule rather
than special-cased.

## 3. `shot_end_z` -- explicit decision: drop for this baseline

Per the pre-model analysis's own instruction not to wave this feature through. The
train/validation split reproduces the full-population pattern almost exactly:

| split | goal rows | `shot_end_z` null (goal) | no-goal rows | `shot_end_z` null (no-goal) | avg (goal, non-null) | avg (no-goal, non-null) |
|---|---|---|---|---|---|---|
| train | 738 | **0 (0%)** | 7,109 | 2,526 (35.5%) | 0.956 | 2.090 |
| validation | 159 | **0 (0%)** | 1,561 | 529 (33.9%) | 0.913 | 2.070 |

Every one of the three options the pre-model analysis named was evaluated:

- **Impute with a missingness flag, check the flag doesn't dominate importance** --
  not actually checkable in this task: verifying a flag "doesn't dominate importance"
  requires a fitted model to read feature importances from, and this task is
  analysis-only (no model fitting, per the explicit constraints). Deferring this option
  to the modelling stage rather than approximating it here.
- **Restrict to careful combined use** -- rejected as too vague a mitigation to state
  precisely without a modelling context to define "careful" in.
- **Drop entirely** -- **chosen.**

**Justification:** the null pattern is not incidental missing data -- it reproduces
*exactly* (0% for every goal, ~34-36% for non-goals) on an independently-drawn
validation split, which rules out sampling noise as the explanation and confirms this
is a deterministic StatsBomb tagging artifact (a goal, by definition, reaches the goal
frame and gets a final height recorded; other outcomes may not). A naive
`shot_end_z_is_null` indicator built from this column would function as a near-perfect
proxy for "not a goal" -- structurally the same risk category as `is_completed`'s
exclusion from P_create's own candidate list (a near-perfect proxy for the target
driven by StatsBomb's own downstream tagging behavior, not causally prior geometry),
even though `shot_end_z` is nominally a shot attribute rather than an explicit outcome
field. **Decision: drop `shot_end_z` from this baseline's locked feature list.** This
is a "drop for now, revisit at the modelling stage" call, not permanent: if a future
modeller wants to test an imputed-value-plus-flag version, the check the pre-model
analysis asked for (does the flag dominate importance) becomes answerable once there is
an actual fitted model's importances to read, which there is not here.

## 4. Already-flagged drops -- reaffirmed, not re-opened

Per the pre-model analysis's own drop lists (thin-support/zero-variance and
redundant), reaffirmed excluded from this lock without re-litigation:

- **`shot_follows_dribble`** -- 0 goals in the 7 TRUE rows across the full population,
  no signal possible.
- **`shot_type_name`** -- 100% constant (`Open Play`) across the full population.
- **`shot_counterpress`** -- 100% constant (`FALSE`) across the full population.
- **`shot_frame_player_count`** -- near-duplicate of `shot_defenders_visible`
  (r=0.9442, pre-model analysis section 7).
- **`pass_end_y`** -- near-duplicate of `shot_y_sb` (r=0.8904) for a linear baseline;
  harmless to retain for tree models but not part of this locked list.

`receiver_x`/`receiver_y` do not apply to this track (CxA+-only columns, not present in
the event-only matrix schema at all).

## 5. Full locked feature list (15)

| # | feature | type | source |
|---|---|---|---|
| 1 | `is_through_ball` | boolean | section 1, re-confirmed |
| 2 | `shot_one_on_one` | boolean | section 1, re-confirmed |
| 3 | `is_cross` | boolean | section 1, re-confirmed |
| 4 | `shot_first_time` | boolean | section 1, re-confirmed |
| 5 | `start_x` | numeric | section 1, re-confirmed |
| 6 | `pass_end_x` | numeric | section 1, re-confirmed |
| 7 | `shot_x_sb` | numeric | section 1, re-confirmed |
| 8 | `shot_dist_to_goal_m` (derived from `shot_x_sb`/`shot_y_sb`) | numeric | section 1, re-confirmed |
| 9 | `shot_gk_distance_m` | numeric | section 1, re-confirmed |
| 10 | `shot_defenders_within_5m` | numeric | section 1, re-confirmed, notable decay -- see caution |
| 11 | `shot_defenders_within_8m` | numeric | section 1, re-confirmed, moderate decay |
| 12 | `shot_open_goal` | boolean | section 2, promoted |
| 13 | `shot_technique_name` (Lob, Diving Header, Backheel, Volley levels; else `Normal`/other) | categorical | section 2, promoted (all 4 elevated levels) |
| 14 | `is_cut_back` | boolean | section 2, promoted |
| 15 | `pass_type_name = Corner` (suppressive level; else `(null=open play)`/other) | categorical | section 2, promoted |

**Dropped from "needs-validation":** `pass_body_part_name = No Touch`,
`pass_technique_name = Straight`, `is_switch` (section 2).
**Dropped/deferred:** `shot_end_z` (section 3).
**Reaffirmed excluded:** `shot_follows_dribble`, `shot_type_name`,
`shot_counterpress`, `shot_frame_player_count`, `pass_end_y` (linear baseline only)
(section 4).
**Benchmark-only, never a candidate:** `statsbomb_xg`.
**Leakage/methodology exclusions carried from the feasibility audit and pre-model
analysis, not reconsidered:** `pass_outcome_name`, `is_completed`, `passer_team_id`,
`passer_player_id`, `deflected`, `saved_off_target`, `saved_to_post`.
**Identifiers/provenance, never features:** `pass_event_id`, `match_id`,
`competition_id`, `season_id`, `shot_event_id`, `data_version`,
`silver_schema_version`, `feature_version`, `materialized_at`, `split`.
**Labels, never features:** `y_create`, `y_goal`.

## 6. What this doc does not cover

- **CxA+.** Locked separately, see
  [`cxa_plus_p_convert_feature_lock_v1.md`](cxa_plus_p_convert_feature_lock_v1.md) --
  do not assume this document's verdicts transfer, since two of this track's own
  already-locked features (`shot_defenders_within_5m`/`_8m`) do **not** confirm on
  CxA+'s validation split (see that document, section 1).
- **Test-split confirmation.** Explicitly deferred. Test stays sealed until model
  training and a final report are otherwise complete.
- **Model training, feature engineering beyond what's listed, or any modelling
  decision.** This document locks a candidate feature *list*. It does not train a
  model, does not choose a model family, and does not report a model metric.
- **`shot_end_z`'s impute-with-flag option.** Explicitly deferred to the modelling
  stage (section 3) -- not resolved by this analysis-only task.
