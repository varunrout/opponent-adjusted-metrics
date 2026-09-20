# CxA+ P_convert Feature Lock v1

Date: 2026-09-20
Track: CxA+ (360-at-reception) only (`oam_features.cxconvert_plus_v1_training_matrix`).
Structured to mirror
[`cxa_event_p_convert_feature_lock_v1.md`](cxa_event_p_convert_feature_lock_v1.md) (the
event-only track's lock), but this document is self-contained.

**Small-validation-split caveat, stated once here and repeated at every call that rests
on it below:** CxA+'s validation split is **420 rows / 46 goals** (per
`docs/cxa_convert_split_policy_and_plan.md`) -- an order of magnitude smaller than
P_create's own CxA+ validation split (19,490 rows). Several "needs-validation" flags in
this track have single-digit validation TRUE-row counts; a mechanical promotion-rule
PASS on numbers that thin is not the same strength of evidence as a PASS on hundreds of
rows, and every such case is flagged explicitly, not silently folded into an ordinary
"Confirmed" row the way a comfortably-supported feature would be.

**Tournament-only population, restated per project convention:** CxA+'s population is a
strict subset of P_create's own CxA+ population, which has zero Premier League rows.
Any CxA+ P_convert model's generalization claim is scoped to international tournament
football only (see the feasibility audit and pre-model analysis for the full argument;
not re-derived here).

Split: `split` column already joined into the training matrix. Train = 1,991 rows (181
goals, 9.091%), validation = 420 rows (46 goals, 10.952%).

## 1. Population-strong features, re-confirmed on train and validation (13, 2 with a
caveat)

Reproduced live from
[`scripts/validate_cxconvert_features_on_split.py`](../../scripts/validate_cxconvert_features_on_split.py)
(raw output:
[`audit_outputs/cxconvert_analysis/feature_lock/split_validation_plus.json`](../../audit_outputs/cxconvert_analysis/feature_lock/split_validation_plus.json)),
same statistics and promotion-rule shape as the event-only lock.

| feature | train stat | validation stat | verdict |
|---|---|---|---|
| `is_through_ball` | 3.43x | 2.48x | Confirmed -- direction holds, real decay |
| `shot_one_on_one` | 2.26x | 2.95x | Confirmed -- **strengthens** |
| `is_cross` | 1.94x | 2.57x | Confirmed -- **strengthens** |
| `shot_first_time` | 1.79x | 1.06x | **Confirmed, but only barely** -- ratio stays just above 1.0; the lift almost entirely vanishes on this split. Kept locked (direction technically holds) but treated as the weakest confirmation in this table, not an ordinary pass. |
| `start_x` | +2.64 units | +3.94 units | Confirmed -- **strengthens** |
| `pass_end_x` | +5.60 units | +6.67 units | Confirmed -- **strengthens** |
| `shot_x_sb` | +4.71 units | +4.30 units | Confirmed -- modest decay |
| `shot_dist_to_goal_m` (derived) | -4.93m | -4.46m | Confirmed -- modest decay |
| `shot_gk_distance_m` | -4.84m | -4.68m | Confirmed -- essentially flat |
| `shot_defenders_within_5m` | +0.294 | **-0.037** | **NOT confirmed -- direction fails on validation.** See caution below, not locked at the same tier. |
| `shot_defenders_within_8m` | +0.370 | **-0.038** | **NOT confirmed -- direction fails on validation.** Same caution. |
| `reception_nearest_opponent_distance_m` | -0.840m | -0.976m | Confirmed -- **strengthens** |
| `reception_opponents_within_5m` | +0.456 | +0.192 | Confirmed -- direction holds, real decay (58% of train's gap lost) |

**11 of 13 confirm cleanly (one, `shot_first_time`, only barely). Two
(`shot_defenders_within_5m`/`_8m`) do not confirm on this track and are held out of the
CxA+ locked set**, even though they are locked for the event-only track (see that
document, section 1). This is exactly the outcome the task's explicit instruction to
recheck every "already-locked" feature was designed to catch -- had this task skipped
re-checking the population-strong bucket (as P_create's own event-only lock did), this
divergence between tracks would have gone unnoticed.

**Why the flip is plausible, not just noise:** the pre-model analysis already flagged
`shot_defenders_within_5m`/`_8m` as confounded with shot-distance-to-goal (closer shots
are both more congested and more likely to score). On CxA+'s validation split (46 goal
rows, 374 non-goal rows), the residual, deconfounded relationship these features
capture is apparently too weak to survive the added noise of a much smaller sample --
the gap collapses from a real positive value to a value indistinguishable from zero
(-0.037/-0.038), not a large negative reversal. **Recommendation: hold both features
out of the CxA+ locked list** for this baseline; they remain locked for event-only,
where the same validation split is ~4x larger and the direction held with real margin.

## 2. "Needs split-validation" features -- promotion results

Same rule as the event-only lock (direction must not flip; validation retains >= 50%
of train's excess-over-baseline magnitude), restated here since this track's numbers
are thinner and the rule's behavior at low counts matters more:

| feature | train stat | validation stat | retention | verdict |
|---|---|---|---|---|
| `shot_open_goal` | 8.25x (excess 7.25) | 4.73x (excess 3.73) | 0.51 | **PROMOTE** -- barely clears the floor; validation rests on only **4 TRUE rows (2 positives)**, the thinnest support of any promoted feature in this document |
| `shot_technique_name = Lob` | 2.01x (excess 1.01) | 1.84x (excess 0.84) | 0.84 | **PROMOTE** -- validation rests on 5 TRUE rows (1 positive) |
| `shot_technique_name = Diving Header` | 2.08x (excess 1.08) | 0.00x (2 TRUE rows, 0 positives) | 0.00 | **DROP** |
| `shot_technique_name = Backheel` | 1.84x (excess 0.84) | 0.00x (3 TRUE rows, 0 positives) | 0.00 | **DROP** |
| `shot_technique_name = Volley` | 1.91x (excess 0.91) | 0.70x (i.e. suppressed, 13 TRUE rows, 1 positive) | -0.33 | **DROP** -- direction flips |
| `is_cut_back` | 1.69x (excess 0.69) | 1.86x (excess 0.86) | 1.25 | **PROMOTE** -- strengthens; validation rests on 10 TRUE rows (2 positives) |
| `pass_body_part_name = No Touch` | 1.57x (excess 0.57) | 0.00x (1 TRUE row, 0 positives) | 0.00 | **DROP** -- essentially no validation data (a single row) |
| `pass_technique_name = Straight` | 0.00x (0 goals in the only 7 TRAIN TRUE rows) | 0.00x (1 TRUE row, 0 positives) | n/a | **DROP** -- no signal on train to begin with for this track; CxA+ has far fewer `Straight`-tagged passes than the event-only track (7 vs 68 train rows), too thin to evaluate at all |
| `is_switch` | suppression 5.33x (excess 4.33) | suppression 0.95x (i.e. essentially no effect, direction fails) | -0.01 | **DROP** -- a striking reversal: the huge train "effect" (excess 4.33) was built on only **112 TRUE train rows with just 2 positives** -- an extreme case of a thin-denominator ratio that does not survive an independently-drawn split at all. A clear illustration of why the promotion rule requires validation confirmation rather than trusting a large train-only ratio on its own. |
| `pass_type_name = Corner` | suppression 2.35x (excess 1.35) | suppression 2.44x (excess 1.44) | 1.07 | **PROMOTE** -- strengthens; validation rests on 42 TRUE rows (2 positives) |

**4 of 10 promote, 6 drop.** Every promoted feature in this section rests on single- or
low-double-digit validation TRUE-row counts (4 to 42) with 1-2 positive examples each
-- **the small-validation-split caveat applies to all four promotions in this section**,
more strongly than to any feature in the event-only lock. `is_switch`'s collapse is the
clearest cautionary example in either track: a train-only ratio of 5.33x, built on just
2 positive rows, evaporates to statistical noise on validation. This is direct evidence
the promotion rule's validation-confirmation step is doing real work, not a formality,
on this track specifically.

## 3. `shot_end_z` -- explicit decision: drop for this baseline (same as event-only)

| split | goal rows | `shot_end_z` null (goal) | no-goal rows | `shot_end_z` null (no-goal) | avg (goal, non-null) | avg (no-goal, non-null) |
|---|---|---|---|---|---|---|
| train | 181 | **0 (0%)** | 1,810 | 631 (34.9%) | 0.973 | 2.156 |
| validation | 46 | **0 (0%)** | 374 | 122 (32.6%) | 0.900 | 2.110 |

**Identical pattern to the event-only track**, reproduced on this track's own,
independently-drawn train/validation split: every goal has a non-null `shot_end_z`,
roughly a third of non-goals do not, in both splits. Same reasoning, same decision:
**drop `shot_end_z` from this baseline's locked list.** Not re-litigated further here --
see the event-only lock's section 3 for the full justification (the impute-with-flag
option requires a fitted model's importances to evaluate "does the flag dominate,"
which is out of scope for this analysis-only task; deferred, not resolved).

## 4. Already-flagged drops -- reaffirmed, not re-opened

- **`shot_follows_dribble`** -- 0 goals in 3 TRUE rows across the full CxA+ population.
- **`shot_type_name`** -- 100% constant (`Open Play`).
- **`shot_counterpress`** -- 100% constant (`FALSE`).
- **`shot_frame_player_count`** -- near-duplicate of `shot_defenders_visible`
  (r=0.9442, pre-model analysis).
- **`pass_end_y`** -- near-duplicate of `shot_y_sb` (r=0.8904) for a linear baseline.
- **`receiver_x`, `receiver_y`** -- exact duplicate of `pass_end_x`/`pass_end_y`
  (r=1.0000, pre-model analysis section 7, reaffirming P_create's own identical
  finding for this pair).

## 5. Full locked feature list (17, 2 held out from the event-only track's own list)

| # | feature | type | source |
|---|---|---|---|
| 1 | `is_through_ball` | boolean | section 1, re-confirmed |
| 2 | `shot_one_on_one` | boolean | section 1, re-confirmed |
| 3 | `is_cross` | boolean | section 1, re-confirmed |
| 4 | `shot_first_time` | boolean | section 1, re-confirmed -- **weak confirmation, see caveat** |
| 5 | `start_x` | numeric | section 1, re-confirmed |
| 6 | `pass_end_x` | numeric | section 1, re-confirmed |
| 7 | `shot_x_sb` | numeric | section 1, re-confirmed |
| 8 | `shot_dist_to_goal_m` (derived) | numeric | section 1, re-confirmed |
| 9 | `shot_gk_distance_m` | numeric | section 1, re-confirmed |
| 10 | `reception_nearest_opponent_distance_m` | numeric (360) | section 1, re-confirmed |
| 11 | `reception_opponents_within_5m` | numeric (360) | section 1, re-confirmed |
| 12 | `shot_open_goal` | boolean | section 2, promoted -- **thin validation support (4 rows), see caveat** |
| 13 | `shot_technique_name = Lob` (else `Normal`/other) | categorical | section 2, promoted -- **thin validation support (5 rows)** |
| 14 | `is_cut_back` | boolean | section 2, promoted -- **thin validation support (10 rows)** |
| 15 | `pass_type_name = Corner` (suppressive level; else `(null=open play)`/other) | categorical | section 2, promoted -- **thin validation support (42 rows)** |

**Held out from this track despite being locked for event-only:** `shot_defenders_within_5m`,
`shot_defenders_within_8m` -- direction fails to confirm on CxA+'s validation split
(section 1).
**Dropped from "needs-validation":** `shot_technique_name` (Diving Header, Backheel,
Volley levels), `pass_body_part_name = No Touch`, `pass_technique_name = Straight`,
`is_switch` (section 2).
**Dropped/deferred:** `shot_end_z` (section 3).
**Reaffirmed excluded:** `shot_follows_dribble`, `shot_type_name`,
`shot_counterpress`, `shot_frame_player_count`, `pass_end_y` (linear baseline only),
`receiver_x`, `receiver_y` (section 4).
**Benchmark-only, never a candidate:** `statsbomb_xg`.
**Leakage/methodology exclusions carried from the feasibility audit and pre-model
analysis, not reconsidered:** `passer_team_id`, `passer_player_id`, `deflected`,
`saved_off_target`, `saved_to_post` (`pass_outcome_name`/`is_completed` do not exist in
this track's schema at all).
**Identifiers/provenance, never features:** `pass_event_id`, `match_id`,
`competition_id`, `season_id`, `receipt_event_id`, `shot_event_id`, `data_version`,
`silver_schema_version`, `feature_version`, `materialized_at`, `split`.
**Labels, never features:** `y_create`, `y_goal`.

## 6. What this doc does not cover

- **The event-only track.** Locked separately, see
  [`cxa_event_p_convert_feature_lock_v1.md`](cxa_event_p_convert_feature_lock_v1.md) --
  its own 15-feature list includes `shot_defenders_within_5m`/`_8m`, which this
  document explicitly does **not** carry over for CxA+ (section 1).
- **Test-split confirmation.** Explicitly deferred, sealed per the split policy.
  Given how thin several of this track's validation counts are (single digits for
  three of the four section-2 promotions), test-split re-confirmation matters more for
  this track than for event-only and should be prioritized when test opens.
- **Model training, feature engineering beyond what's listed, or any modelling
  decision.** This document locks a candidate feature *list* for CxA+. It does not
  train a model, does not choose a model family, and does not report a model metric.
- **`shot_end_z`'s impute-with-flag option.** Explicitly deferred to the modelling
  stage (section 3).
