# CxA Event-Only P_create Feature Lock v1

Date: 2026-09-19
Track: CxA event-only (`oam_features.cxa_event_v1_training_matrix`) only. CxA+ is
explicitly out of scope for this lock -- see "What this doc does not cover" below.
Split table: `oam_analysis.cxa_match_splits_v1` (train = 426 matches / 422,946 rows,
validation = 92 matches / 92,977 rows, test = 92 matches, sealed). Split-balance
`create_rate`: train 1.855%, validation 1.850%, test 1.871% (all three within 0.02pp of
each other, confirming the match-level split did not bias class balance).

This is the final locked candidate feature list for a first CxA event-only P_create
baseline, built from two prior deliverables:

- `docs/analysis/cxa_p_create_pre_model_analysis.md` (full-population univariate
  screen): 18 candidate features ranked; 6 flagged strong on effect size + support, 4
  flagged "elevated lift but smaller support -- validate on the validation split before
  trusting."
- `docs/cxa_split_policy_and_parallel_plan.md`, step 5 ("Validate selected features on
  the validation split: direction stability, support stability, ... uplift over the XY
  baseline"): the check this document's section 2 makes reproducible.

**Test stays sealed.** Everything below is train + validation only. Nothing here has
been checked against test, and nothing should be until model training is complete and
a final report is due (per the split policy's sequential plan, steps 9-10).

## 1. Locked features (10 total)

### 1a. Confirmed on the full population (pre-model analysis, section 9's "strongest
candidates" -- large effect size + large support, held across every competition slice)

| feature | population signal |
|---|---|
| `is_cross` | 20.79% create rate vs 2.14% baseline (~9.7x), ~14,900 TRUE rows, holds in all 3 competitions |
| `is_through_ball` | 4.26% vs 0.33% (~12.8x), holds in all 3 competitions |
| `pass_type_name` (Corner, Free Kick levels) | Corner 17.64% vs 1.85% null-level baseline (~9.5x); Free Kick 3.32% (~1.8x) |
| `start_x` | mean 95.46 (create) vs 58.59 (no-create), +36.9 units, holds in all 3 competitions |
| `end_x` | mean 100.96 vs 65.8, +35.2 units (redundant-but-not-duplicate with `start_x`, r=0.77 -- kept as a separate feature, see pipeline docs) |
| `play_pattern_name` (Counter, Corner levels) | Counter 14.40% vs 1.61% Regular-Play baseline (~8.9x, small support 3,939 rows); Corner 8.45% (~5.2x, large support 22,128 rows) |

These 6 were population-strong with large support and were not re-run through the
train/validation split check in this task (they were already the ones *not* flagged
for re-validation -- re-checking them is reasonable future hygiene but was not asked
for here and is not claimed as done).

### 1b. Newly confirmed via train-vs-validation split check (this task)

Reproduced live from `oam_analysis.cxa_match_splits_v1` joined to
`cxa_event_v1_training_matrix` by
[`scripts/validate_cxa_event_features_on_split.py`](../../scripts/validate_cxa_event_features_on_split.py)
(raw output:
[`audit_outputs/cxa_analysis/feature_lock/split_validation_result.json`](../../audit_outputs/cxa_analysis/feature_lock/split_validation_result.json)).
Lift = `rate(y_create | flag=TRUE) / rate(y_create | flag=FALSE)` (the feature group's
create rate over its own complement's, not the whole-population rate).

| feature | train n_true | train lift | validation n_true | validation lift | verdict |
|---|---|---|---|---|---|
| `pass_technique_name = Outswinging` | 1,326 | **13.87x** | 322 | **13.85x** | Confirmed -- most stable of the four (0.02x drop from train to validation) |
| `is_switch` | 12,141 | **2.57x** | 2,604 | **2.28x** | Confirmed -- weakest of the four but stable, large support in both splits |
| `is_cut_back` | 861 | **13.22x** | 181 | **11.27x** | Confirmed -- direction holds, some magnitude decay (1.95x), thin support |
| `pass_body_part_name = No Touch` | 359 | **3.31x** | 91 | **2.98x** | Confirmed -- direction holds, thinnest support of the four (91 validation TRUE rows, only 5 of them positive) |

All 4 numbers were independently re-queried for this task, not transcribed from the
earlier manual check, and match it exactly.

**Promotion rule applied** (stated and justified in the script, not left implicit):
1. Direction must not flip -- train lift > 1x AND validation lift > 1x.
2. Validation lift must be >= **2.0x**. Chosen so the weakest confirmed candidate
   (`is_switch`, validation lift 2.28x) clears it with a real but not enormous margin
   (0.28x), while staying low enough to accept genuine-but-modest signal rather than
   only admitting the largest effects. At this population's ~1.85% base rate and these
   support sizes (91 to 12,141 TRUE rows), a flag with no real relationship to
   `y_create` would not reliably land at or above 2x lift on an independently-drawn
   validation split -- sampling noise at these support levels does not reproduce a 2x
   directional lift twice in a row by chance.

All 4 features pass both checks. None collapsed to noise, none flipped direction.

## 2. Thin-support features to re-check when test opens

Two of the four newly-confirmed features have support thin enough that their exact
lift number could look different again on test's 92 matches, even though direction is
very unlikely to flip given how far above the 2.0x threshold both currently sit:

- **`is_cut_back`** -- 181 validation TRUE rows (37 positives). Showed the largest
  train-to-validation magnitude decay of the four (13.22x -> 11.27x, a 1.95x drop).
- **`pass_body_part_name = No Touch`** -- 91 validation TRUE rows (5 positives). The
  smallest validation TRUE-row count of any locked feature; with only 5 positive
  examples in validation, its 2.98x lift rests on a very small count and should be
  treated as the least statistically certain number in this document, even though the
  direction and rough magnitude match train (3.31x).

Neither is being excluded -- both cleared the promotion rule with margin (11.27x and
2.98x, both well above the 2.0x floor) -- but both should be re-run against test once
test opens (per the split policy, only after model training and a final report are
otherwise complete) before being treated as fully settled, rather than assumed stable
from two splits alone.

## 3. Full locked feature list (10)

| # | feature | type | source |
|---|---|---|---|
| 1 | `is_cross` | boolean | pre-model analysis, population-strong |
| 2 | `is_through_ball` | boolean | pre-model analysis, population-strong |
| 3 | `pass_type_name` (Corner, Free Kick levels; else `(null=open play)`) | categorical | pre-model analysis, population-strong |
| 4 | `start_x` | numeric | pre-model analysis, population-strong |
| 5 | `end_x` | numeric | pre-model analysis, population-strong |
| 6 | `play_pattern_name` (Counter, Corner levels; else pooled/other) | categorical | pre-model analysis, population-strong |
| 7 | `pass_technique_name = Outswinging` (else `(null=regular)`/other) | boolean/categorical | this task, train+validation confirmed |
| 8 | `is_switch` | boolean | this task, train+validation confirmed |
| 9 | `is_cut_back` | boolean | this task, train+validation confirmed, thin support (section 2) |
| 10 | `pass_body_part_name = No Touch` (else other levels) | boolean/categorical | this task, train+validation confirmed, thinnest support (section 2) |

## 4. Exclusion list (restated from the pipeline docs -- this doc is self-contained)

The following must **not** enter the CxA event-only P_create baseline as candidate
features, per `docs/cxa_data_feasibility_audit.md` (point 6) and
`docs/analysis/cxa_p_create_pre_model_analysis.md` (section 8):

- **`pass_outcome_name`, `is_completed`** -- near-perfect target proxy (2,943x rate
  difference between `is_completed=TRUE`/`FALSE`), not a causal signal about which
  *completed* passes create chances. Reference/diagnostic column only.
- **`passer_team_id`, `passer_player_id`** -- identity/nuisance columns, fold-safe
  metadata only, never published model features (same rule CxG applied to
  shooter/defending-team identity).
- **`receiver_x`, `receiver_y`** -- CxA+-only columns, not present in the event-only
  matrix at all; listed here for completeness since this doc is meant to be a
  self-contained reference. (Near-duplicate of `end_x`/`end_y` at r=0.9999 when they do
  appear, in CxA+.)
- **`pass_event_id`, `match_id`, `competition_id`, `season_id`, `receipt_event_id`,
  `data_version`, `silver_schema_version`, `feature_version`, `materialized_at`** --
  identifiers and pipeline provenance metadata, not features. `match_id` in particular
  is the correct *grouping* key for the train/validation/test split and must never be
  used as a model input.
- **`y_create`, `y_goal`** -- labels, never features (stating this explicitly despite
  being obvious, since this doc is the single reference other work should be able to
  read without cross-referencing three other documents).

## 5. What this doc does not cover

- **CxA+.** CxA+'s 2 locked 360 features
  (`reception_nearest_opponent_distance_m`, `reception_opponents_within_5m`) were
  confirmed strong on the full population in the pre-model analysis but have **not**
  been run through this same train/validation split check. That is separate,
  not-yet-scheduled follow-up work -- do not assume CxA+ features are lock-confirmed
  because this document exists.
- **Test-split confirmation.** Explicitly deferred, see section 2 and the split
  policy's sequential plan (steps 9-10: freeze happens after validation-split
  confirmation, test is run once at the end for final reporting, not before).
- **Model training, feature engineering beyond what's listed, or any modelling
  decision.** This document locks a candidate feature *list*. It does not train a
  model, does not choose a model family, and does not report a model metric.
