# CxA+ P_create Feature Lock v1

Date: 2026-09-19
Track: CxA+ (360-at-reception) only (`oam_features.cxa_plus_v1_training_matrix`).
Structured to mirror
[`docs/analysis/cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md)
(the event-only track's lock), but this document is self-contained -- it does not
assume that one has been read.

**Tournament-only population, restated here and on every CxA+ artifact going
forward, not just the original feasibility audit:** `cxa_plus_v1_training_matrix`
contains **zero Premier League rows**. Its 133,143 rows / 166 matches are 100% FIFA
World Cup + UEFA Euro (StatsBomb's 360 coverage in this corpus is tournament-only).
Any CxA+ model's generalization claim is scoped to international tournament football
only -- it says nothing about domestic league play, which has systematically different
tempo, spacing, and squad rotation. This caveat applies to the feature lock below the
same way it will apply to any future CxA+ model card.

Split table: `oam_analysis.cxa_match_splits_v1`, filtered to the 166 matches present in
`cxa_plus_v1_training_matrix` -- train = 119 matches / 95,083 rows, validation = 24
matches / 19,490 rows, **test = 23 matches, sealed**. `create_rate`: train 2.094%,
validation 2.155%, test 2.256% (all three within 0.16pp, split balance holds on this
smaller population too).

## 1. Locked features (9)

Confirmed by
[`scripts/validate_cxa_plus_features_on_split.py`](../../scripts/validate_cxa_plus_features_on_split.py)
(raw output:
[`audit_outputs/cxa_analysis/plus_feature_lock/split_validation_result.json`](../../audit_outputs/cxa_analysis/plus_feature_lock/split_validation_result.json)),
reusing the event-only track's script structure and promotion rule rather than
reinventing it:

- Boolean/categorical features: `lift = rate(y_create | flag=TRUE) / rate(y_create |
  flag=FALSE)`.
- Promotion rule: direction must not flip (lift > 1x on train AND validation) AND
  validation lift >= **2.0x** (same threshold, same justification as the event-only
  lock -- a property of the statistic, not of the track).
- `reception_opponents_within_5m` is a continuous count, not a boolean flag, but its
  confirmation statistic is still a ratio (mean count at `y_create=TRUE` over mean
  count at `y_create=FALSE`) and reuses the identical >=2.0x / no-direction-flip rule.
- `reception_nearest_opponent_distance_m` is reported as a **gap** (mean distance at
  `y_create=FALSE` minus mean distance at `y_create=TRUE`, since lower distance is the
  stronger signal here) with an analogous, separately-justified floor of **1.0m** --
  see the script's docstring for why.

| feature | train stat | validation stat | verdict |
|---|---|---|---|
| `is_cross` | 35.89x | 28.64x | Confirmed -- direction holds, largest support of the four strongest (796 train / 154 validation TRUE rows) |
| `is_through_ball` | 24.70x | 26.19x | Confirmed -- **strengthens** on validation |
| `pass_type_name = Corner` | 23.14x | 23.69x | Confirmed -- **strengthens** on validation |
| `play_pattern_name = From Counter` | 8.19x | 8.88x | Confirmed -- **strengthens** on validation |
| `pass_technique_name = Outswinging` (also confirmed for the `Straight` level, see note below) | 33.50x | 30.05x | Confirmed -- large lift, thinner support (142 train / 29 validation TRUE) |
| `is_cut_back` | 25.47x | 22.61x | Confirmed -- direction holds, thin support (102 train / 21 validation TRUE) |
| `is_switch` | 3.19x | 3.29x | Confirmed -- weakest boolean lift of the group but **strengthens** on validation, large support (1,744 train / 383 validation TRUE) |
| `reception_nearest_opponent_distance_m` | gap 3.75m | gap 3.51m | Confirmed -- direction and magnitude hold, full-population coverage (no support concern) |
| `reception_opponents_within_5m` | 3.63x | 3.41x | Confirmed -- direction and magnitude hold, full-population coverage |

That is 8 statistics for 9 named features: `pass_technique_name` locks both the
`Outswinging` and `Straight` levels as a single categorical feature (matching the
pre-model analysis's original framing of this column as one feature with multiple
informative levels, same as `pass_type_name` locking both `Corner` and `Free Kick`
below) -- `Straight` was population-strong in the pre-model analysis
(23.96% vs 1.85% baseline) and is carried into this lock as part of the same
categorical feature as `Outswinging`, not separately re-verified against the split
here (it was not one of the four features flagged for re-validation and is treated the
same way the event-only lock treated its own already-population-strong features).

**All 9 hold direction on validation, with no meaningful decay.** Four of the eight
ratio statistics actually strengthen from train to validation (`is_through_ball`
24.70x->26.19x, `pass_type_name=Corner` 23.14x->23.69x, `play_pattern_name=From
Counter` 8.19x->8.88x, `is_switch` 3.19x->3.29x); the other four (`is_cross`,
`pass_technique_name=Outswinging`, `is_cut_back`, `reception_opponents_within_5m`) and
the distance gap decay modestly but stay large multiples of the promotion floor. This
is a cleaner result than the event-only track, where every one of the four
re-validated features decayed somewhat from train to validation -- expected, since
CxA+'s lift magnitudes are far higher to begin with (chance creation is a rarer, more
concentrated event in this 360 population), so proportionally similar sampling noise
moves the number less relative to its size. See section 2 for the one feature that did
show real, meaningful decay.

Reference (not re-derived here): `start_x`/`end_x` are already locked as part of the
shared event-only feature base
([`docs/analysis/cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md),
section 1a) and carry over into the CxA+ candidate set unchanged -- they are computed
identically in both training matrices and were not re-checked against this smaller,
tournament-only split.

## 2. `pass_body_part_name = No Touch` -- not promoted to the same confidence tier

| feature | train stat | validation stat | verdict |
|---|---|---|---|
| `pass_body_part_name = No Touch` | 3.32x | 2.11x | Mechanically clears the 2.0x floor (barely) -- **not treated as confirmed at the same confidence as the other 9** |

This is the one real magnitude decay in the CxA+ set (3.32x -> 2.11x, a 36% relative
drop -- far larger than any other feature's train-to-validation change) on the
thinnest support of any candidate feature in this entire project: **22 validation TRUE
rows, of which only 1 is a positive** (`y_create=TRUE`). A single row changing outcome
would materially move this number; the script's mechanical 2.0x threshold does not
know that and reports PASS, which is exactly why this needs a human judgment call
layered on top of the mechanical check rather than folding it into section 1's list
silently.

**Recommendation: hold this feature out of the first CxA+ baseline, and revisit once
test opens.** The case for inclusion (direction holds, still-elevated lift, matches
the event-only track's own `No Touch` finding which had a somewhat sturdier validation
sample) is real but not strong enough, on 1 positive validation example, to justify
locking it at the same tier as the other 9 -- which either held flat or strengthened
on a comparable or larger validation sample. If a modeller wants to include it anyway
for a first pass, it should carry an explicit caveat in whatever reports that model's
features (e.g. "included despite n=22 validation support, unconfirmed at test"), not
be listed as an ordinary locked feature.

## 3. Exclusion list (restated -- this doc is self-contained)

The following must **not** enter the CxA+ P_create baseline as candidate features, per
`docs/cxa_data_feasibility_audit.md` (point 6) and
`docs/analysis/cxa_p_create_pre_model_analysis.md` (sections 6 and 8):

- **`receiver_x`, `receiver_y`** -- near-duplicate of `end_x`/`end_y` at r=0.9999
  (pre-model analysis, section 6). Adds no information `end_x`/`end_y` doesn't already
  carry; including both would inflate apparent importance for one underlying signal.
- **`pass_outcome_name`, `is_completed`** -- near-perfect target proxy (2,943x rate
  difference on the event-only population; the same completion-outcome field, same
  reasoning applies here), not a causal signal about which *completed* passes create
  chances. Reference/diagnostic column only.
- **`passer_team_id`, `passer_player_id`** -- identity/nuisance columns, fold-safe
  metadata only, never published model features.
- **`pass_event_id`, `match_id`, `competition_id`, `season_id`, `receipt_event_id`,
  `data_version`, `silver_schema_version`, `feature_version`, `materialized_at`** --
  identifiers and pipeline provenance metadata, not features. `match_id` is the correct
  *grouping* key for the split and must never be a model input.
- **`y_create`, `y_goal`** -- labels, never features.

## 4. Full locked feature list (9, plus 1 held-out)

| # | feature | type | train / validation |
|---|---|---|---|
| 1 | `is_cross` | boolean | 35.89x / 28.64x |
| 2 | `is_through_ball` | boolean | 24.70x / 26.19x |
| 3 | `pass_type_name` (Corner, Free Kick levels) | categorical | Corner 23.14x / 23.69x (Free Kick population-strong, not re-verified here, see section 1) |
| 4 | `play_pattern_name` (Counter, Corner levels) | categorical | Counter 8.19x / 8.88x (Corner population-strong, not re-verified here) |
| 5 | `pass_technique_name` (Outswinging, Straight levels) | categorical | Outswinging 33.50x / 30.05x (Straight population-strong, not re-verified here) |
| 6 | `is_cut_back` | boolean | 25.47x / 22.61x |
| 7 | `is_switch` | boolean | 3.19x / 3.29x |
| 8 | `reception_nearest_opponent_distance_m` | numeric (360) | gap 3.75m / 3.51m |
| 9 | `reception_opponents_within_5m` | numeric (360) | 3.63x / 3.41x |
| -- | `start_x`, `end_x` | numeric | shared event-only base, reference only (see section 1) |
| held out | `pass_body_part_name = No Touch` | boolean | 3.32x / 2.11x -- see section 2, not locked at this tier |

## 5. What this doc does not cover

- **Test-split confirmation.** Explicitly deferred. Test (23 matches) stays sealed
  until model training and a final report are otherwise complete, per the split
  policy's sequential plan. `pass_body_part_name = No Touch`'s held-out status (section
  2) should be the first thing re-checked once test opens.
- **The event-only track.** Already locked separately in
  [`docs/analysis/cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md).
  Not re-derived or re-verified here.
- **Model training, feature engineering beyond what's listed, or any modelling
  decision.** This document locks a candidate feature *list* for CxA+. It does not
  train a model, does not choose a model family, and does not report a model metric.
