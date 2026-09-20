# CxA P_convert Pre-Model Target and Feature Analysis

Date: 2026-09-20
Reads only: `oam_features.cxconvert_event_v1_training_matrix` (11,303 rows),
`oam_features.cxconvert_plus_v1_training_matrix` (2,830 rows). Writes no BigQuery
tables. Does not train, score, or select a model.

Rubric mirrors `docs/analysis/cxa_p_create_pre_model_analysis.md` (P_create's own
pre-model analysis) exactly -- target usability/sparsity, per-feature signal, redundancy,
leakage/eligibility, modelling recommendations. All numbers below are freshly queried
from BigQuery for this task (`scripts/analyze_cxconvert_pre_model.py`, raw output under
`audit_outputs/cxconvert_analysis/pre_model_study/`), independent of the pipeline's own
`validate_cxconvert_training_matrix.py` checks. Full-population exploratory analysis
only, per the split policy's own "full-population EDA is valid exploratory work"
carve-out -- no train/validation split-aware analysis in this task, and no feature lock.

Candidate feature list, population definition, and the two leakage exclusions already
made at materialization time (`deflected`, `saved_off_target`, `saved_to_post`) are read
from `docs/cxa_convert_data_feasibility_audit.md`, not re-derived. Split sizes and
class balance per split are read from `docs/cxa_convert_split_policy_and_plan.md`, not
re-derived.

## 1. Target usability and sanity check

`Y_goal` is well-defined for every row by construction: both matrices' population is
already restricted to `y_create = TRUE` (see feasibility audit), so `y_goal` is a plain
non-null boolean on every row, not a conditional/subset target the way `y_create` was
relative to all passes. There is no `y_goal`-vs-`y_create` cross-tab to re-run here (that
check belongs to P_create's own analysis, already done); the only carry-forward sanity
item is the single known `is_completed = FALSE` anomaly row from P_create's audit, which
is confirmed still present, unchanged, in this population's `y_goal = FALSE` group
(`leakage_is_completed_event.json`: 1 row with `is_completed = FALSE`, 0 goals). Nothing
new to investigate here; not re-litigated further.

## 2. Target analysis by descriptive slices (`target_overall.json`,
`target_by_*.json`)

**Overall, confirming the feasibility audit's numbers exactly:** event 11,303 rows /
1,038 goals / 9.183%; plus 2,830 rows / 263 goals / 9.293%.

**By shot body part** -- Head shots convert somewhat higher (10.62% event / 9.86% plus)
than foot shots (8.6-8.8% both tracks); support for "Other" (39/18 rows) is too thin to
read its 12.8%/16.7% rate as reliable.

**By shot technique** -- a clear, large gradient: Normal (the 84% baseline majority)
converts at 8.53%/8.93%, while Volley (12.6%/14.2%), Diving Header (23.4%/20.0%), and Lob
(27.8%/22.2%) convert far higher. These are technique categories that structurally imply
the keeper is already beaten or badly positioned (a lob is by definition an attempt to
beat an out-of-position keeper) -- descriptively real, but every elevated level except
Volley has thin per-level support (50-130 rows), noted for signal-table purposes in
section 5.

**By pass type feeding the shot** -- passes with no special `pass_type_name` tag
(ordinary open play, 9,020/2,294 rows) convert at 9.58%/10.24%, essentially the
population baseline. `Corner` (7.01%/4.15%) and `Throw-in` (5.05%/0%) convert notably
*below* baseline -- set-piece deliveries create a chance (they're all `y_create=TRUE`
rows already) but the resulting shot is less likely to go in, plausibly because these
deliveries are contested crosses/headers into a crowded box rather than clean strikes.
`Free Kick` (10.68%/7.92%) sits closer to baseline. Descriptive only at this stage.

**By play pattern** -- `From Counter` converts highest in the event track (15.34%, 567
rows) but is closer to baseline in the plus track (11.63%, only 129 rows -- thinner
support); `From Corner` converts below baseline in both tracks (7.86%/7.14%), consistent
with the pass-type finding above (corner deliveries create chances but convert them at a
lower rate). `Regular Play` (the largest single category, 4,181/983 rows) sits almost
exactly at the population baseline in both tracks (9.04%/9.66%).

**By competition-season** (`target_by_competition_event.json`,
`target_by_competition_plus.json`): goal rate ranges 7.12%-10.59% across the five
event-track competition-seasons and 7.18%-10.48% across the plus track's three -- a real
but modest spread, all slices retain 90+ positives, no slice is unusable.

## 3. Target sparsity by match -- a genuine difference from P_create, flagged explicitly

`sparsity_by_match_event.json` / `sparsity_by_match_plus.json`:

| track | matches | matches with 0 goals | min/max goals per match | quartiles (goals) |
|---|---|---|---|---|
| event | 610 | **119 (19.5%)** | 0 / 7 | 0 / 1 / 1 / 2 / 7 |
| plus | 166 | **33 (19.9%)** | 0 / 6 | 0 / 1 / 1 / 2 / 6 |

**This is a real, new sparsity finding that did not apply to P_create** (P_create's own
pre-model analysis found *zero* matches with zero `y_create` positives, at both track
levels). Here, roughly 1 in 5 matches has **no** `Y_create=TRUE` pass that converted to
a goal at all -- expected given only ~9% of an already-rare event converts, and a
typical match contributes just ~18/17 `Y_create=TRUE` rows (median row count per match,
same range P_create's own audit already established). This does not invalidate the
match-level split (the split is still the right grouping key, and no split was found to
be too thin in the split policy doc), but it is a real property that must be carried
into the next, split-aware stage: **per-match evaluation of P_convert will be
meaningfully noisier than per-match P_create evaluation was**, and roughly a fifth of
matches in either track contribute a pure-negative example set with no positive to
learn from locally. Stated here explicitly so the modelling stage does not discover it
cold.

## 4. Feature missingness (carried-over pass context + shot attributes)

**Carried-over pass-context missingness is a strict subset of P_create's own
already-documented pattern** (same `pass_type_name`/`pass_technique_name`/
`pass_body_part_name` null-as-a-real-category convention -- not re-derived here, see
P_create's own pre-model analysis section 4). One number worth restating for this
population specifically: `pass_type_name` is tagged (non-null) for 2,283/11,303 (20.2%)
of event-track rows and 536/2,830 (18.9%) of plus-track rows -- a real jump up from
P_create's full-population 20.1% tagged rate... actually consistent, not a jump (the
P_create figure was computed over all passes, not just `Y_create=TRUE` ones; the near-
identical rate here is coincidental, not evidence of a relationship, and not pursued
further).

**Shot's own attributes -- one missingness pattern is load-bearing, not incidental:**
`shot_end_z` (shot height at the goal line), already flagged by the feasibility audit
as 32.2% null over the full `Y_create=TRUE` population, splits sharply by outcome
(`numeric_signal_event.json`, `numeric_signal_plus.json`):

| track | y_goal | rows | `shot_end_z` null | null % |
|---|---|---|---|---|
| event | FALSE | 10,265 | 3,634 | 35.4% |
| event | TRUE | 1,038 | **0** | **0%** |
| plus | FALSE | 2,567 | 896 | 34.9% |
| plus | TRUE | 263 | **0** | **0%** |

**Every single goal in both tracks has a non-null `shot_end_z`; roughly a third of
non-goals do not.** This is a structural StatsBomb tagging pattern, not random missing
data: a goal, by definition, reaches the goal frame and so always has a recorded
arrival height; a shot that misses well wide, gets blocked early, or is saved before
fully tracked may never get a final height recorded. **Flagged as leakage-adjacent, not
an outright exclusion** (unlike `deflected`/`saved_off_target`/`saved_to_post`, which
describe something that happened *to* the shot after it was struck): `shot_end_z`
itself is a real, causally-prior attribute of where the shot was aimed, and its non-null
values carry genuine signal (see section 5 -- goals arrive lower on average). But a raw
"is `shot_end_z` null" indicator would function as an almost-perfect target proxy the
same way `is_completed` did for `y_create` in P_create's own audit (section 8 there),
and naive mean-imputation of the null 35% would silently corrupt that group's
distribution. **This needs an explicit decision at the feature-lock stage** (impute with
a missingness flag and validate the flag doesn't dominate importance, vs. drop the
column, vs. use it only in careful combination with other geometry) -- not resolved
here, but must not be waved through as an ordinary numeric feature without this caveat
attached.

## 5. Feature signal, ranked strongest to weakest

Simple group comparison (mean/rate for `y_goal=TRUE` vs `y_goal=FALSE`), not a model.
Event-only track (`numeric_signal_event.json`, `numeric_signal_event_dist_to_goal.json`,
`cat_*_event.json`, `bool_*_event.json`):

| rank | feature | no-goal | goal | effect |
|---|---|---|---|---|
| 1 | `shot_open_goal` | 8.75% base | 59.38% | **6.8x lift**, but only 96/11,303 rows (0.85%) -- needs validation, not locked |
| 2 | `pass_technique_name = Through Ball` (== `is_through_ball`) | 8.27% base | 29.67% | **3.6x lift**, solid support (482 TRUE rows) |
| 3 | `shot_one_on_one` | 8.37% base | 24.96% | **3x lift**, solid support (553 rows) |
| 4 | `shot_technique_name = Lob` | ~8.5% base | 27.78% | 3.3x lift, thin support (72 rows) |
| 5 | `shot_technique_name = Diving Header` | ~8.5% base | 23.38% | 2.7x lift, thin support (77 rows) |
| 6 | distance to goal center (`shot_x_sb`/`shot_y_sb` derived) | 16.4m | 11.2m | goals are struck ~5.2m closer to goal on average -- large, robust effect |
| 7 | `shot_gk_distance_m` | 13.63m | 8.19m | goals struck ~5.4m closer to the keeper -- largely a restatement of #6 (keeper sits near the goal line), see redundancy section 6 |
| 8 | `shot_technique_name = Backheel` | ~8.5% base | 18.0% | 2.1x lift, very thin (50 rows) |
| 9 | `statsbomb_xg` (benchmark, not a candidate) | 0.0786 | 0.2273 | 2.9x -- expected, this is StatsBomb's own shot-quality model output, see section 8 |
| 10 | `shot_abs_y_from_center` (`|shot_y-40|`) | 8.23 | 5.53 | goals are struck from more central positions, meaningful and intuitive |
| 11 | `is_cross` | 7.99% base | 13.75% | **1.7x lift**, large support (2,350 TRUE rows) -- and confirmed *not* just a headers proxy, see section 7 overlap check |
| 12 | `shot_first_time` | 7.83% base | 13.02% | 1.66x lift, large support (2,942 rows) |
| 13 | `is_cut_back` | 9.06% base | 14.18% | 1.57x lift, moderate support (282 rows) |
| 14 | `pass_height_name = Low Pass` | 7.96% (Ground) baseline | 11.02% | 1.4x lift vs Ground, large support |
| 15 | `pass_end_x` (mean) | 100.40 | 106.56 | +6.2 units -- passes that convert end further upfield |
| 16 | `start_x` (mean) | 95.15 | 98.44 | +3.3 units -- weaker than end_x, expected |
| 17 | `is_switch` | 9.41% base | 6.19% | **inverse, 0.66x** -- switches convert *worse* than average |
| 18 | `pass_type_name = Corner` | 9.58% base (untagged) | 7.01% | inverse, 0.73x -- corners create chances but convert them less often |
| 19 | `shot_aerial_won` | 9.41% base | 7.56% | weak inverse |
| 20 | `shot_under_pressure` | 9.43% base | 8.47% | weak inverse, near-flat |
| 21 | `minute` (mean) | 49.66 | 51.20 | mild, later-match skew |
| 22 | `pass_length` (mean) | 23.40 | 23.82 | negligible |
| 23 | `shot_defenders_within_5m`/`_8m` | 2.28 / 3.93 | 2.52 / 4.24 | *higher*, counter-intuitively, for goals -- a distance-to-goal confound, not a real "more pressure helps" effect, see section 6 |
| 24 | `shot_frame_player_count` / `shot_defenders_visible` | 13.13 / 8.66 | 11.70 / 7.92 | goals happen in slightly less crowded frames overall (fewer total players visible), consistent with counter-attacking / space-creation intuition, but weak relative to the radius-band features above |

CxA+ track (`numeric_signal_plus.json`, `cat_*_plus.json`, `bool_*_plus.json`) --
**same ranking and direction on every shared feature** (`shot_open_goal` 6.8x on 21/2,830
rows, `is_through_ball` 3.5x on 131 rows, `shot_one_on_one` 2.5x on 130 rows, `is_cross`
1.9x on 629 rows, `is_switch` inverse 0.33x on 188 rows, distance-to-goal and
`shot_gk_distance_m` both show the same ~5m gap), plus the CxA+-only reception-time 360
features:

| feature | no-goal | goal | effect |
|---|---|---|---|
| `reception_nearest_opponent_distance_m` | 3.77m | 2.88m | receiver is more tightly marked on converting chances -- real signal, though a smaller gap than P_create's own `y_create` version of this same feature (which distinguished chance-creation, not conversion) |
| `reception_opponents_within_5m` | 1.67 | 2.05 | 1.2x, modest |
| `reception_opponents_visible`/`reception_teammates_visible`/`reception_frame_player_count` | 8.99 / 6.30 / 15.29 | 8.37 / 5.57 / 13.94 | all slightly lower for goals -- same "less crowded" pattern as the event track's shot-freeze-frame counts, consistent direction, modest magnitude |

**Strongest overall candidates for a first modelling pass:** `is_through_ball`,
`shot_one_on_one`, distance-to-goal (`shot_x_sb`/`shot_y_sb` or the derived distance),
`is_cross`, `shot_first_time` -- these combine real effect size with large support and
hold in the same direction across both tracks. `shot_open_goal` has the single largest
lift in the whole table but thin support (needs split-validation, not locked outright,
same treatment P_create gave its own high-lift/thin-support levels).

## 6. Thin-support and near-constant re-check

**Re-examined, per the task's explicit instruction** (`thin_support_event.json`,
`thin_support_plus.json`, plus the `cat_shot_type_*`/`bool_shot_counterpress_*` query
results):

- **`follows_dribble`**: 7/11,303 (event) and 3/2,830 (plus) TRUE, **0 goals in either
  group**. Confirmed thin exactly as the feasibility audit flagged it. **Drop --
  thin-support**, zero-lift as well as zero-support, nothing to validate.
- **`shot_type_name`**: reconfirmed 100% `Open Play` in both tracks (`cat_shot_type_event`/
  `_plus.json`, single row each). **Excluded entirely, not carried into any further
  analysis**, exactly as instructed.
- **New finding, not flagged by the feasibility audit: `shot_counterpress` is also
  constant -- 100% `FALSE` across all 11,303 event rows and all 2,830 plus rows**
  (`bool_shot_counterpress_event.json`/`_plus.json`, single group each; independently
  confirmed by `overlap_under_pressure_vs_counterpress_event.json`, which shows every
  row falls in `shot_counterpress = FALSE`). This makes structural sense on reflection
  (`counterpress` marks pressure applied *immediately after losing the ball*, which
  doesn't apply to the team about to shoot) but was not caught at the audit stage.
  **Drop -- zero variance, no signal possible.**
- Other elevated-lift-but-thin levels already surfaced in section 5 (`shot_open_goal`
  96/21 rows, `shot_technique_name` Lob/Diving Header/Backheel/Overhead Kick 50-77 rows
  each, `pass_body_part_name = No Touch` 32/9 rows, `pass_technique_name = Straight`
  109 rows) are **not** dropped -- same treatment P_create gave comparable levels
  (e.g. its own "No Touch" body-part finding): real, large lift on modest support,
  routed to "needs split-validation" rather than "drop," per section 9.

## 7. Feature redundancy and overlap

**Pairwise numeric correlations** (`redundancy_numeric_event.json`,
`redundancy_numeric_plus.json`):

| pair | r | reading |
|---|---|---|
| `shot_defenders_visible` <-> `shot_frame_player_count` | **0.9442** | near-duplicate -- total frame headcount is almost entirely driven by the opponent count in this population. **Recommendation: drop `shot_frame_player_count`, keep `shot_defenders_visible`** (the more directly interpretable of the two -- opponent count near the shot, not raw total headcount). |
| `pass_end_y` <-> `shot_y_sb` | **0.8904** | near-duplicate -- the shot is taken from almost the same lateral position the pass arrived at. **Recommendation: drop `pass_end_y` for a linear baseline, keep `shot_y_sb`** (the shot's own attribute is causally closer to the outcome; harmless to keep both for tree models, same convention P_create used for its own redundant-but-not-identical pairs). |
| `pass_end_x` <-> `shot_x_sb` | 0.7887 | related but **not** a near-duplicate (below the 0.8 threshold P_create's own audit used) -- keep both. |
| `receiver_x` <-> `pass_end_x` (plus only) | **1.0000** | exact duplicate, same finding P_create's own audit already made for this pair (its `receiver_x`/`receiver_y` vs `end_x`/`end_y` finding) -- carries straight through unchanged since these are the identical underlying columns. **Recommendation: drop `receiver_x`/`receiver_y`**, already flagged, reaffirmed here, not re-litigated. |
| `receiver_x` <-> `shot_x_sb` (plus only) | 0.7947 | related, not a duplicate -- keep. |
| `shot_defenders_within_5m` <-> `shot_defenders_within_8m` | 0.7714 | related, not quite a duplicate (below 0.8) -- keep both, same treatment as P_create's own radius-band pairs. |
| `shot_gk_distance_m` <-> `shot_defenders_within_5m`/`_8m` | -0.4439 / -0.4548 | moderate inverse, expected (crowded penalty-box shots tend to be closer to the keeper too) -- not redundant, both add information. |
| `statsbomb_xg` <-> `shot_gk_distance_m` | -0.5566 (event) / -0.5592 (plus) | moderate, expected (xG is itself partly a function of shot distance) -- confirms `statsbomb_xg` behaves as a geometry-driven benchmark, not an arbitrary external number; not a reason to treat it as an ordinary candidate (see section 8). |
| `statsbomb_xg` <-> `shot_defenders_within_5m` | 0.1522 | weak. |
| `reception_nearest_opponent_distance_m` <-> `shot_gk_distance_m` (plus) | 0.5930 | moderate -- reception-time marking pressure and shot-time keeper distance are related but distinct moments, not redundant. |
| `reception_opponents_within_5m` <-> `shot_defenders_within_5m` (plus) | 0.7447 | moderate-high but below 0.8 -- different moments (reception vs. shot), keep both, consistent with the feasibility audit's framing of these as complementary, not redundant. |

**Categorical/boolean overlap checks** (`overlap_*.json`):

- **`shot_one_on_one` x `shot_open_goal`**: mostly disjoint categories (only 19/11,303
  rows have both TRUE). When both are TRUE the rate (57.9%) tracks `open_goal` alone
  (59.7%), not the product of the two lifts -- `open_goal` dominates when both apply,
  but each also carries real independent signal in the (far more common) case where only
  one is TRUE (`one_on_one` alone: 23.8%; neither: 8.0%). **Not redundant -- keep both.**
- **`shot_first_time` x `pass_technique_name = Through Ball`**: the Through Ball lift
  holds regardless of `first_time` (33.5% not-first-time vs. 25.1% first-time, both far
  above the ~8-13% baseline for other technique levels at the same `first_time` value).
  **Complementary signals, not redundant** -- each contributes information the other
  doesn't fully capture.
- **`is_cross` x `shot_body_part_name`**: crosses raise the goal rate **within every
  body-part category**, not just headers (Right Foot: 7.79% -> 15.64%; Left Foot: 7.87%
  -> 13.70%; Head: 8.89% -> 12.59%). This is a useful negative result for a hypothesis
  worth checking explicitly: `is_cross` is **not** simply a redundant proxy for "the shot
  was a header" -- it carries real information about the chance's quality independent of
  finishing technique. **Not redundant -- keep both.**

## 8. Leakage re-confirmation

**`statsbomb_xg` reconfirmed reference/benchmark-only, not an ordinary candidate**
(`leakage_statsbomb_xg_correlation_event.json`/`_plus.json`): correlation with `y_goal`
is 0.3914 (event) / 0.3818 (plus) -- a real, moderate, expected relationship for a shot
quality model's own output, not a literal duplicate of the label (a duplicate would
correlate far higher and non-linearly). This is exactly the treatment the feasibility
audit already assigned it (section 4b there) -- reaffirmed, not reconsidered, and
carried through the same way CxG's own xG-vs-predicted-target benchmark comparisons were
always kept separate from the candidate feature list.

**`pass_outcome_name`/`is_completed` (event-only track; these columns do not exist in
the plus-track schema at all, confirmed by the table schema, not assumed) are reaffirmed
excluded, not reconsidered as candidates**, despite looking "safer" in this population
than they did for P_create: `leakage_is_completed_event.json` shows 11,302/11,303 rows
`TRUE` and exactly 1 `FALSE` (the carried-over section-1 anomaly, 0 goals). Because this
population is already restricted to `Y_create=TRUE` passes, `is_completed` is *even
closer to a constant here* than it was over P_create's full population -- which makes it
useless as a `Y_goal` predictor (near-zero variance) on top of being excluded on
methodology grounds (it was the near-perfect proxy for the upstream target, `y_create`,
not a causal signal about `y_goal`). Its near-constancy in this subset is additional
evidence for exclusion, not a reason to revisit it.

**Identity columns** (`passer_team_id`, `passer_player_id`) are carried as metadata only
in both matrices, per the feasibility audit's original decision -- not reconsidered as
candidates here.

**Already-excluded outcome fields** (`deflected`, `saved_off_target`, `saved_to_post`)
were excluded at materialization time (feasibility audit section 4b) and are not present
as columns in either training matrix at all -- there is nothing to re-examine for them
in this analysis; noted for completeness only.

**One new leakage-adjacent caution surfaced by this analysis, not present in the
original audit:** `shot_end_z`'s missingness pattern (section 4) -- not excluded, but
flagged for careful handling at the feature-lock stage rather than treated as an
ordinary numeric column with incidental nulls.

**Result: leakage audit passes**, with one new caution added (`shot_end_z`
missingness) beyond what the feasibility audit and P_create's own exclusion precedents
already covered.

## 9. Modelling recommendations

This section makes no modelling decision -- it hands off an evidence-based punch list for
the next (split-aware feature lock) step, categorized the same way P_create's own
pre-model analysis closed out.

**Locked (large effect size + large support, consistent direction across both tracks):**
- `is_through_ball` (event: `pass_technique_name = Through Ball`; same underlying tag in
  both tracks)
- `shot_one_on_one`
- `is_cross` (confirmed not redundant with `shot_body_part_name`, section 7)
- `shot_first_time`
- `shot_x_sb`, `shot_y_sb` (shot location / derived distance-to-goal -- largest, most
  robust effect in the whole table)
- `shot_gk_distance_m` (real signal, though partially a restatement of shot distance to
  goal -- keep both, tree models can decompose the shared variance)
- `shot_defenders_within_5m`, `shot_defenders_within_8m` (real signal once the
  distance-to-goal confound is understood, not dropped -- flag the confound for the
  modelling stage rather than removing the feature)
- `pass_end_x`, `start_x` (event-only track's carried pass context; `receiver_x`
  dropped in favor of `pass_end_x` for CxA+, section 7)
- CxA+ only: `reception_nearest_opponent_distance_m`, `reception_opponents_within_5m`

**Needs split-validation (elevated lift, smaller support, or single-slice risk):**
- `shot_open_goal` (largest single lift in the table, 96/21 rows)
- `shot_technique_name` (Lob/Diving Header/Backheel/Volley levels; Normal is baseline)
- `is_cut_back`
- `pass_body_part_name = No Touch`
- `pass_technique_name = Straight`
- `is_switch` (inverse direction -- worth confirming the negative lift holds, not just
  its magnitude)
- `pass_type_name = Corner` (inverse direction, confirmed descriptively in section 2,
  worth validating before trusting as a real conversion-suppressing effect vs. a
  confound with shot technique/body part on corners)

**Needs careful handling, not a plain locked/drop call (new finding this stage):**
- `shot_end_z` -- real signal in its non-null values (goals arrive lower), but its null
  pattern is near-perfectly correlated with `y_goal = FALSE` (0% null for goals, ~35%
  null for non-goals in both tracks). Decide impute-with-missingness-flag vs. drop vs.
  restricted use at the feature-lock stage; do not promote a naive
  `shot_end_z_is_null` indicator without this caveat attached.

**Drop -- thin support / zero variance:**
- `shot_follows_dribble` (0 goals in 7/3 TRUE rows, no signal possible)
- `shot_type_name` (100% constant `Open Play`, already excluded per the feasibility
  audit)
- `shot_counterpress` (100% constant `FALSE`, new finding this stage, section 6)

**Drop -- redundant:**
- `shot_frame_player_count` (r=0.9442 with `shot_defenders_visible`, section 7)
- `pass_end_y` (r=0.8904 with `shot_y_sb`, section 7 -- keep for tree models, drop for
  a linear baseline, same convention P_create used)
- `receiver_x`, `receiver_y` (plus track; r=1.0000 with `pass_end_x`/`pass_end_y`,
  reaffirmed from P_create's own finding)

**Benchmark-only, never an ordinary candidate:**
- `statsbomb_xg` (section 8)

**Excluded (leakage / methodology, reaffirmed not reconsidered):**
- `pass_outcome_name`, `is_completed` (event-only schema; even closer to constant in
  this population than in P_create's, reinforcing rather than weakening the exclusion)
- `passer_team_id`, `passer_player_id` (identity/nuisance columns)
- `deflected`, `saved_off_target`, `saved_to_post` (excluded before materialization,
  not present in either matrix's schema)

**Needs more work before modelling, not resolved by this analysis:**
- 19.5% of event-track matches and 19.9% of plus-track matches have zero
  `Y_goal=TRUE` rows (section 3) -- a genuine sparsity difference from P_create's own
  analysis (which found zero such matches). This must be stated explicitly wherever
  P_convert model evaluation is documented, and factored into how per-match/per-slice
  metrics are interpreted at the modelling stage.
- CxA+'s inherited absence of Premier League rows (already documented by P_create's own
  pre-model analysis, unchanged here since P_convert's plus population is a strict
  subset) continues to scope any CxA+ P_convert model's generalization claim to
  tournament football only.

**Not evaluated here, left to the modelling stage per the split policy:** pairwise/
interaction effects beyond the specific overlap checks in section 7, calibration, and
any comparison against the `statsbomb_xg` benchmark. This analysis is univariate/
bivariate by design, matching the "real analysis, not a model" scope of this task.
