# CxA P_create Pre-Model Target and Feature Analysis

Date: 2026-09-19
Reads only: `oam_features.cxa_event_v1_training_matrix` (608,722 rows),
`oam_features.cxa_plus_v1_training_matrix` (133,143 rows, 166 matches). Writes no
BigQuery tables. Does not train, score, or select a model.

Rubric mirrors the structure of the earlier (deleted, pre-methodology-lock) CxA
diagnostic layer -- target usability, target sparsity, per-feature signal, redundancy,
slice stability, leakage/eligibility, modelling recommendations -- not its code or its
data assumptions (that attempt ran against a different, now-superseded
`action_features` table). All numbers below are freshly re-queried from BigQuery for
this task (`scripts/analyze_cxa_p_create_pre_model.py`, raw output under
`audit_outputs/cxa_analysis/pre_model_study/`), independent of the pipeline's own
`validate_cxa_training_matrix.py` checks.

Each numbered section below states the question, how it was answered, what the numbers
show, and what that means for modelling -- the same discipline the mirrored rubric
used, condensed into prose per section instead of a fixed key/value template.

## 1. Target usability

**Question:** is `y_create` cleanly separable from `y_goal`, and is there leakage
between the two targets?

**Cross-tab, both tracks** (`target_cross_tab.json`):

| track | y_create | y_goal | rows |
|---|---|---|---|
| cxa_event_v1 | FALSE | FALSE | 597,419 |
| cxa_event_v1 | TRUE | FALSE | 10,265 |
| cxa_event_v1 | TRUE | TRUE | 1,038 |
| cxa_plus_v1 | FALSE | FALSE | 130,313 |
| cxa_plus_v1 | TRUE | FALSE | 2,567 |
| cxa_plus_v1 | TRUE | TRUE | 263 |

`y_goal = TRUE` never occurs with `y_create = FALSE` in either table (0 rows for that
combination) -- `y_goal` is a strict subset of `y_create = TRUE` by construction, exactly
as the pipeline intended (`y_goal` = the linked shot became a goal, only defined when a
linked shot exists at all). No leakage between the two targets: `y_goal` cannot be used
to predict `y_create` because it is only ever observed *given* `y_create = TRUE`, and no
`y_create = FALSE` row carries information that would let a model infer a
hypothetical `y_goal`. **Target usability: clean.**

## 2. Sanity check -- the known anomaly (1 row, `is_completed = FALSE` AND `y_create = TRUE`)

Investigated specifically, not just counted (`anomaly_row_event.json`):

- `pass_event_id = 4b222484-3ad0-41b7-b789-930ead1b17bb`, match 3788766 (UEFA Euro),
  minute 28. The pass's own `outcome_name = 'Incomplete'`.
- Its linked shot (`shots.key_pass_id` = this pass): `event_id =
  71ab8b64-054b-44c8-9059-9a04fd8906e2`, shooter `player_id = 7156`, outcome
  `'Wayward'` (off target, not a goal -- consistent with `y_goal = FALSE` for this row).
- The pass's own `recipient_id = 7156` -- **the intended recipient is the same player
  who took the shot.** The pass row also carries `shot_assist = TRUE`.

**Explanation:** this is a genuine StatsBomb tagging inconsistency between two
independently-computed fields on the same underlying event, not a pipeline bug and not
something to "fix" by forcing `is_completed`/`y_create` to agree. `shots.key_pass_id`
and `passes.shot_assist` are both derived from the same underlying assist link and both
agree the pass created the shot; `passes.outcome_name` is StatsBomb's separate
completion-geometry classification, and in this one case it did not classify the
ball's arrival at the shooter as a clean "Complete" reception (most likely an
un-controlled first-time strike off a ball that technically never settled at the
receiver's feet -- StatsBomb's own documentation notes pass completion can disagree
with assist tagging for exactly this kind of instantaneous first-time contact). Dropping
this row, or overriding its label to match `is_completed`, would silently discard a
real true positive to enforce an assumption (own-outcome-implies-own-completion) that
the audited data itself disproves. **Keep the row, keep `y_create = TRUE` as-is; this
is the correct, if rare, ground truth** (1 row out of 608,722 -- 0.00016% -- has no
measurable effect on class balance or training regardless of how it's handled).

## 3. Target sparsity and class balance

**By competition/season** (`sparsity_by_competition_event.json`,
`sparsity_by_competition_plus.json`):

| track | competition | season | rows | positives | create rate | matches |
|---|---|---|---|---|---|---|
| event | Premier League | 27 | 368,619 | 7,175 | 1.946% | 380 |
| event | FIFA World Cup | 3 | 62,881 | 1,180 | 1.877% | 64 |
| event | FIFA World Cup | 106 | 68,515 | 1,039 | 1.516% | 64 |
| event | UEFA Euro | 43 | 54,819 | 926 | 1.689% | 51 |
| event | UEFA Euro | 282 | 53,888 | 983 | 1.824% | 51 |
| plus | FIFA World Cup | 106 | 51,089 | 992 | 1.942% | 64 |
| plus | UEFA Euro | 43 | 41,016 | 891 | 2.172% | 51 |
| plus | UEFA Euro | 282 | 41,038 | 947 | 2.308% | 51 |

Every competition-season slice in both tracks has 900+ positives and a create rate
between 1.5% and 2.3% -- **no slice is too sparse to model reliably at this grain.**

**Critical composition finding, not in the original scope but load-bearing for
modelling:** `cxa_plus_v1` has **zero rows from the Premier League** (competition_id
2). CxA+'s 133,143 rows are 100% FIFA World Cup + UEFA Euro (international
tournament football only), while CxA event-only is 61% Premier League by row count.
This is expected (StatsBomb's 360 coverage in this corpus is tournament-only) but it
means **a CxA+ model cannot be assumed to generalize to domestic league play** --
tournament football has systematically different tempo, spacing, and squad rotation
than a 38-game league season. This must be stated explicitly in any CxA+ model card,
not left implicit.

**By match** (`sparsity_by_match_event.json`, `sparsity_by_match_plus.json`): 0 matches
with zero `y_create` positives in either track. CxA event-only: 610 matches, positives
per match range 4-37 (quartiles 4/15/18/22/37); only 1 match has fewer than 6 positives.
CxA+: 166 matches, positives per match range 5-34 (quartiles 5/13/17/20/34); only 1
match has fewer than 6. **No match-level sparsity problem** -- every match contributes a
usable number of positive examples, which also supports the match-level
train/validation/test split design (no split can accidentally land a near-zero-positive
match).

## 4. Feature analysis -- distribution and missingness

**Numeric features** (`missingness_event.json`): `pass_length`, `pass_angle`,
`start_x`, `start_y`, `end_x`, `end_y`, `minute`, `second`, `possession_id` are **0%
null** across all 608,722 rows -- these come straight off `oam_core.events`/`passes`
required-ish columns and have no missingness to handle.

**Categorical features** -- missingness here is StatsBomb's own tagging convention,
not a data-quality defect, and must be encoded as a real category, not imputed away:

| feature | null count | null % | why |
|---|---|---|---|
| `pass_height_name` | 0 | 0% | always tagged (Ground/Low/High Pass) |
| `pass_body_part_name` | 40,912 | 6.7% | untagged for some pass types (see body-part table below -- concentrated in Kick-Off/Keeper-arm-adjacent rows) |
| `pass_type_name` | 486,454 | 79.9% | StatsBomb only tags a *special* type (Corner/Free Kick/Throw-in/Kick Off/Recovery/Interception); regular open-play passes are left null by design |
| `pass_technique_name` | 601,582 | 98.8% | StatsBomb only tags technique for swerved/driven deliveries (Inswinging/Outswinging/Straight/Through Ball); an ordinary pass has none |

Modelling implication: `pass_type_name` and `pass_technique_name` are not "mostly
missing data" to drop -- their null value **is** the informative "ordinary pass"
category and must be kept as an explicit level (`(null=open play)` /
`(null=regular)`), not imputed with a mode or dropped as a sparse column. The signal
tables in section 5 confirm this: the null level for both columns has a near-baseline
create rate, while every non-null (special) level is elevated.

**CxA+ 360 features** (`redundancy_plus_receiver_vs_end.json`): `receiver_x`/
`receiver_y` and `reception_nearest_opponent_distance_m` /
`reception_opponents_within_5m`/`_8m` are null for **154 of 133,143 rows (0.12%)** --
these are receptions whose 360 frame exists (so the row qualifies for the CxA+
population) but has no valid-coordinate actor row for the receiver (an incomplete
frame capture). Negligible in volume; candidate handling is either drop these 154 rows
or impute distance features with a sentinel + a `reception_geometry_missing` flag
before training -- a decision for the modelling stage, not resolved here.

## 5. Feature signal, ranked strongest to weakest

Simple group comparison (mean/rate for `y_create=TRUE` vs `y_create=FALSE`), not a
model. CxA event-only track (`numeric_signal_event.json`,
`categorical_signal_*.json`):

| rank | feature | no-create | create | effect |
|---|---|---|---|---|
| 1 | `pass_type_name = Corner` | 0.98% base rate | 17.64% | **18x lift**, largest single-level effect in the table |
| 2 | `is_cut_back` | 0.155% | 2.495% | **16x lift** (rare flag, ~2,300 TRUE rows total) |
| 3 | `pass_technique_name = Outswinging` | 1.64% base | 24.73% | **15x lift** |
| 4 | `is_through_ball` | 0.332% | 4.264% | **12.8x lift** |
| 5 | `play_pattern_name = From Counter` | 1.61% base (Regular Play) | 14.40% | **9x lift**, small support (3,939 rows) |
| 6 | `is_cross` | 2.135% | 20.791% | **9.7x lift**, large support (~14,900 TRUE rows) -- best combination of strength and volume |
| 7 | `start_x` (mean) | 58.59 | 95.46 | +36.9 units (pitch is 0-120) -- chance-creating passes originate deep in the attacking third, not just anywhere |
| 8 | `end_x` (mean) | 65.8 | 100.96 | +35.2 units -- correlated with `start_x`, see redundancy (section 6) |
| 9 | `play_pattern_name = From Corner` | 1.61% base | 8.45% | 5.3x lift, large support (22,128 rows) |
| 10 | `pass_body_part_name = No Touch` | 0.94% (Head) baseline | 6.14% | 6.5x lift, tiny support (521 rows) -- likely a deflection-into-a-shot pattern, worth a closer look before trusting it |
| 11 | `is_switch` | 2.778% | 7.007% | 2.5x lift |
| 12 | `pass_height_name = High Pass` | -- | 3.05% vs 1.51% (Ground) | ~2x lift, large support |
| 13 | `pass_length` (mean) | 21.49 | 23.44 | modest (+9%) |
| 14 | `minute` (mean) | 45.33 | 49.8 | mild (chance creation skews slightly later in matches) |
| 15 | `pass_angle` (mean abs) | 1.285 | 1.359 | weak |
| 16 | `start_y` / `end_y` | ~40.0 both groups | ~40 both groups | **no signal** -- expected; pitch width has no inherent attacking-direction bias |

CxA+ 360 features (`signal_360_plus.json`), same ranking logic:

| rank | feature | no-create | create | effect |
|---|---|---|---|---|
| 1 | `reception_nearest_opponent_distance_m` (mean) | 7.39m | 3.69m | receiver is roughly **twice as tightly marked** on chance-creating receptions -- strongest 360 signal, large effect, full population coverage |
| 2 | `reception_opponents_within_5m` (mean) | 0.47 | 1.71 | **3.6x** more close pressure |
| 3 | `reception_opponents_within_8m` (mean) | 1.08 | 3.25 | **3x** more pressure in the wider radius |
| 4 | `reception_teammates_visible` (mean) | 7.94 | 8.93 | mild -- more attacking support visible, as expected |
| 5 | `reception_opponents_visible` (mean, whole frame) | 7.67 | 6.23 | **inverse** direction to the radius-band features -- fewer opponents visible in total but more of them packed close to the receiver; consistent with final-third reception frames capturing a narrower, more compressed slice of the pitch, not a contradiction |
| 6 | `reception_frame_player_count` | 15.61 | 15.16 | no meaningful signal |

**Strongest overall candidates for a first modelling pass:** `is_cross`,
`pass_type_name` (Corner level), `is_through_ball`, `start_x`/`end_x`,
`play_pattern_name` (Counter/Corner levels), and, for CxA+,
`reception_nearest_opponent_distance_m` and `reception_opponents_within_5m` -- these
combine large effect size with large support, which distinguishes them from
high-lift-but-tiny-support levels like "No Touch" body part or "From Counter" play
pattern that deserve validation-split confirmation before being trusted (per the split
policy's train-only feature confirmation step).

## 6. Feature redundancy

`redundancy_event.json`, Pearson correlation across the full population:

| pair | r | reading |
|---|---|---|
| `start_x` <-> `end_x` | 0.7725 | moderately-high, expected (passes progress forward on average) but **not** a near-duplicate -- length/angle add real information beyond start position alone |
| `start_y` <-> `end_y` | 0.7391 | same pattern, y-axis |
| `pass_angle` <-> `start_y` | -0.4122 | moderate, geometric artefact of how `angle` is defined relative to pitch orientation -- not a redundancy to fix, an expected consequence of the StatsBomb angle convention |
| `pass_length` <-> `start_x` | -0.2930 | weak-moderate |
| `pass_length` <-> `end_x` | 0.1311 | weak |
| `pass_angle` <-> `end_y` | 0.0594 | weak |
| `start_x` <-> `start_y`, `end_x` <-> `end_y`, `pass_length` <-> `pass_angle` | ~0.01, 0.01, -0.01 | independent, as expected for orthogonal pitch dimensions and a length/angle polar decomposition |

**No pair among `pass_length`/`pass_angle`/`start_x`/`start_y`/`end_x`/`end_y` is a
near-duplicate (all \|r\| < 0.8).** But there is an *analytical* redundancy worth flagging
separately from the statistical one: `pass_length` and `pass_angle` are a deterministic
polar transform of `(start_x, start_y, end_x, end_y)` -- they carry no information a
model couldn't derive from the four raw coordinates. This is harmless for tree-based
models (which can split on derived features cheaply) but worth excluding one
representation for a linear/logistic baseline to avoid an ill-conditioned design
matrix.

**CxA+-specific finding** (`redundancy_plus_receiver_vs_end.json`): `receiver_x`/
`receiver_y` (the 360-tracked reception position) correlate with `end_x`/`end_y` (the
pass's own recorded end location) at **r = 0.9999** -- effectively identical.
**Recommendation: drop `receiver_x`/`receiver_y` as a near-duplicate of `end_x`/
`end_y`** for modelling (keep `end_x`/`end_y` since it exists in both tracks and keeps
the two tracks' baseline feature set consistent); `receiver_x`/`receiver_y` add no
information `end_x`/`end_y` doesn't already carry, and carrying both would inflate
apparent feature importance for what is really one signal.

## 7. Slice stability

**CxA event-only, by competition** (`slice_stability_event.json`): the `start_x` gap
(create vs no-create) is 35.8/39.0/38.1 units across Premier League/World Cup/Euro
respectively; the `is_cross` lift is 8.7x/12.3x/11.2x; the `is_through_ball` lift is
12.9x/12.4x/11.9x. All three signals hold in the same direction, at similar magnitude,
in every competition -- **none of the strongest event-only signals is being driven by a
single competition.**

**CxA+, by competition** (`slice_stability_plus.json`, World Cup and Euro only --
Premier League has no CxA+ rows, section 3): `reception_nearest_opponent_distance_m`
gap is 4.03m (World Cup: 7.59 -> 3.56) vs 3.51m (Euro: 7.27 -> 3.76); `opponents_within_5m`
lift is 3.7x (World Cup) vs 3.5x (Euro). Consistent across both available competitions.
**Caveat carried over from section 3: this stability claim only covers international
tournament football** -- there is no data to check whether it holds in league play,
because CxA+ contains none.

## 8. Leakage audit (independently re-verified, not assumed from the pipeline)

**Causal boundary.** Every event-only candidate feature (`pass_length`, `pass_angle`,
`start_x`/`start_y`, `end_x`/`end_y`, `pass_height_name`, `pass_type_name`,
`pass_technique_name`, `pass_body_part_name`, `is_through_ball`/`is_switch`/
`is_cross`/`is_cut_back`, `minute`/`second`/`possession_id`/`play_pattern_name`) is an
attribute of the pass event itself, sourced directly from `oam_core.passes`/`events`
columns that describe the pass action as executed -- none references a later event.
For CxA+, the 360 features (`reception_frame_player_count`,
`reception_nearest_opponent_distance_m`, `reception_opponents_within_5m`/`_8m`,
`reception_opponents_visible`, `reception_teammates_visible`) are joined on
`(match_id, event_uuid = receipt_event_id)` -- the frame captured *at* the resolved
Ball Receipt event, confirmed by the pipeline's own leakage spot-check
(`validate_cxa_training_matrix.py`, 200/200 sampled receipts real and causally after
the pass) and not re-litigated here.

**New finding from this audit: `pass_outcome_name`/`is_completed` must be explicitly
excluded from any candidate feature list**, and this is not obvious from the causal
boundary alone. `leakage_is_completed_vs_ycreate.json`:

| `is_completed` | rows | `y_create` positives | create rate |
|---|---|---|---|
| TRUE | 480,053 | 11,302 | 2.3543% |
| FALSE | 128,669 | 1 (the section-2 anomaly) | 0.0008% |

A 2,943x rate difference. `is_completed` is not *post-reception* information in the
literal timestamp sense -- it is determined by whether the pass reaches the intended
receiver at all, which is the same moment the reception happens, not a later one. But
functionally it behaves as a near-perfect proxy for `y_create` (only 1 exception in
608,722 rows, explained in section 2) rather than a genuine predictive signal about
*which completed passes* create chances -- the actual modelling question. Training on
it would let a model achieve high apparent performance by re-deriving "this basically
has to be a completed pass" rather than learning real chance-creation geometry.
**`pass_outcome_name` and `is_completed` are reference/diagnostic columns, kept in the
training matrix for label construction and this kind of audit, and must not enter the
candidate feature list.**

**Identity columns.** `passer_team_id` and `passer_player_id` are the only identity
columns present in either training matrix (confirmed by listing every column in both
`CREATE OR REPLACE TABLE` statements in
`scripts/materialize_cxa_event_v1_training_matrix.py` /
`materialize_cxa_plus_v1_training_matrix.py` -- no `recipient_id`/`recipient_name` or
receiving-team column was ever selected into either matrix, so there is nothing to
exclude there beyond what the pipeline already omitted). Per the methodology
(fold-safe nuisance attributes only, never published model features -- the same rule
CxG applied to shooter/defending-team identity), **`passer_team_id` and
`passer_player_id` must be excluded from any candidate feature list.** They are
retained in the matrix only as metadata (useful for grouping/debugging, e.g. this
audit's slice checks), exactly as documented in
`docs/cxa_data_feasibility_audit.md`, point 6.

**Result: leakage audit passes**, with one addition to the exclusion list beyond what
the original feasibility audit called out (`pass_outcome_name`/`is_completed`).

## 9. Modelling recommendations

This section makes no modelling decision -- it hands off an evidence-based punch list
for review.

**Strongest candidates (large effect size + large support, hold across slices):**
- `is_cross`, `is_through_ball` (event-only, both tracks)
- `pass_type_name` with `Corner`/`Free Kick` pooled as distinct levels, rest as
  `(null=open play)`
- `start_x`, `end_x` (keep both -- redundant-but-not-duplicate per section 6; consider
  dropping `pass_length`/`pass_angle` for a linear baseline only, keep for tree models)
- `play_pattern_name`, especially the `From Counter` / `From Corner` levels
- CxA+ only: `reception_nearest_opponent_distance_m`, `reception_opponents_within_5m`

**Worth including but validate on the validation split before trusting (elevated
lift, smaller support):** `is_cut_back`, `pass_technique_name` (Outswinging/Straight
levels), `pass_body_part_name = No Touch`, `is_switch`.

**Exclude from candidate features entirely:**
- `pass_outcome_name`, `is_completed` -- near-perfect target proxy, not a causal
  signal (section 8).
- `passer_team_id`, `passer_player_id` -- identity/nuisance columns per methodology
  (section 8).
- `receiver_x`, `receiver_y` (CxA+) -- near-duplicate of `end_x`/`end_y`, r=0.9999
  (section 6).
- `pass_event_id`, `match_id`, `competition_id`, `season_id`, `receipt_event_id`,
  `data_version`, `silver_schema_version`, `feature_version`, `materialized_at` --
  identifiers/provenance metadata, not features (some of these, e.g. `match_id`, are
  the correct *grouping* key for the split, never a model input).

**Needs more work before modelling, not resolved by this analysis:**
- 154 CxA+ rows (0.12%) with no resolvable 360 geometry (section 4) -- decide
  drop-vs-impute-with-flag.
- CxA+'s complete absence of Premier League rows (section 3) means any CxA+ model's
  generalization claim is scoped to tournament football only -- this should be stated
  explicitly wherever a CxA+ model is documented or deployed, not discovered later.
- `pass_body_part_name = No Touch` and `play_pattern_name = From Counter` show strong
  lift on small support (521 and 3,939 rows respectively) -- confirm these hold on the
  validation split before treating them as confirmed signal, per the split policy's
  train-only feature confirmation step.

**Not evaluated here, left to the modelling stage per the split policy:** pairwise/
interaction effects between features, calibration, and any comparison against a
baseline model. This analysis is univariate/bivariate by design, matching the "real
analysis, not a model" scope of this task.
