# CxA P_convert Data Feasibility Audit

Date: 2026-09-20
Scope: P_convert stage only (CxA = P_create x P_convert; P_create is complete and
merged to main, see `docs/analysis/cxa_p_create_test_eval_v1.md`). P_convert = P(Y_goal
| Y_create = TRUE) -- given a pass has been correctly identified as a chance-creating
pass, does the resulting shot become a goal.

All numbers below were queried live against `oam_core` / `oam_features` /
`oam_analysis` on 2026-09-20 (read-only), reproducible via the inline SQL in each
section. None are assumed or re-derived from memory. Per the same rule P_create
locked: `WHERE silver_schema_version = 'statsbomb_silver_v1_2'` on every `oam_core`
query, no exceptions -- the 3x row-duplication artifact P_create's audit documented
still applies to every `oam_core` table, including `shots` and
`shot_freeze_frame_players`.

## 1. Population definition and the key_pass_id -> shot join

Base population: every row where `y_create = TRUE` in
`oam_features.cxa_event_v1_training_matrix` / `cxa_plus_v1_training_matrix` -- i.e.
every pass P_create's own audit already validated as a `shots.key_pass_id` match (not
`passes.shot_assist`, which P_create's audit confirmed undercounts by ~9%). This
population is not re-derived here; it is read directly from the already-frozen
P_create tables.

**The key_pass_id -> shot join is confirmed 1:1, both directions, with zero
exceptions:**

```sql
-- event-only: 11,303 Y_create=TRUE passes
SELECT COUNT(*) total, COUNT(DISTINCT pass_event_id) distinct_passes,
       COUNTIF(shot_event_id IS NULL) unmatched, COUNT(DISTINCT shot_event_id) distinct_shots
FROM (creates LEFT JOIN shots ON shots.key_pass_id = creates.pass_event_id)
-- result: total=11303, distinct_passes=11303, unmatched=0, distinct_shots=11303
```

Independently confirmed no `key_pass_id` value is shared by more than one shot
(`GROUP BY key_pass_id HAVING COUNT(*) > 1` on the full `shots` table returns zero
rows). This was checked, not assumed, per the task's explicit instruction -- a
many-to-one or one-to-many join here would have silently corrupted the training
matrix grain.

## 2. Target: `Y_goal`, confirmed field and values

`shots.outcome_name` is confirmed (queried, not assumed) to have exactly 8 distinct
values on the full `oam_core.shots` corpus:

| outcome_id | outcome_name | rows (full shots corpus) |
|---|---|---|
| 96 | Blocked | 4,439 |
| 97 | **Goal** | 1,647 |
| 98 | Off T | 5,030 |
| 99 | Post | 286 |
| 100 | Saved | 3,495 |
| 101 | Wayward | 747 |
| 115 | Saved Off Target | 57 |
| 116 | Saved to Post | 36 |

**`Y_goal = (shots.outcome_name = 'Goal')`** -- the same field P_create's own
`y_goal` column already used (P_create's `y_goal` was illustrative/secondary at that
stage; this stage promotes it to the primary target, unchanged in definition).

## 3. Row counts and class balance -- checked, not assumed

| track | population (Y_create=TRUE rows) | Y_goal=TRUE | goal-conversion rate |
|---|---|---|---|
| event-only | 11,303 | 1,038 | **9.183%** |
| CxA+ | 2,830 | 263 | **9.293%** |

As expected and confirmed: **far less rare than P_create's own ~1.9-2.1% `Y_create`
rate** -- roughly a 1-in-11 event instead of a 1-in-50 event. The two tracks' rates
are close to each other (9.18% vs 9.29%), unlike P_create's own event-vs-plus gap
(1.857% vs 2.126%), so P_convert's population-composition difference between tracks is
smaller than P_create's was -- worth noting, not something to assume holds at the
split level (see the split policy doc for per-split rates, which do show more
spread on this much smaller population).

## 4. Candidate features

Two families: features describing the resulting **shot itself**, and **carried-over
context from the creating pass** (the same feature families P_create's own audit
enumerated, read directly from the existing P_create training matrices, not
re-derived).

### 4a. Carried-over pass context (from P_create, unchanged)

`start_x`, `start_y`, `end_x`, `end_y` (renamed `pass_end_x`/`pass_end_y` in the
P_convert matrix to avoid colliding with the shot's own `end_x`/`end_y`),
`pass_length`, `pass_angle`, `pass_height_name`, `pass_type_name`,
`pass_technique_name`, `pass_body_part_name`, `is_through_ball`, `is_switch`,
`is_cross`, `is_cut_back`, `play_pattern_name`, `minute`, `second`, `possession_id`,
`period`. CxA+ additionally carries `receipt_event_id`,
`reception_frame_player_count`, `receiver_x`, `receiver_y`,
`reception_nearest_opponent_distance_m`, `reception_opponents_within_5m`,
`reception_opponents_within_8m`, `reception_opponents_visible`,
`reception_teammates_visible` -- all already computed by P_create's pipeline, carried
unchanged.

`passer_team_id`/`passer_player_id` are carried as metadata only, not candidate
features, same rule as P_create (fold-safe nuisance identity). `pass_outcome_name`/
`is_completed` are also carried as metadata only, not candidates -- P_create's own
pre-model analysis found these to be a near-perfect target proxy for `Y_create`; for
this population every row already has `Y_create = TRUE`, so `is_completed` is even
closer to a constant here (11,302/11,303 rows are `TRUE`, the one exception being the
same anomaly row P_create's pre-model analysis already investigated and explained) --
functionally useless as well as leakage-adjacent for `Y_create`, no reason to promote
it as a `Y_goal` candidate either.

### 4b. Shot's own attributes (`oam_core.shots`, confirmed live for the 11,303-row
event-only population)

| candidate | description | coverage |
|---|---|---|
| `shot_x_sb`, `shot_y_sb` (`location_x`/`location_y`) | shot origin on the pitch | 0 null |
| `shot_end_x`, `shot_end_y` (`end_x`/`end_y`) | where the shot was aimed/ended up (x/y) | 0 null |
| `shot_end_z` (`end_z`) | shot height at the goal line | **3,634/11,303 (32.2%) null** -- flagged, not silently assumed complete; StatsBomb only populates height for a subset of shots (likely those with full 3D tracking, e.g. more heavily-covered competitions/venues). A real missingness pattern to handle at feature-promotion time, not an error. |
| `shot_body_part_name` | Right Foot (5,627) / Left Foot (3,161) / Head (2,476) / Other (39) | 0 null |
| `shot_technique_name` | Normal (9,483) / Half Volley (1,030) / Volley (532) / Diving Header (77) / Lob (72) / Overhead Kick (59) / Backheel (50) | 0 null |
| `shot_type_name` | **constant: 100% `Open Play`** for this population -- flagged as zero-variance, not a useful feature here, not an error. Makes sense structurally: a shot only has a `key_pass_id` (this population's defining join) if it followed an open-play pass; penalties/free-kick-direct-shots never carry a `key_pass_id`. |
| `statsbomb_xg` | StatsBomb's own precomputed shot-quality model output | 0 null. **Flagged as reference/benchmark only**, same treatment CxG gave it -- this is another model's prediction of almost exactly what P_convert is trying to predict; listed as a candidate per the task's "don't pre-filter" instruction, but any future promotion decision should treat it as a comparison baseline, not an ordinary input, exactly as CxG's baseline analysis did. |
| `first_time` | shot struck without a controlling touch | 2,942/11,303 (26.0%) true (StatsBomb null=false convention, same as P_create's pass boolean flags -- confirmed by checking, not assumed) |
| `aerial_won` | shot followed winning an aerial duel | 1,389/11,303 (12.29%) true |
| `follows_dribble` | shot followed a dribble | 7/11,303 (0.06%) true -- extremely rare, flagged now for the feature-promotion stage the same way P_create flagged `pass_technique_name=Straight` |
| `open_goal` | goal was open (no keeper/defenders blocking) at the moment of the shot | 96/11,303 (0.85%) true |
| `one_on_one` | shooter one-on-one with the keeper | 553/11,303 (4.89%) true |
| `under_pressure`, `counterpress` (from `oam_core.events`, joined on the shot's own `event_id`) | defensive pressure on the shooter at the moment of the shot | available, same convention as CxG's own event-context features |
| shot's own `minute`/`second` (from `events`) | phase timing at the shot (near-identical to the creating pass's minute/second, carried separately for completeness) | available |

**Excluded from the candidate list up front -- confirmed genuine outcome-leakage
fields, same discipline as P_create's `is_completed`/`pass_outcome_name` exclusion,
but caught here at the audit stage rather than a later pre-model stage since these are
unambiguous on inspection, not something requiring data investigation to discover:**

- `outcome_id`, `outcome_name` -- the label source itself, obviously never a feature.
- `deflected` (164/11,303, 1.45% true) -- describes an event during the shot's flight
  (a defender touching the ball en route) that directly determines/alters the
  outcome. This is not known before the shot is struck; it is part of what happens to
  the shot, the same category of leakage as a "shot was saved" field.
- `saved_off_target` (44/11,303, 0.39% true) and `saved_to_post` (24/11,303, 0.21%
  true) -- both literally describe the shot's own save/outcome characteristics. These
  are the exact "shot was saved" / "shot on target"-style fields the task named as the
  expected leakage risk. Excluded.

### 4c. Shot freeze-frame -- GK/defender positioning at the moment of the shot

**A finding that changes the task's initial framing, surfaced by checking rather than
assuming:** the task described GK positioning as available "if 360 data available for
CxA+." Live investigation found a **separate, distinct StatsBomb structure** --
`oam_core.shot_freeze_frame_players` -- that is **not** the same table as the
`three_sixty_frames`/`three_sixty_players` pair P_create's CxA+ track used for
reception-time context. This is StatsBomb's long-standing "shot freeze frame" feature
(present in their open data well before the newer "360" product existed), and its
coverage for this population is **near-universal, not CxA+-exclusive**:

- **11,303/11,303 (100%) of event-only Y_create=TRUE shots have shot freeze-frame
  data** (146,913 total player rows across them, ~13 players per frame on average).
- **99.95% (11,297/11,303) have exactly one identifiable opposing goalkeeper**
  (`teammate = FALSE AND position_name = 'Goalkeeper'`) -- checked for zero-or-multiple
  cases explicitly: 6 shots have zero identifiable opposing keepers (plausible
  open-goal/edge-case frames), **none** have more than one. No `x`/`y`/`teammate`
  nulls anywhere in the 146,913 player rows for this population.

**Because coverage is effectively universal for the whole Y_create=TRUE population
(not restricted to the CxA+ 360-at-reception subset), shot freeze-frame candidate
features are included in BOTH tracks' training matrices, not only CxA+** -- a
deliberate deviation from the task's initial framing, made because the live data does
not support restricting this family to CxA+, and stated here explicitly rather than
silently expanding scope. Candidate features (mirroring the geometry conventions
P_create's CxA+ reception-pressure features already established, for consistency
across this project -- approximate-metres distance bridge, radius-band opponent
counts):

| candidate | description |
|---|---|
| `shot_gk_x`, `shot_gk_y` | the identified opposing goalkeeper's tracked position at the shot (null for the 6 shots with no identifiable keeper) |
| `shot_gk_distance_m` | approximate-metres distance from the shot location to the GK |
| `shot_defenders_within_5m`, `shot_defenders_within_8m` | count of non-teammate outfield players within radius bands of the shot location (same convention as P_create's `reception_opponents_within_5m`/`_8m`) |
| `shot_defenders_visible` | total non-teammate players in the freeze frame |
| `shot_frame_player_count` | total players in the freeze frame (teammates + opponents) |

CxA+ retains its own separate, already-existing reception-time 360 context (section
4a) in addition to these shot-time freeze-frame features -- the two are complementary
(different moments: reception vs. the shot itself), not redundant, and both are listed
unfiltered per the task's instruction not to pre-filter.

## 5. Split coverage -- verified, not assumed

Every `Y_create = TRUE` row's `match_id` is present in
`oam_analysis.cxa_match_splits_v1`, confirmed live for both tracks:

| track | Y_create=TRUE rows | rows with no split match | distinct matches covered |
|---|---|---|---|
| event | 11,303 | **0** | 610 (all of them) |
| plus | 2,830 | **0** | 166 (all of them) |

This is expected -- the population is a strict subset of the already-split P_create
population -- but was verified by an explicit `LEFT JOIN ... WHERE match_id IS NULL`
check rather than assumed from that logical argument alone, per the task's
instruction. Zero unmatched rows in both tracks, across every match in both
populations (chance-creating-and-converting-or-not passes occur in literally every
match in the corpus, none had zero).

## 6. Validation performed on the materialized matrices

See [`scripts/validate_cxconvert_training_matrix.py`](../scripts/validate_cxconvert_training_matrix.py):

- Row-count sanity: re-counts `Y_create = TRUE` rows in the P_create matrices and
  asserts the P_convert matrix has the exact same row count and grain (no fan-out from
  the shot or freeze-frame joins).
- 1:1 join re-check: independently re-verifies every `pass_event_id` in the P_convert
  matrix resolves to exactly one `shot_event_id`, and vice versa.
- Split coverage re-check: re-confirms every row has a non-null `split` value.

## 7. No modelling in this stage

This audit and the pipeline it gates (see
[`docs/cxa_convert_split_policy_and_plan.md`](cxa_convert_split_policy_and_plan.md))
stop after materializing and validating
`oam_features.cxconvert_event_v1_training_matrix` /
`cxconvert_plus_v1_training_matrix`. No feature promotion, no pre-model signal
analysis, no candidate model -- all separate, later tasks, mirroring P_create's own
staged sequence exactly.
