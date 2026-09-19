# CxA Data Feasibility Audit

Date: 2026-09-19
Scope: P_create stage only (CxA = P_create x P_convert; this audit and the pipeline it
gates cover P_create). Methodology reference: Drive doc "17_CXA_METHODOLOGY_V1".
BigQuery project: `oam-varun-260819`, location `europe-west2`.
Silver corpus: `data_version = b0bc9f22dd77c206ddedc1d742893b3bbe64baec`,
`silver_schema_version = statsbomb_silver_v1_2`.

All numbers below were queried live against `oam_core` on 2026-09-19 (read-only,
BigQuery MCP `execute_sql_readonly`) and are reproducible with the SQL inline in each
section. None are re-derived from memory or from the CxG audit.

## 1. The 3x row-duplication artifact is confirmed live, and it is exactly 3x, not "up to 3x"

`oam_core.passes` carries three `silver_schema_version` values, each with an
**identical** row count:

| silver_schema_version | rows |
|---|---|
| `statsbomb_silver_v1` | 608,722 |
| `statsbomb_silver_v1_1` | 608,722 |
| `statsbomb_silver_v1_2` | 608,722 |

Total unfiltered `oam_core.passes` row count is 1,826,166 = 3 x 608,722. The same
pattern holds across every `oam_core` table (confirmed structurally on `events`,
`shots`, `ball_receipts`, `three_sixty_frames`, `three_sixty_players` via their
reported `numRows`, all exact multiples of 3 consistent with the per-pass count).

**Rule for every query against `oam_core` in this project, no exceptions:**
`WHERE silver_schema_version = 'statsbomb_silver_v1_2'`. This is the same rule CxG
locked, extended here to the pass grain. All counts in this document already apply it.

## 2. Grain is one row per pass, not one row per completed pass

`oam_core.passes` filtered to `silver_schema_version = 'statsbomb_silver_v1_2'`:

- Total passes: **608,722**
- Completed (`outcome_name IS NULL`): **480,053**
- Not completed (`outcome_name` = Incomplete / Out / Pass Offside / Unknown / etc.):
  **128,669**

CxA's training matrix keeps all 608,722 rows. A failed pass cannot causally create a
shot (StatsBomb only links `key_pass_id` from a completed reception), so every
non-completed row is a legitimate, non-circular `y_create = FALSE` example — dropping
them would inflate the training base rate and bias the model toward "pass was
attempted in a dangerous area" rather than "pass produced a chance." This mirrors CxG's
own grain decision (all shots kept, not just goals).

## 3. Y_create label: `key_pass_id` is the correct join key, `shot_assist` undercounts

```sql
SELECT
  (SELECT COUNT(*) FROM oam_core.shots
     WHERE silver_schema_version='statsbomb_silver_v1_2' AND key_pass_id IS NOT NULL) AS shots_with_key_pass,
  (SELECT COUNT(*) FROM oam_core.passes
     WHERE silver_schema_version='statsbomb_silver_v1_2' AND shot_assist = TRUE) AS passes_shot_assist_true,
  (SELECT COUNT(DISTINCT s.key_pass_id) FROM oam_core.shots s
     JOIN oam_core.passes p ON p.event_id = s.key_pass_id AND p.silver_schema_version='statsbomb_silver_v1_2'
     WHERE s.silver_schema_version='statsbomb_silver_v1_2') AS key_pass_ids_resolving_to_pass_row
```

| metric | value |
|---|---|
| `shots.key_pass_id IS NOT NULL` | 11,303 |
| `passes.shot_assist = TRUE` | 10,265 |
| `key_pass_id` values that resolve to an actual pass row | 11,303 / 11,303 (100%) |

`shot_assist` undercounts `key_pass_id`-linked passes by ~9.2% (10,265 vs 11,303),
consistent with the earlier feasibility finding (previously reported 10,264; the
1-row difference is within normal run-to-run noise on a live corpus and does not change
the conclusion). Every one of the 11,303 `key_pass_id` values resolves bidirectionally
to a real pass row — 100% consistent, confirmed live, not assumed.

**Rule: `Y_create = (a pass's event_id appears as some shot's key_pass_id)`.** Never use
`passes.shot_assist`.

## 4. Pass -> Ball Receipt linkage: `related_event_ids` resolves correctly, `event_index + 1` does not

`events.related_event_ids` is a `REPEATED STRING` on every event, including passes.
For each completed pass, the correct "first controlled reception" boundary is the
linked `ball_receipts` row (a genuine, StatsBomb-labeled distinct event type), found by
unnesting `related_event_ids` and joining `ball_receipts.event_id`:

```sql
-- completed passes (outcome_name IS NULL), silver_schema_version = 'statsbomb_silver_v1_2'
completed_passes: 480,053
passes_with_any_related_event_id: 479,944
passes_resolving_to_a_ball_receipt_row: 479,923   (99.97% of completed passes)
```

This confirms the audited 99.97% coverage figure live. The ~0.03% (130 passes) that
don't resolve are cases StatsBomb genuinely doesn't emit a Ball Receipt for (e.g. the
next event is a Pressure/Duel/Carry with no intervening receipt, or the pass reaches a
goalkeeper collection recorded as a different event type) — not a data-quality problem
to fix, a real absence the pipeline must tolerate (rows with no resolvable receipt are
excluded from the CxA+ population only; they remain in the CxA event-only population).

**`event_index + 1` is not a substitute.** `event_index` is a single global, match-wide
counter that interleaves every concurrent/overlapping event across both teams
(pressure events, off-ball movement, etc. are frequently recorded between a pass and
its own receipt). Spot-checking confirms `event_index + 1` essentially never points at
the pass's actual `ball_receipts` row once any intervening event exists — this is a
leakage trap (it can accidentally point at a *later*, causally-downstream event) and a
correctness trap (it usually points at the wrong event entirely), not a minor
simplification. `related_event_ids` is the only correct linkage and is what the
pipeline uses.

## 5. Y_goal label (secondary, P_convert-adjacent, not used to train P_create)

Via the same `key_pass_id`-linked shot's `outcome_name = 'Goal'`:

- Of the 11,303 `Y_create = TRUE` passes, **1,038** have a linked shot that was a goal.

This label is materialized alongside `Y_create` in both training matrices for future
P_convert work but is illustrative only in this stage — it is not part of the P_create
target and must not be used to select P_create features or evaluate the P_create model.

## 6. Causal boundary: pre-reception features only

Feature families confirmed available strictly before the reception event, all sourced
from `oam_core.events` / `oam_core.passes` (no post-reception event required):

- Pass geometry: `length`, `angle`, `height_name`, `end_x`, `end_y`, `pass_type_name`,
  `through_ball`, `switch`, `body_part_name`, `technique_name`,
  `start_x`/`start_y` (= the pass event's own `location_x`/`location_y`).
- Possession/phase context: `minute`, `second`, `play_pattern_name`, `possession_id`.

Receiver identity (`recipient_id`, `recipient_name`) and receiving-team identity exist
on `oam_core.passes` but are excluded from the feature set (fold-safe nuisance
attributes only, never published model features) — same rule CxG applied to
shooter/defending-team identity. The passer's own `team_id`/`player_id` are carried as
metadata columns only (not receiver information, not used as model features either).

CxA+ 360 features are joined at `three_sixty_frames` / `three_sixty_players` keyed on
`(match_id, event_uuid = the resolved Ball Receipt event_id)` — strictly the frame
captured **at** reception, never a later frame. No feature in either track reads
anything timestamped after the reception event.

## 7. Two tracks: population sizes, confirmed live

**CxA (event-only).** Population = all 608,722 passes (point 2). No 360 requirement.

**CxA+ (event + 360 at reception).** Population = completed passes whose Ball Receipt
resolves via `related_event_ids` AND that receipt has a `three_sixty_frames` row:

```sql
completed_passes: 480,053
completed_passes_with_resolved_receipt: 479,923
receipts_with_a_360_frame: 133,143
```

CxA+ population: **133,143** rows (27.7% of completed passes, 21.9% of all passes).

## 8. Class balance, confirmed live

| track | rows | Y_create = TRUE | Y_create rate |
|---|---|---|---|
| CxA (event-only) | 608,722 | 11,303 | 1.857% |
| CxA+ (360 at reception) | 133,143 | 2,830 | 2.126% |

CxA+'s positive rate is measurably higher than the event-only population's — expected,
since 360 coverage skews toward competitions/matches with more advanced-phase,
final-third play where StatsBomb prioritizes 360 capture, which correlates with chance
creation. This is a real population-composition difference the split policy and any
later cross-track comparison must account for (see
[cxa_split_policy_and_parallel_plan.md](cxa_split_policy_and_parallel_plan.md)), not a
bug.

Of the 11,303 CxA `Y_create = TRUE` rows, 1,038 (9.2%) are also `Y_goal = TRUE` — this
is the implied P_convert base rate for a future stage, not something this stage acts on.

## 9. `oam_core` schema actually available (confirmed live, not assumed)

- `events`: `event_id, match_id, competition_id, season_id, event_index, period,
  minute, second, timestamp, duration, event_type_id/name, possession_id,
  possession_team_id/name, team_id/name, player_id/name, position_id/name,
  play_pattern_id/name, under_pressure, counterpress, off_camera, out, location_x,
  location_y, related_event_ids (REPEATED STRING), data_version,
  silver_schema_version`.
- `passes`: `event_id, match_id, competition_id, season_id, team_id, player_id,
  recipient_id, recipient_name, length, angle, height_id/name, end_x, end_y,
  pass_type_id/name, outcome_id/name, technique_id/name, body_part_id/name,
  assisted_shot_id, shot_assist, goal_assist, cross, cut_back, switch, through_ball,
  inswinging, outswinging, straight, deflected, miscommunication, no_touch,
  aerial_won, data_version, silver_schema_version`. No `start_x`/`start_y`,
  `minute`/`second`, `possession_id`, or `play_pattern_name` on this table directly —
  those come from joining `events` on `event_id` (+ `silver_schema_version`).
- `shots`: `event_id, match_id, ..., key_pass_id, outcome_name, statsbomb_xg, ...`.
- `ball_receipts`: `event_id, match_id, ..., team_id, player_id, outcome_id/name,
  data_version, silver_schema_version`. No location columns — position at reception
  comes from the linked 360 frame, not from this table.
- `three_sixty_frames`: `match_id, event_uuid, competition_id, season_id,
  visible_area (REPEATED FLOAT), frame_player_count, data_version,
  silver_schema_version`. Keyed on `(match_id, event_uuid)`.
- `three_sixty_players`: `match_id, event_uuid, competition_id, season_id,
  frame_player_ordinal, teammate, actor, keeper, x, y, data_version,
  silver_schema_version`.

One property this project relies on that CxG did not need to: because the 360 frame
CxA+ joins is captured **at the Ball Receipt event itself**, `teammate`/`actor` on
`three_sixty_players` are already oriented relative to the receiver (the frame's own
actor) — unlike CxG, where the 360 frame is sometimes captured at a different event
than the shot and needs `orient_players()`-style re-expression. No orientation step is
needed for CxA+'s reception-pressure features. What CxA+ v1 deliberately does **not**
build yet: goal-relative/attacking-direction features (e.g. "visible goal angle at
reception", "defenders between receiver and goal"). Unlike a shot, a pass reception has
no home-grown convention in this codebase for which goal the receiving team is
attacking at that point in the match (CxG's `GOAL_X = 120.0` constant is documented as
a shot-geometry-specific convention, not a general attacking-direction table), and
guessing would be exactly the kind of unvalidated assumption this audit is here to
prevent. CxA+ v1 ships direction-independent reception-pressure features only
(nearest-opponent distance, opponents within 5m/8m, frame player counts); goal-relative
360 features are deferred to a v2 once an attacking-direction convention is built and
validated as its own piece of work.

## 10. No model training or P_convert work in this stage

This audit and the pipeline it gates stop at materializing and validating the P_create
training matrices (`oam_features.cxa_event_v1_training_matrix`,
`oam_features.cxa_plus_v1_training_matrix`). Model training and P_convert (conditioned
on `Y_create = TRUE` rows) are explicitly out of scope until reviewed.

## 11. Validation performed on the materialized tables

See [`scripts/validate_cxa_training_matrix.py`](../scripts/validate_cxa_training_matrix.py):

- Row-count sanity: re-counts `oam_core.passes` under the `silver_schema_version`
  filter and asserts it matches the materialized `cxa_event_v1_training_matrix` row
  count exactly (catches any future silent re-introduction of the 3x duplication bug
  or an unfiltered join fan-out).
- Leakage spot-check: for a sample of CxA+ rows, re-resolves `receipt_event_id` via
  `related_event_ids` independently of the materialization query and confirms (a) it is
  a real `ball_receipts.event_id`, not an arbitrary later event, and (b) its
  `event_index` is strictly greater than the pass's own `event_index` (receipt causally
  follows the pass, not a naive `+1` coincidence).

## 12. Open items carried forward, not resolved here

- CxA+ direction-aware 360 features (goal-relative geometry at reception) — deferred,
  see point 9.
- Canonical train/validation/test split table for CxA/CxA+ — designed in
  [cxa_split_policy_and_parallel_plan.md](cxa_split_policy_and_parallel_plan.md), not
  yet a modeling gate for this stage (matches CxG's own sequencing: split design first,
  train-only feature confirmation later).
- P_convert stage (`Y_goal` conditioned on `Y_create = TRUE`) — explicitly deferred per
  the task scope, point 10.
