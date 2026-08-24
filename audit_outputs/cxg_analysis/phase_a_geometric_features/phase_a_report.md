# Phase A — Geometric/Categorical Defender Features (CxG+ only)

New candidate features for the CxG+ track. **Not yet qualified** by the governed
univariate/correlation/PCA/bivariate chain -- that is an explicit, separate follow-up task
that will process this phase's output alongside Phase B (defender-style archetype
clustering, built independently in a parallel session and not touched here).

Run artifacts:
- `src/opponent_adjusted/analysis/odi/contracts.py` -- extended (additively) with
  `bucket_defender_role` and its marker constants, right next to the existing
  `BACKLINE_POSITION_MARKERS` it reuses the convention from. The existing
  `BACKLINE_POSITION_MARKERS` value and `_is_backline` function are untouched.
- `src/opponent_adjusted/features/cxg/phase_a_geometric.py` -- pure, testable computation
  logic (role centroids, nearest-defender ranking, per-shot feature assembly).
- `scripts/materialize_cxg_phase_a_geometric_features.py` -- orchestration: additive
  `ALTER TABLE ADD COLUMN IF NOT EXISTS` + scoped `UPDATE ... FROM` a staging table, never a
  `CREATE OR REPLACE` of the target table.
- `tests/features/cxg/test_phase_a_geometric.py` -- 10 tests, including role bucketing
  against every observed `position_name`, weighted-centroid math, distance ranking, the
  null-handling edge case, and (critically) the coordinate-frame flip described below.

---

## Step 1: investigation findings

### 1. Coordinate normalization

**No flip is applied anywhere in the existing codebase for raw event/freeze-frame
coordinates, and none was needed for the two reference features checked:**

- `goalward_progress_sb` (`src/opponent_adjusted/features/cxg/event_context.py:123-124,
  166-168`) hard-codes the goal at `(120, 40)` and uses `event.location_x/y` raw.
- `defensive_line_depth` (`src/opponent_adjusted/features/cxg/three_sixty_geometry.py:119-120`)
  likewise assumes goal-at-120 orientation on freeze-frame `x` raw.

The module docstring in `three_sixty_frame.py` confirms this is the established, deliberate
convention: StatsBomb's own per-event coordinate normalization already expresses every
event's `location_x/y` relative to that event's own acting team's attacking direction (goal
always at x=120 in that event's own frame) -- no period/team-based flip is ever added on top
of it anywhere in this repo. **Reused unchanged** for the role-centroid computation (Step 2):
a single dataset-wide aggregation from `oam_core.events.location_x/y`, not split by
period/team.

**However, this investigation surfaced a real, previously-undocumented subtlety that
directly caused a bug in this task's own first draft (see the EDA section below for how it
was caught and fixed):** `shot_freeze_frame_players.x/y` is expressed in the *shooting*
team's attacking frame (also goal-at-120, per `three_sixty_frame.py`'s documented
convention -- confirmed correct and unchanged). But a *defender's* role centroid (computed
from `oam_core.events`) is naturally expressed in *their own* team's attacking frame. For a
player defending a shot, those two frames are 180-degree opposite orientations of each
other -- their own goal sits at x=120 in the shot's frame, but at x=0 in their own team's
frame. Comparing a freeze-frame defender coordinate directly to a same-named-role centroid
without reconciling these two frames silently produces a systematically wrong (much too
large) distance. This wasn't an issue for `goalward_progress_sb` or `defensive_line_depth`
because both stay entirely within a single shot's own frame; it only appears once a
feature (like this task's `nearest_defender_zone_displacement`) needs to compare a
shot-frame coordinate to a separately-computed, own-team-frame reference point. Fixed with a
rigid 180-degree reflection (`GOAL_X - x, 2*GOAL_CENTRE_Y - y`) applied to the freeze-frame
defender coordinate before the comparison -- see `_flip_to_own_attacking_frame` in
`phase_a_geometric.py` and the EDA section for the numeric evidence this was correct.

### 2. Duel sub-type availability (note for Phase B)

**Not currently modeled in Bronze/Silver.** Searched `src/opponent_adjusted/pipelines/silver/contracts.py`
fully (every table contract) and grepped the repo case-insensitively for `duel` -- zero
matches. There is no `duels` table contract and no duel-outcome/duel-type columns anywhere.
The closest existing signal is a plain `aerial_won: bool` flag already present on both
`shots` and `passes` (not a duel-type/outcome breakdown, and not itself a duel event). If
Phase B's archetype clustering wants aerial-vs-ground or won-vs-lost duel sub-types, that
data is not sourced from raw StatsBomb `duel` events anywhere in this pipeline yet and would
need new Bronze/Silver work -- flagging this for the parallel session, not acting on it here.

---

## Step 2: the four features

All four computed only for the CxG+ (360-eligible) cohort -- `cxg_shot_base_features.has_360_frame`,
the same population as every other `defensive_360`/`line_shape_360` feature (3,960 shots).

### `nearest_defender_role` / `second_nearest_defender_role` (categorical)

`bucket_defender_role(position_name)` (new function, `analysis/odi/contracts.py`) buckets
StatsBomb's raw `position_name` into 5 roles: `GK`, `CB`, `Fullback_WingBack`, `Midfield`,
`Attack`. Extends the exact `BACKLINE_POSITION_MARKERS` substring-matching convention
(case-sensitive, ordered) rather than a new classification scheme:

1. Exact match `"Goalkeeper"` -> `GK`
2. Contains `"Center Back"` -> `CB` (checked before the generic Back rule below, or every
   center-back would misclassify)
3. Contains `"Back"` -> `Fullback_WingBack` (catches Left/Right Back and Left/Right Wing Back)
4. Contains `"Midfield"` -> `Midfield`
5. Else -> `Attack`

Verified against all 25 distinct `position_name` values actually present in
`oam_core.shot_freeze_frame_players` -- every one falls cleanly into exactly one bucket
(test: `test_bucket_defender_role_every_observed_position_name`).

Defender identification/ranking reuses `shot_freeze_frame_players` (`teammate = FALSE`,
non-null coordinates) -- the same source table the ODI script already uses for defender
identification, per the task's explicit "don't rebuild from scratch" instruction. No
existing 1st/2nd-nearest ranking mechanism was found anywhere in the repo (confirmed during
Step 1 -- `nearest_defender_distance_m` is a plain `min()` reduction with no ordering
retained), so ranking by distance is new logic here, added as `nearest_defenders()` in
`phase_a_geometric.py`. The distance metric itself reuses `metre_distance`
(`three_sixty_frame.py`) unchanged -- not a new formula.

### `nearest_defender_zone_displacement` (continuous, metres)

Each role bucket's centroid = the true dataset-wide **weighted** mean `location_x/y` across
every `oam_core.events` row with that role (weighted by event count per underlying
`position_name`, not a naive average of per-position-name means -- verified by
`test_role_centroids_weighted_not_naive_mean`). The nearest defender's shot-frame coordinate
is flipped into their own attacking frame (see Step 1 above) before computing
`metre_distance` to their role's centroid.

### `nearest_defender_gap` (continuous, metres)

`metre_distance` between the 1st- and 2nd-nearest defenders' shot-frame coordinates
directly -- no flip needed (a rigid reflection is a distance-preserving isometry, so
flipping both points would leave their pairwise distance unchanged; proven in
`test_flip_to_own_attacking_frame_is_180_degree_reflection` and left out of the gap
calculation deliberately, documented in code).

### Edge case: fewer than 2 identifiable defenders

`second_nearest_defender_role` and `nearest_defender_gap` are `NULL` (never fabricated) when
fewer than 2 defenders are found in the freeze frame, with the reason recorded explicitly in
a new `nearest_defender_rank_null_reason` column (`fewer_than_2_defenders_identified`) --
following the same null-reason-column convention already established in
`cxg_odi_features_v1`. When 0 defenders are found, `nearest_defender_role`/
`nearest_defender_zone_displacement` are naturally `NULL` too (nothing to report), covered
by the same reason value.

---

## Step 3: where these landed, and why

All 6 new columns (4 features + `nearest_defender_rank_null_reason` +
`phase_a_materialized_at` lineage timestamp) were added to **`oam_features.cxg_defensive_360_features`**,
not `cxg_line_shape_360_features`. Justification: these are per-defender-**identity**
features (which specific defender, their role, their distance to a teammate) -- the same
unit of analysis as that table's existing `nearest_defender_distance`/
`nearest_defender_distance_delta`. `cxg_line_shape_360_features` holds team-**shape**
features (defensive line depth/width/hull area) -- a different, aggregate unit of analysis
that doesn't fit a single-defender-identity column set.

**Materialization mechanism, deliberately additive:** `ALTER TABLE ... ADD COLUMN IF NOT
EXISTS` (idempotent) followed by a scoped `UPDATE ... FROM` a staging table keyed on
`event_id` -- not a `CREATE OR REPLACE TABLE` rebuild. `cxg_defensive_360_features`'s
existing 33 columns are computed by the frozen F1-F14 JSON-blob pipeline in
`materialize_cxg_feature_family_tables.py`, which this task's script does not call or
modify.

**`cxg_training_matrix_v1` view refresh:** rather than editing the frozen
`_create_training_view` function, this task's script simply **calls the existing,
unmodified function again** after the column update. That function already introspects each
family table's live columns via `INFORMATION_SCHEMA` at call time (not a hard-coded column
list), so re-running it picks up the 6 new columns automatically -- confirmed live: the
view went from 165 to **171 columns**, all 6 new ones present, still 15,737 rows (the full
CxG population; CxG+ non-eligible rows correctly get `NULL` via the existing `LEFT JOIN`).

---

## Step 4: sanity EDA (not full qualification)

### A real bug caught and fixed during this step

The first materialization run (before the coordinate-frame flip described in Step 1) produced
`nearest_defender_zone_displacement` with **mean 44.4m, max 96.7m** -- on a 105m-long
(approximate-metres) pitch, that's "systematically huge," exactly the symptom this step's
sanity check is meant to catch. Root-caused to the shot-frame-vs-own-team-frame mismatch
described in Step 1, fixed with the 180-degree reflection, and re-run. Post-fix:

| Metric | Before fix | After fix |
|---|---|---|
| mean | 44.44m | 37.06m |
| median | 43.05m | 37.73m |
| max | 96.71m | 67.01m |
| min | 3.54m | 0.73m |

The post-fix **per-role breakdown is the strongest evidence the fix is correct** -- a clean,
monotonic, football-sensible pattern:

| `nearest_defender_role` | n | mean displacement (m) | median | min | max |
|---|---|---|---|---|---|
| `GK` | 97 | **6.10** | 6.51 | 0.73 | 10.13 |
| `CB` | 1,329 | 27.62 | 28.38 | 1.26 | 37.94 |
| `Midfield` | 1,263 | 40.69 | 40.83 | 3.11 | 56.32 |
| `Fullback_WingBack` | 779 | 43.60 | 44.31 | 2.72 | 54.24 |
| `Attack` | 399 | 51.73 | 51.86 | 26.48 | 67.01 |

Goalkeepers -- who play near their own goal essentially every minute of the match, so their
"typical position" and their "position when facing a shot" should almost coincide -- show a
tight, small displacement (mean 6.1m). The displacement grows monotonically the further
forward a role normally plays: an attacker being the *nearest* defender to a shot is
inherently an unusual, badly-out-of-position situation, so a large displacement there is a
meaningful signal, not an artifact. This pattern (independently, without being built for
this purpose) also cross-validates the fix: `GOAL_X - 9.55 = 110.45` (GK's own-frame
centroid, flipped) lines up almost exactly with a plausible average shot-frame GK depth,
matching how `gk_depth` is defined elsewhere in this codebase (`GOAL_X - gk.x`).

### Null rates

| Column | Null rate |
|---|---|
| `nearest_defender_role` | 2.35% (93 / 3,960) |
| `nearest_defender_zone_displacement` | 2.35% (93 / 3,960) |
| `second_nearest_defender_role` | 2.55% (101 / 3,960) |
| `nearest_defender_gap` | 2.55% (101 / 3,960) |

101 shots have fewer than 2 identifiable defenders (8 of those have exactly 1 -- `nearest_*`
populated, `second_*`/`gap` null; the other 93 have 0). All 101 carry
`nearest_defender_rank_null_reason = 'fewer_than_2_defenders_identified'`; the remaining
3,859 rows have `NULL` in that reason column (nothing to explain).

### Categorical distributions

`nearest_defender_role`: CB 1,329 (33.6%), Midfield 1,263 (31.9%), Fullback_WingBack 779
(19.7%), Attack 399 (10.1%), GK 97 (2.4%), null 93 (2.3%).

`second_nearest_defender_role`: Midfield 1,299 (32.8%), CB 1,272 (32.1%),
Fullback_WingBack 677 (17.1%), Attack 380 (9.6%), GK 231 (5.8%), null 101 (2.6%).

**Honest observation, not a bug:** CB is the single largest bucket for the nearest defender,
as expected, but Midfield is close behind at ~32% both times. This most likely reflects
StatsBomb's `position_name` being each player's *nominal formation slot* for the match
(fixed at lineup time, occasionally updated at substitution) rather than their exact role at
the instant of a specific defensive action -- a team defending a shot commonly has a
nominal central midfielder tracking back to help, which the taxonomy correctly still calls
"Midfield." This is exactly the kind of observation the follow-up qualification task should
take into account; not something to correct here.

### `nearest_defender_gap`

| Metric | Value (metres) |
|---|---|
| n (non-null) | 3,859 |
| min | 0.24 |
| median | 3.75 |
| mean | 4.19 |
| max | 19.11 |

Plausible range for the physical distance between two nearby defenders on a pitch -- no sign
of a coordinate-scale issue (this feature was never affected by the frame mismatch, since
both defenders share the same frame).

---

## Row-count reconciliation

| Check | Expected | Actual | Match |
|---|---|---|---|
| Cohort size (`has_360_frame`) | 3,960 | 3,960 | Yes |
| `cxg_defensive_360_features` total rows (unchanged by this additive update) | 3,960 | 3,960 | Yes |
| Rows with `nearest_defender_role` populated | 3,960 - (0-defender shots) | 3,867 | Consistent (93 null) |
| Rows with `second_nearest_defender_role`/`nearest_defender_gap` populated | 3,960 - (<2-defender shots) | 3,859 | Consistent (101 null) |
| `cxg_training_matrix_v1` column count | 165 + 6 | 171 | Yes |
| `cxg_training_matrix_v1` row count (unchanged, full CxG population) | 15,737 | 15,737 | Yes |

---

## Test suite

`python -m pytest -q` -> **278 passed**, no regressions. This task added 10 new tests
(`tests/features/cxg/test_phase_a_geometric.py`): role bucketing against every observed
`position_name`, weighted-vs-naive centroid math, distance-based ranking (including the
fewer-than-2-defenders edge case), the coordinate-frame flip's correctness (180-degree
self-inverse property, and a synthetic case proving the flip prevents the exact
"systematically huge displacement" bug this task's own EDA caught), and the full
`compute_shot_features` assembly for both the fully-populated and edge-case paths. The
overall test count includes additional tests landed by the parallel Phase B session
(`tests/analysis/defstyle/`) during this task's execution -- not authored here, not
modified here, confirmed not interfered with.

---

## What was explicitly NOT done (per task constraints)

- No full univariate/correlation/PCA/bivariate qualification run on these 4 features -- a
  separate follow-up task, to process this phase and Phase B together.
- Phase B (defender-style archetype clustering) was not touched, duplicated, or interfered
  with -- confirmed via the parallel test-count increase above being left entirely alone.
- No `player_id`/`team_id` exposed as an output column -- both used only as internal
  join/filter keys in the SQL (`shot_freeze_frame_players.teammate`, `event_id` joins); the
  6 new columns are role/categorical/continuous-geometric/lineage only.
- `statsbomb_xg` was not referenced anywhere in this task.
- No existing frozen feature code was modified -- `cxg_defensive_360_features`'s existing
  33 columns, the K-Means defensive-profile clustering, ODI, and every other existing 360
  feature are untouched; `_create_training_view` was called, not edited;
  `BACKLINE_POSITION_MARKERS`/`_is_backline` in `odi/contracts.py` are untouched (only new
  code was added alongside them).

---

## Summary for hand-off

Four new CxG+ candidate features, landed in `oam_features.cxg_defensive_360_features`
(additive ALTER + scoped UPDATE, 3,960 rows, ~2.3-2.6% null by design) and now visible in
`cxg_training_matrix_v1` (171 columns, 15,737 rows). A genuine coordinate-frame bug (shot's
attacking frame vs. defender's own-team attacking frame, opposite orientations) was found
and fixed during this task's own sanity EDA, evidenced by a clean, monotonic,
football-sensible displacement-by-role pattern (GK 6.1m << CB 27.6m < Midfield 40.7m <
Fullback/WingBack 43.6m < Attack 51.7m) that would not otherwise have emerged. Not yet
statistically qualified -- that is explicitly the next task's job.
