# Phase C — Rolling-Window Defensive Features (v3, BOTH tracks)

Candidate features only (not yet statistically qualified — full univariate/correlation/PCA/
bivariate qualification is a separate follow-up task, same precedent as Phase A/B).

## Discrepancy flagged up front

The task brief cites `v2_feature_methodology_locked.md` as the source of the locked design
decisions. That file does not exist anywhere in the repo (confirmed via `find` for the exact
filename, `*methodology*`, and `*locked*` — no hits). This is noted as a real premise
mismatch rather than silently ignored. It was not blocking: the task's own inline text fully
specifies the locked decisions (both tracks, per-minute-of-window normalization, exponential
decay with a half-life to be justified in Step 1, global team_id + chronological-date
pooling), so implementation proceeded from that self-contained spec.

## Step 1 — Investigation findings

**Existing rolling-window pattern reused, not reinvented.** `_momentum()`
(`src/opponent_adjusted/features/cxg/event_context_extended.py:198-241`, frozen, not
modified) computes `territorial_dominance_last_5m` as `(team_mass - opp_mass) / total_mass`
over attacking-type events (`Pass`, `Carry`, `Dribble`, `Ball Receipt*`, `Shot`) inside a
strictly backward-looking `period`-scoped window (`0 <= shot_clock - event_clock_s(event) <=
WINDOW_S`, `WINDOW_S = 300.0`). No attack-direction flip is needed — StatsBomb events are
already normalized per-event to each acting team's own attacking direction (confirmed
previously in Phase A's docstring, re-confirmed here). Because that module is frozen, Phase
C's field-tilt extension (2b) duplicates this exact mass/ratio math in a new module
(`phase_c_rolling_window.py::territorial_dominance_extended`), parameterized by window size
instead of hardcoded to 300s.

**Defensive action-type set confirmed identical to Phase B's.**
`analysis/defstyle/features.py::ACTION_TYPES` = `Pressure`, `Duel`, `Interception`,
`Clearance`, `Block`, `Foul Committed`, `50/50` (7 values, imported unchanged — not
re-derived).

**Matches-per-team distribution** (live, 610 split-assigned matches, 74 teams): min = 3,
median = 8, max = 38. Bimodal: 13 teams sit at the floor (exactly 3 matches), 20 teams sit at
the ceiling (38 matches). No team has 1 or 2 matches — genuine cross-match cold start is
purely a team's literal first match in the dataset.

**Both-tracks time-column consistency confirmed.** `cxg_event_context_features` already
covers exactly the same 15,737-row / 610-match population as `cxg_shot_base_features` (both
tracks share one event-wide Gold table; CxG+ is a `has_360_frame` subquery of it, not a
separately-computed population). `minute`/`second`/`period` are 100%-populated across both
tracks (0 nulls, confirmed with `data_version`/`silver_schema_version` pinned). No
event-coverage gap found between the two tracks for this feature set.

**Cost discipline resolution.** `oam_core.events` is confirmed unpartitioned/unclustered
(no `timePartitioning`, no `clustering` metadata; 6,470,469 rows, ~2.29 GB) — a windowed-SQL
rewrite of the rolling-window logic would be an equally-expensive full scan with no cost
advantage over the established "fetch once, process per-match in Python" pattern. That
pattern (matching `_momentum()` and Phase B's `COUNTS_QUERY`) satisfies both the reuse
instruction and cost discipline simultaneously: a single lean events query (no
passes/carries/etc. join, since Phase C needs only
`period`/`minute`/`second`/`timestamp`/`event_type_name`/`team_id`/`location_x`/`location_y`)
scoped to the 610-match governed universe via `cxg_match_splits_v1`, pinning
`data_version`/`silver_schema_version` together throughout (this exact 3x-fanout risk from
`oam_core.events`'s 3 schema-version copies was independently re-confirmed live during
investigation on an unpinned test join: 47,211 rows instead of 15,737, exactly 3x).

**Half-life parameter: `CROSS_MATCH_HALF_LIFE_MATCHES = 3.0`**, justified against the live
matches-per-team distribution above:

- Matches the common football-analytics "recent form" window (3–5 games).
- For the 13 floor teams (only 3 total matches each), a half-life of 3 keeps all 3 matches
  contributing non-trivially to any later match (gap-1/2/3 weights ≈ 0.79 / 0.63 / 0.50)
  rather than collapsing to "basically just the last match," which a much shorter half-life
  (e.g. 1) would do.
- For the 20 ceiling teams (38 matches), matches more than ~15 games back decay below ~3%
  weight (`0.5^(15/3) = 0.031`), appropriately favouring recent defensive form for teams with
  a long observed run.

## Step 2 — Feature construction (both tracks)

All three features are computed ONCE per shot over the full 15,737-shot event-wide
population; CxG+ (3,960 shots) inherits them automatically as the `has_360_frame` subset of
the same table — no separate per-track computation was needed since none of these features
depend on 360 freeze-frame data.

**2a. `defensive_action_rate_{15,30,45,60}m`** — per-minute rate of the DEFENDING team's
defensive actions (the 7-type set above) in trailing windows before the shot, strictly
backward-looking. Defending team resolved via a single batched join to
`oam_core.matches.home_team_id`/`away_team_id` (whichever differs from the shot's own
`team_id`), not a per-match query.

Denominator convention: `elapsed_minutes = min(window_minutes, actual_elapsed_since_period_
start) / 60`, where `actual_elapsed_since_period_start` is the shot's clock minus the
empirically-derived period-start clock (the minimum observed `event_clock_s` among that
period's prior events — not an assumed fixed StatsBomb boundary). This makes a partial-window
rate mathematically unbiased (dividing by time actually observed) rather than the
"truncated/biased rate silently presented as a full window" anti-pattern the task
warns against. The only genuine null is `elapsed_minutes <= 0` (reason
`zero_elapsed_time_in_period`) — a shot literally at the first recorded instant of a period.
Live: 0 of 15,737 shots hit this in either track (no shot in the corpus is the literal first
event of its period), so every shot has a real (possibly partial-window) rate — no shot
silently loses this feature to cold start.

**2b. `territorial_dominance_last_15m`** — `_momentum()`'s exact mass/ratio math, window
extended from 300s to 900s (15 minutes). Same normalization, same attack-direction handling
(none needed), same team-vs-opposition split. Null only when `total_mass == 0` (no attacking
events observed in the window) or `shot_clock` is unavailable — the same null condition the
original 5-minute feature already has, just measured at 16/15,737 shots for the extended
15-minute window (vs. presumably fewer at 5 minutes, since a longer window has more
opportunity to observe at least one attacking event).

**2c. `cross_match_defensive_rate`** — two-stage, single-query-per-stage design (no
per-team/per-match Python loop):

1. Per-(team_id, match_id) whole-match defensive-action rate: total defensive-action count
   (both periods) divided by observed match duration in minutes (per-period `max(clock) -
   min(clock)`, summed across periods) — 1,220 team-match pairs (610 matches × 2 teams),
   computed from the same single event fetch as 2a/2b, no extra query.
2. For each shot, the defending team's PRIOR matches (chronological by `match_date` +
   `kick_off`, pooled globally by `team_id` — NOT scoped to competition/season, per the
   locked decision) are combined via `cross_match_rolling_rate`: exponential-decay weighted
   average, `weight(gap) = 0.5^(gap / 3.0)`, gap = matches back from the current one.
   Cold start (a team's literal first match, zero prior matches) is an explicit null, reason
   `team_first_match_in_dataset` — never a silent zero/impute.

## Step 3 — Materialized location

Landed additively on `oam_features.cxg_event_context_features` (ALTER TABLE ADD COLUMN IF
NOT EXISTS + scoped `UPDATE ... FROM` a staging table keyed on `event_id`, never `CREATE OR
REPLACE`) — the same precedent as Phase A's `cxg_defensive_360_features` extension. Chosen
over a `*_360_features` table because these features are event-log-based and apply to the
FULL 15,737-shot event-wide population, not just the 3,960-shot 360-eligible subset; landing
in `event_context` keeps the family taxonomy honest (per-defender/360-identity features stay
in the `_360` tables, event-log features stay in `event_context`).

`oam_features.cxg_training_matrix_v1` (the Gold VIEW joining all family tables) was
refreshed by re-calling the existing, unmodified `_create_training_view()` — it introspects
each family table's live columns at call time, so simply re-calling it (not editing it) was
sufficient to pick up the new columns for both `cxg_event_context_features`-derived rows.

The `oam_analysis.cxg_event_model_matrix_v1` / `cxg_plus_360_model_matrix_v1` snapshot
tables (in `oam_analysis`, built by `run_cxg_split_analysis.py`'s `materialize_split_
matrices()`) were intentionally left untouched — confirmed live that they are already stale
relative to `cxg_training_matrix_v1` w.r.t. Phase A/B/v2's own columns (e.g.
`nearest_defender_role`, `nearest_defender_style_archetype` are absent from them too), so
this is an established, pre-existing pattern in this pipeline: downstream analysis scripts
join new feature tables directly by `event_id` rather than relying on those two snapshot
tables to auto-refresh. Not re-litigated here; a future qualification task can join Phase
C's columns from `cxg_event_context_features` the same way prior phases' scripts join
`cxg_defensive_360_features`/archetype tables.

Script: `scripts/materialize_cxg_phase_c_rolling_window.py`.

## Step 4 — Basic sanity EDA (not full qualification)

Read live from `oam_features.cxg_training_matrix_v1` (15,737 rows = CxG event-wide track;
`WHERE has_360_frame` = 3,960 rows = CxG+ track).

| Feature | Track | n | null (cold-start) | null (other) | min | median | mean | max |
|---|---|---|---|---|---|---|---|---|
| `defensive_action_rate_15m` | CxG event | 15,737 | 0 | 0 | 0.0 | 2.80 | 2.97 | 23.28 |
| `defensive_action_rate_15m` | CxG+ | 3,960 | 0 | 0 | 0.0 | 2.53 | 2.67 | 15.04 |
| `defensive_action_rate_60m` | CxG event | 15,737 | 0 | 0 | 0.0 | 2.87 | 3.01 | 23.28 |
| `defensive_action_rate_60m` | CxG+ | 3,960 | 0 | 0 | 0.0 | 2.56 | 2.71 | 15.04 |
| `territorial_dominance_last_15m` | CxG event | 15,737 | — | 16 | −1.0 | 0.138 | 0.137 | 1.0 |
| `territorial_dominance_last_15m` | CxG+ | 3,960 | — | 3 | −1.0 | 0.180 | 0.174 | 1.0 |
| `cross_match_defensive_rate` | CxG event | 15,737 | 906 (5.76%) | 0 | 1.49 | 2.67 | 2.66 | 4.65 |
| `cross_match_defensive_rate` | CxG+ | 3,960 | 273 (6.89%) | 0 | 1.77 | 2.44 | 2.48 | 3.90 |

`defensive_action_rate_{15,30,45,60}m` has no null-reason column entries in either track
(`defensive_action_rate_null_reason` is `NULL`-everywhere-populated for 0 rows) — every shot
in the corpus has at least a nonzero elapsed window, per Step 2's finding.
`territorial_dominance_last_15m`'s small null count (16 / 3 out of 15,737 / 3,960) matches
the pre-existing null condition on the frozen 5-minute feature (no attacking events observed
in-window), not a new failure mode introduced by the 15-minute extension.
`cross_match_defensive_rate`'s null rate (5.76% event-wide, 6.89% CxG+) is entirely
`team_first_match_in_dataset` cold start — `null (other)` is 0 in both tracks, i.e. the
defending-team resolution (via the `matches` home/away join) never failed to identify a
defending team for any shot.

**Plausibility check (single-prior-match dominance).** Verified live against team_id 784
(one of the 13 floor teams with exactly 3 matches, matches 7532 → 7546 → 7562
chronologically): team 784's whole-match defensive rate in their first match (7532) is
`210 actions / 98.133 minutes = 2.139940927440208`. Every shot in their second match (7546,
where team 784 is defending) carries `cross_match_defensive_rate = 2.139940927440208` —
an exact match, confirming the rolling value for a team with exactly one prior match is
dominated entirely by that one match, as required (the weight cancels out of the normalized
average with only one term, also covered by
`test_cross_match_rolling_rate_single_prior_match_dominated_by_it`).

## Row-count reconciliation

- CxG event-wide track: 15,737 rows before and after — Phase C is purely additive
  (`ALTER TABLE ADD COLUMN` + scoped `UPDATE`), no row was added, removed, or filtered.
- CxG+ track (`has_360_frame`): 3,960 rows before and after, same reasoning — CxG+ is a
  `WHERE has_360_frame` view over the same underlying table, so it inherits the new columns
  automatically with no separate materialization step.
- `oam_features.cxg_event_context_features` row count unchanged at 15,737 (verified via the
  materialize script's own verification query: `n = 15737` for all summary counts).

## Tests + regression check

19 new unit tests added in `tests/features/cxg/test_phase_c_rolling_window.py`, covering:
`decay_weight` (magnitude/monotonicity/domain validation), `cross_match_rolling_rate`
(cold-start null, single-prior-match dominance, recency weighting direction, exact
weighted-average math against a hand-computed reference), `period_start_clock`,
`defensive_action_rates` (all 4 null-reason branches, full-window vs. partial-window
denominator correctness, team/type/period filtering), and `territorial_dominance_extended`
(null-without-clock, full dominance, window exclusion of stale events).

Full suite: **310 passed** (291 pre-existing + 19 new), 0 regressions. Baseline was
re-verified live before starting (291), not assumed from a stale prior count.

## Files

- `src/opponent_adjusted/features/cxg/phase_c_rolling_window.py` — pure computation logic.
- `tests/features/cxg/test_phase_c_rolling_window.py` — unit tests.
- `scripts/materialize_cxg_phase_c_rolling_window.py` — BigQuery materialization.
- `audit_outputs/cxg_analysis/phase_c_rolling_window_features/phase_c_report.md` — this report.
