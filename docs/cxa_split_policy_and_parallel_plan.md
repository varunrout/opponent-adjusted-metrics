# CxA / CxA+ Split Policy And Parallel Modelling Plan

Date: 2026-09-19
Mirrors: [cxg_split_policy_and_parallel_plan.md](cxg_split_policy_and_parallel_plan.md),
adapted from CxG's shot-level grain to CxA's pass-level grain. Data facts referenced
below are sourced from [cxa_data_feasibility_audit.md](cxa_data_feasibility_audit.md);
this document does not re-derive them.

## Decision

Same decision CxG made: full-dataset analysis (EDA, null profiling, summary stats) is
valid exploratory work, but P_create feature promotion and model evaluation must be
split-aware, by `match_id`, before any feature is treated as confirmed or any metric is
reported as final. Test data stays sealed until final reporting.

## Why (pass-level specific)

- CxA's unit of leakage is the same as CxG's: a match. Two passes from the same match
  share possession structure, tactical context, and (for CxA+) the same 360 capture
  quality — splitting by pass row instead of match would leak match-level context
  across train/validation/test exactly as it would for shots.
- CxA has one leakage risk CxG did not: **within-possession pass sequences**. Multiple
  passes in the same `possession_id` are highly correlated (a team building toward a
  chance produces several passes in a row, several of which may or may not be the
  eventual `key_pass_id`). Splitting by `match_id` already resolves this — a whole
  possession, and therefore every pass in it, stays on one side of the split — but it
  is called out explicitly here because it is easy to defeat by accident (e.g. a
  future `GroupKFold` mistakenly grouped by `possession_id` instead of `match_id` would
  still leak across matches only, not within a match, which is a smaller but real gap;
  `match_id` is the only correct group key).
- CxA+'s population is a 21.9%-of-all-passes subset selected by 360 coverage
  (point 7 of the audit). 360 coverage is a property of the **match** (StatsBomb
  captures 360 per-match, not per-event-selectively), so CxA+'s population is nested
  inside whichever matches happen to have 360 data — the same `has_360_match` property
  CxG's split table already tracks.

## Model Tracks

### CxA (event-only)

Scope: every pass in the corpus.

Population:
- All rows in `oam_features.cxa_event_v1_training_matrix`.
- Confirmed row count: 608,722 passes (audit point 2/7).
- Confirmed positive rate: 11,303 `Y_create = TRUE` (1.857%, audit point 8).

Initial baseline (mirrors CxG's XY-only baseline):
- `start_x`, `start_y` (pass origin only, no pass-shape information).

Candidate geometry layer:
- Train-only feature confirmation will decide promotion from: `pass_length`,
  `pass_angle`, `end_x`, `end_y`, `pass_height_name`, `pass_type_name`,
  `pass_technique_name`, `pass_body_part_name`, `is_through_ball`, `is_switch`,
  `is_cross`, `is_cut_back`, `possession_id`-derived phase context, `play_pattern_name`,
  `minute`, `second`.

### CxA+ (360 at reception)

Scope: passes whose reception is covered by a StatsBomb 360 frame.

Population:
- Rows in `oam_features.cxa_plus_v1_training_matrix`.
- Confirmed row count: 133,143 passes (audit point 7).
- Confirmed positive rate: 2,830 `Y_create = TRUE` (2.126%, audit point 8) — noted
  above as a real population-composition difference from the CxA event-only track, not
  an artifact; do not treat CxA and CxA+ Y_create rates as directly comparable without
  accounting for this.

Initial baseline:
- Same as CxA (`start_x`, `start_y`).

Candidate 360 core (v1, direction-independent only — see audit point 9 for why
goal-relative reception features are deferred):
- `reception_nearest_opponent_distance_m`
- `reception_opponents_within_5m`
- `reception_opponents_within_8m`
- `reception_opponents_visible`
- `reception_teammates_visible`
- `reception_frame_player_count`

Deferred expansion (not built this stage, requires a validated attacking-direction
convention for pass/reception context, separate work from this pipeline):
- Goal-relative geometry at reception (visible goal angle, defenders between receiver
  and goal, distance to goal).

## Required Split Design

**Decision: reuse CxG's existing match-level split assignment rather than generating a
new independent one.** CxG's `oam_analysis.cxg_match_splits_v1` already covers the
identical 610-match corpus CxA's passes span (confirmed live: `COUNT(DISTINCT
match_id)` on `oam_core.passes` under the schema-version filter = 610, matching
`cxg_match_splits_v1`'s row count exactly). There is no correctness reason to draw a
second, independent random assignment over the same match population, and doing so
would actively hurt future cross-project work (a match landing in CxG-train but
CxA-test would make it impossible to later build any feature or evaluation that spans
both projects without leakage). CxA's split table therefore **copies the same
`match_id -> split` assignment, same `split_seed`**, and adds CxA's own pass-level
aggregate stats per match (a different aggregation grain than CxG's shot-level
counts, but the same partition).

Canonical split table:
- BigQuery table: `oam_analysis.cxa_match_splits_v1`
- Key: `match_id`
- Fields: `split` (copied from `cxg_match_splits_v1`), `split_seed` (copied),
  `has_360_match` (copied), `pass_count`, `completed_pass_count`,
  `create_count` (Y_create=TRUE passes), `create_rate`, `plus_pass_count` (CxA+
  population rows), `plus_create_count`, `plus_create_rate`, `source_run_id`
  (`cxg_match_splits_v1`'s `run_id`, for provenance), `materialized_at`.

Resulting split proportions (inherited from CxG, confirmed unchanged by construction):
train 70% / validation 15% / test 15% of matches.

Split constraints (same as CxG, restated at pass grain):
- Split by `match_id`, never by pass row, never by `possession_id`.
- `create_rate` (the pass-level equivalent of CxG's goal-rate balance) is reported per
  split for both tracks as part of split validation, not assumed balanced a priori.
- CxA+ population and `plus_create_rate` are reported per split; validation/test must
  each retain enough CxA+ rows to evaluate CxA+ meaningfully (CxG's constraint,
  restated) — expected to hold automatically since `has_360_match` is inherited
  unchanged from CxG, which already validated this.
- Test stays sealed after creation, same as CxG.

## Sequential Plan

1. Build `oam_analysis.cxa_match_splits_v1` by joining `cxg_match_splits_v1` to
   pass-level aggregates from `cxa_event_v1_training_matrix` /
   `cxa_plus_v1_training_matrix`, grouped by `match_id`.
2. Validate split balance for both CxA and CxA+ (`create_rate`, CxA+ coverage per
   split) — report only, no modelling.
3. Create split-aware modelling surfaces (mirrors CxG's `*_model_matrix_v1` tables):
   - `oam_analysis.cxa_event_model_matrix_v1`
   - `oam_analysis.cxa_plus_360_model_matrix_v1`
4. Train-only feature confirmation (train split only):
   - univariate signal on train only
   - pair-interaction / redundancy screening on train only
5. Validate selected features on the validation split:
   - direction stability, support stability, uplift over the XY baseline.
6. Establish clean baselines:
   - CxA event-wide XY baseline (`start_x`, `start_y` only, logistic).
   - CxA+ 360-cohort XY baseline.
7. Train candidate P_create models in parallel (CxA event-wide, CxA+ 360) — still
   P_create only, no P_convert.
8. Compare validation metrics (log_loss, Brier, AUC — same metric family CxG used) and
   calibration.
9. Freeze final feature set and hyperparameters for P_create.
10. Run test once for final P_create model report.
11. Only after the P_create test report: begin P_convert (Y_goal conditioned on
    Y_create = TRUE), as its own separate, later-reviewed stage.

## Current Status

Steps 1-2 of the sequential plan above (`cxa_match_splits_v1`, materialized and
validated) are built as part of this task's deliverables (see
[cxa_data_feasibility_audit.md](cxa_data_feasibility_audit.md), point 11, and
`scripts/materialize_cxa_match_splits.py`). Steps 3-11 are future intent, matching how
CxG's own split-policy document distinguishes its built Steps 1-4 from its still-future
Steps 5-11 — this document should be read the same way: a design that is partially
executed, not a status report claiming the whole pipeline is done.
