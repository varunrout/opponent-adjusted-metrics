# Task: CxG+ ODI Feature — Phase 1 Build (Phase 0 spike is complete and approved)

## Context

Phase 0 spike is done and reported at `audit_outputs/cxg_analysis/odi_defprofile_spike/freeze_frame_identity_report.md`. Bottom line: `shot_freeze_frame_players` has 100% non-null `player_id`/`position_id` for defender rows (`teammate = FALSE`), across all competitions and both inside/outside the 360 cohort, with roster coherence confirmed. Decision: **build ODI directly on `shot_freeze_frame_players` identity — no fuzzy geometric matching needed.**

Two known gaps to carry forward as documented eligibility, not silently dropped:
- 274 shots (1.7%) have no `freeze_frame` array at all.
- 93 of the 3,960 360-eligible shots (2.3%) have `has_360_frame = TRUE` but zero `shot_freeze_frame_players` rows.

Before writing the involvement/rolling pipeline, **first verify that `shot_freeze_frame_players.x`/`y` is a genuine shot-time snapshot** (not stale or interpolated from an earlier event) — check StatsBomb's freeze_frame semantics against how the existing `three_sixty_context.py` module treats frame timing, and confirm in 1-2 sentences in the Phase 1 report before proceeding. This affects whether "nearest defender at shot time" is actually accurate.

Follow the repository's existing governance style throughout: explicit typed contracts (`contracts.py` pattern), validation/QA before anything is called done, local evidence under `audit_outputs/`, no BigQuery table considered real until row-count/reconciliation checks pass. Do not touch frozen S/E1-E12 feature code or existing `three_sixty_context.py` F-family derivations. Do not start model training, calibration, or promotion — this is feature engineering only.

## Build: ODI (On-pitch Defensive Index) pipeline

**1. Defensive involvement event stream.**
For every shot with a resolvable nearest defender (`shot_freeze_frame_players`, `teammate = FALSE`, minimum-distance `player_id` to shot location), write one row per shot: `(player_id, team_id, match_id, shot_event_id, shot_timestamp_seconds, period, statsbomb_xg, is_goal)`. New BigQuery table `oam_analysis.cxg_defensive_involvement_v1` with a typed contract following `contracts.py` (nullable fields, `data_version`/`silver_schema_version` lineage, documented grain key). Explicitly record shots excluded due to the two known gaps above (missing freeze_frame, missing freeze-frame rows despite 360 coverage) rather than silently omitting them.

**2. Rolling 15-minute ODI aggregator.**
For each shot in the corpus, compute each on-pitch defender's ODI at that timestamp: sum of `statsbomb_xg` minus goals conceded, over involvement rows for that player in the trailing 15 match-clock minutes. Must be period-aware — check how existing E-family time-based features (`event_context.py`/`event_context_extended.py`) handle match-clock/period boundaries and follow the same convention; do not bridge windows incorrectly across halftime/ET boundaries. The window must strictly exclude the shot currently being scored (no self-leakage). Cold-start (first 15 minutes of a player's tournament involvement) must be null with an explicit eligibility flag — not zero, not an assumed average.

**3. On-pitch roster resolution at timestamp.**
Reusable helper: given a match and a timestamp, determine who is on the pitch for the defending team and their nominal position, using `starting_xi_players` position assignment adjusted for `substitutions` (`minute`/`second`/`period`) before that timestamp. Needed for both ODI eligibility and position-slot aggregation (e.g. mean center-back ODI).

**4. CxG+ feature output.**
Produce final per-shot ODI features into `oam_analysis.cxg_odi_features_v1` (typed contract, joinable to `cxg_analysis_360_v1`/`cxg_plus_360_model_matrix_v1` on `event_id`). At minimum: nearest-defender ODI, mean back-line ODI. Propose additional features if justified by the data (e.g. GK ODI using existing `defending_keeper` logic).

**5. Validation before calling this done.**
- Row-count reconciliation against the 360-eligible cohort (3,960 shots, minus the 93-shot known gap — numbers should reconcile exactly).
- Null-rate report explaining eligibility (cold-start, missing freeze-frame) vs. any genuine unexplained missingness.
- Real-match sanity narrative: pick an actual match/player, show ODI dropping after conceding shots in the trailing window, confirm the numbers behave sensibly.

## Deliverables checklist

- [ ] Shot-time snapshot verification note in the Phase 1 report
- [ ] `oam_analysis.cxg_defensive_involvement_v1` + typed contract
- [ ] `oam_analysis.cxg_odi_features_v1` + typed contract
- [ ] On-pitch roster resolution helper (reusable, tested)
- [ ] Validation report under `audit_outputs/cxg_analysis/` with row counts, null-rate explanation, sanity narrative
- [ ] No model training, no changes to frozen feature code

Report back with a summary before moving to Phase 2 (defensive profile clustering, separate task).
