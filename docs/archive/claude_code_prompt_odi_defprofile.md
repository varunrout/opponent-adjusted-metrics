# Task: CxG+ Opponent-Adjustment Features — Spike, then ODI + Defensive Profile Clustering

## Context

You're working on `opponent-adjusted-metrics-gcp`, a GCP-based StatsBomb analytics pipeline. CxG/CxG+ (expected-goals modelling) has gone through locked analysis phases documented in `docs/cxg_split_policy_and_parallel_plan.md` and `audit_outputs/cxg_analysis/`. The project is named "Opponent-Adjusted Metrics" but nothing in CxG/CxG+ today actually adjusts for opponent quality — it's a contextual xG model, not an opponent-adjusted one. This task adds the missing piece: two new CxG+ feature families that condition shot quality on the defenders actually facing the shot.

Follow the repository's existing governance style throughout: explicit typed contracts (see `src/opponent_adjusted/pipelines/silver/contracts.py` for the pattern), validation/QA checks before anything is called done, local evidence under `audit_outputs/`, no BigQuery table considered real until row-count/reconciliation checks pass, and no claims of "complete" or "frozen" without those checks. Do not touch existing frozen S/E1-E12 feature code. Do not start any model training, calibration, or promotion — this task is feature engineering only, same phase discipline as the rest of `oam_analysis`.

## Two features being built

**1. ODI (On-pitch Defensive Index).** For each shot, identify the defending players near the shot and compute a rolling defensive-quality signal for each: sum of StatsBomb `shots.statsbomb_xg` for shots where that player was the nearest defending player, over a trailing 15-minute match-clock window, minus goals actually conceded on those shots. This becomes a CxG+ feature (e.g. nearest-defender ODI, mean back-line ODI) reflecting current in-match defensive form of the specific players facing the shot — not team-level reputation.

**2. Defensive profile clustering.** Cluster the pre-shot defensive personnel geometry (using existing `defensive_360`/`line_shape_360` primitives, which are anonymous positional geometry, not player identity) into a small number of discrete defensive shape archetypes (e.g. 4-8 clusters via k-means/GMM). Each shot gets a categorical profile-cluster label as a CxG+ feature. This one does not need player identity — it's structural shape only, and it doesn't have the ambiguity problem ODI has.

## Phase 0 — MANDATORY FIRST STEP: freeze-frame identity data-shape spike

Do not write any ODI pipeline code before completing this phase and reporting results.

We discovered `shots.freeze_frame[]` (StatsBomb's per-shot freeze-frame, distinct from the generic 360 `three_sixty_players`/`Frame` used by existing `three_sixty_context.py` F-feature derivations) is already parsed into Silver table `shot_freeze_frame_players` (contract in `contracts.py`), with columns `player_id`, `player_name`, `position_id`, `position_name`, `teammate`, `x`, `y` per defender — i.e. potentially already-identified defenders, unlike the anonymous `three_sixty_players` table the rest of CxG+ uses. This table exists in `oam_core` but nothing downstream currently reads it.

Investigate and report back before proceeding:

1. What is the actual non-null population rate of `player_id` in `shot_freeze_frame_players`, across all shots and broken out by competition (WC22 / Euro 2020 / Euro 2024)? Is it populated for shots outside full-360-coverage matches too, or only for the 3,960-shot 360 cohort?
2. For rows with `teammate = False` (i.e. defenders), what fraction have non-null `player_id` and non-null `position_id`? This is the number that matters most — it tells us whether defender identity is usable directly, avoiding any anonymous-dot matching problem entirely.
3. Cross-check: for a sample of matches, do the `player_id`s appearing in `shot_freeze_frame_players` for a defending team match that team's `starting_xi_players` / `substitutions` roster for that match (sanity check that IDs are coherent, not corrupted or from the wrong team)?
4. Compare row-for-row (same `shot_event_id`) against the existing `three_sixty_players`/generic-frame path used by `three_sixty_context.py` for the same shots — do the `(x, y)` coordinates approximately correspond between the two frame sources (same defenders, same positions), confirming these are twin views of the same underlying freeze-frame rather than materially different data?

Write findings to `audit_outputs/cxg_analysis/odi_defprofile_spike/freeze_frame_identity_report.md`, with actual counts/percentages, not estimates. This determines everything downstream:

- **If defender `player_id` is well-populated (say >80% for defenders)**: ODI can be built directly on `shot_freeze_frame_players` identity — skip any fuzzy geometric matching entirely. This is the good outcome; proceed to Phase 1 using this table as the identity source.
- **If defender `player_id` is sparse or unreliable**: report the actual rate and stop for a decision checkpoint before writing any matching/inference logic — do not silently fall back to guessed identity without flagging it back to the user first.

## Phase 1 — ODI pipeline (only after Phase 0 confirms identity is usable)

1. **Defensive involvement event stream.** For every shot with a resolvable nearest defender (from `shot_freeze_frame_players`, `teammate = False`, minimum-distance `player_id` to the shot location, or ball location if more appropriate — document which you use and why), write one row per shot: `(player_id, team_id, match_id, shot_event_id, shot_timestamp_seconds, statsbomb_xg, is_goal)`. New BigQuery table, e.g. `oam_analysis.cxg_defensive_involvement_v1`, with an explicit typed contract following the `contracts.py` pattern (nullable fields, lineage columns `data_version`/`silver_schema_version`, documented grain key).

2. **Rolling 15-minute ODI aggregator.** For each shot in the corpus (the same shot used as the involvement event, plus every other shot from the same match), compute each on-pitch defender's ODI at that timestamp: sum of `statsbomb_xg` minus goals conceded, over involvement rows for that player in the trailing 15 match-clock minutes (period-aware; do not bridge across periods incorrectly — check how existing E-family time-based features in `event_context.py`/`event_context_extended.py` handle match-clock/period boundaries and follow the same convention). The window must strictly exclude the shot currently being scored (no self-leakage). Document cold-start handling explicitly: first 15 minutes of a player's tournament involvement should be null with an eligibility flag, not zero or an assumed league-average.

3. **On-pitch roster resolution at timestamp.** For any point in a match, determine who is on the pitch for the defending team and their nominal position: `starting_xi_players` position assignment, adjusted for `substitutions` (`minute`/`second`/`period`) before that timestamp. Write this as a reusable helper, since it's needed both for ODI eligibility and for aggregating position-slot ODI (e.g. mean center-back ODI).

4. **CxG+ feature output.** Produce final per-shot ODI-derived features (at minimum: nearest-defender ODI, mean back-line ODI; propose others if the data supports it, e.g. GK ODI using existing `defending_keeper` logic) into a new typed Gold-pattern table, e.g. `oam_analysis.cxg_odi_features_v1`, joinable to `cxg_analysis_360_v1`/`cxg_plus_360_model_matrix_v1` on `event_id`.

5. **Validation before calling this done:** row-count reconciliation against the 3,960-shot (or whatever Phase 0 determines) 360-eligible cohort, null-rate report explaining eligibility vs. genuine missingness (matching the existing null-governance pattern used elsewhere in `oam_analysis.cxg_null_profile_v1`), and a sanity spot-check narrative (e.g. "player X's ODI dropped after conceding these 2 shots in the 10 minutes prior" on a real match) confirming the numbers behave sensibly, not just that the pipeline runs without erroring.

## Phase 2 — Defensive profile clustering (independent of Phase 1, can run in parallel or after)

1. Extract pre-shot defensive shape features per 360-eligible shot from existing `defensive_360`/`line_shape_360` Gold family tables (`cxg_defensive_360_features`, `cxg_line_shape_360_features`) — do not recompute geometry from scratch, reuse what's already derived and validated.
2. Select a fixed-size, well-justified subset of shape scalars for clustering input (document the selection reasoning — avoid just throwing every column in). Handle missingness per the existing 360-family null-governance approach (many of these fields have documented eligibility gaps).
3. Fit clustering **train-split only** (use `oam_analysis.cxg_match_splits_v1` / `cxg_plus_360_model_matrix_v1` — this is a hard requirement, not optional, matching the project's split-aware discipline from `docs/cxg_split_policy_and_parallel_plan.md`). Try k in a reasonable range (e.g. 4-8), report silhouette score or equivalent per k, and propose a k with justification rather than picking one arbitrarily.
4. Assign every shot (train/validation/test) a cluster label via nearest-centroid to the train-fit clusters — validation/test shots must never influence centroid fitting.
5. Produce a human-readable interpretation per cluster (e.g. "cluster 2: deep low block, high compactness, GK covering near post" — derived from the actual mean feature values per cluster, not guessed) written into the validation report.
6. Output: `oam_analysis.cxg_defensive_profile_clusters_v1` with typed contract, joinable on `event_id`, containing cluster label plus the underlying k/model version for reproducibility.
7. Validation: row-count reconciliation against the 360-eligible cohort, cluster size distribution (flag if any cluster is degenerate/near-empty), and the interpretation writeup.

## What NOT to do

- Do not add ODI or profile-cluster features into any CxG+ candidate model spec or run any train/validate/test model fitting — that's a separate, later phase requiring explicit sign-off, same as the rest of this project's phase discipline.
- Do not touch or modify frozen S/E1-E12 feature code or existing `three_sixty_context.py` F-family derivations.
- Do not silently invent a fuzzy-matching identity-resolution scheme if Phase 0 finds `shot_freeze_frame_players` identity is unreliable — stop and report back instead.
- Do not skip Phase 0. It determines whether Phase 1 is straightforward (identity already exists) or needs a fundamentally different, harder approach (identity must be inferred).

## Deliverables checklist

- [ ] `audit_outputs/cxg_analysis/odi_defprofile_spike/freeze_frame_identity_report.md` with real counts
- [ ] New Silver/Analysis table contracts added to the appropriate contracts file(s), following existing patterns
- [ ] `oam_analysis.cxg_defensive_involvement_v1`
- [ ] `oam_analysis.cxg_odi_features_v1`
- [ ] `oam_analysis.cxg_defensive_profile_clusters_v1`
- [ ] Validation reports for both features under `audit_outputs/cxg_analysis/` with row counts, null-rate explanations, and sanity narratives
- [ ] No model training, no changes to frozen feature code, no BigQuery table claimed done without a validation report backing it
