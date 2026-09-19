# Task: Correlation/Redundancy Screen + PCA — New Governed Step Between Univariate and Bivariate

## Context: pipeline stage order (locked, supersedes the old plan's ordering)

`docs/cxg_split_policy_and_parallel_plan.md` originally sequenced redundancy/correlation *after* bivariate. That ordering is wrong and is being corrected: the real sequence is

**Summary Statistics → EDA → Univariate → Correlation/Redundancy Screen (this task) → PCA if applicable (this task) → Bivariate (next task, not this one)**

This task builds the Correlation/Redundancy Screen and PCA step as its own governed analysis step — with its own BigQuery tables, its own charts, its own validation report — the same way every prior phase in this project has worked (ODI Phase 1, defensive-profile clustering Phase 2, `opponent_adjusted` extension). **Do not proceed to bivariate interaction testing in this task** — that's explicitly the next task, out of scope here.

**Two tracks, kept separate, per the project's locked architecture:**
- **CxG (`cxg_event`)** — event-wide, 15,737 shots, feature pool restricted to `base_identity_target` + `event_context` only (CxG structurally has no 360/opponent-adjusted features — confirmed via `event_families = {base_identity_target, shot_geometry, event_context, buildup}` in `src/opponent_adjusted/analysis/cxg.py`).
- **CxG+ (`cxg_plus`)** — 360-eligible cohort, 3,960 shots. Feature pool = CxG's own qualified candidates (inherited, since CxG+ is an extension of CxG) **plus** `defensive_360`, `goalkeeper_360`, `line_shape_360`, `opponent_adjusted`.

**Critical non-negotiable principle: this step must never hard-drop a feature for having weak/no univariate signal.** A feature that shows nothing alone can still show a real interaction effect in bivariate — that's the whole point of doing bivariate at all. The *only* thing this step is allowed to trim is **true near-duplicate redundancy** (two features so highly correlated with each other that testing both in bivariate wastes budget on an effectively-duplicate hypothesis, not because either one is individually weak). Keep this distinction explicit throughout — it is easy to accidentally conflate "weak signal" with "redundant" and they are not the same filter.

## Step 0: Verify the univariate-qualified candidate pools (cross-check against numbers already computed this session — confirm, don't blindly trust)

These were computed live against `oam_analysis.cxg_split_univariate_v1` this session (train/validation/test point-biserial correlation, sign-stable across all three splits, `min(|r_train|,|r_val|,|r_test|)` ≥ 0.05). **Re-run the query yourself and confirm these match before proceeding** — if they don't match exactly, stop and report the discrepancy rather than silently using either version.

**CxG track (`cxg_event`), 6 candidates:**
`shot_x_sb` (base_identity_target), `last_action_interval_s`, `previous_action_time_gap_s`, `last_box_entry_to_shot_s`, `regain_height_speed_interaction`, `first_box_entry_to_shot_s` (all event_context).

**CxG+ track (`cxg_plus`), 18 numeric candidates:**
`visible_goal_angle_proxy`, `gk_distance_to_shooter`, `defensive_reset_index`, `rest_defence_reset_fraction`, `goal_mouth_defender_count`, `estimated_goalface_occlusion`, `defenders_between_ball_and_goal`, `defensive_line_depth`, `shot_corridor_occlusion`, `shooter_space_previous_linked_event`, `defensive_hull_area`, `defensive_centroid_x`, `min_defensive_compactness_sequence`, `pre_shot_receiver_space`, `nearest_defender_distance_delta`, `defensive_compactness`, `defensive_width`, `max_goal_exposure`.

**Plus, for CxG+ specifically:** `defensive_profile_cluster` (categorical, INT64-stored, `distinct_count=4` — already proven real/stable via Phase 2's own analysis, not in the numeric univariate table by construction, but must be included as a qualified CxG+ candidate) and the ODI trio (`nearest_defender_odi`, `mean_backline_odi`, `gk_odi` — 0/3 sign-stable, deliberately included anyway per the non-filtering principle above, specifically to test whether ODI's weak univariate signal is context-conditional in bivariate).

**Prerequisite check before finalizing the CxG+ pool — do this first, it changes the pool composition:** the 6 CxG candidates currently only have univariate stats computed under the `cxg_event` track (the full 15,737-shot population). Before treating them as pre-qualified for CxG+'s pool, re-run their point-biserial correlation (train/validation/test) specifically within `cxg_plus_360_model_matrix_v1`'s splits (the 3,960-shot 360-only population) — the 360-tracked matches aren't a random sample of all matches (skewed toward specific competitions), so signal strength/stability could differ in this subpopulation. Report which of the 6 hold up under this re-check and which don't; carry forward only the ones that remain sign-stable and meaningful within the 3,960-cohort specifically for CxG+'s pool (the full 15,737-population qualification still stands for CxG's own pool, untouched).

## Step 1: Pairwise correlation matrix, train-split only, per track

For each track, compute Pearson `r` for every pair within that track's qualified numeric candidate pool (from Step 0), fit on the train split only — same discipline as every prior redundancy screen in this project (Phase 2 defensive-profile clustering used this exact method: `defprofile/features.py`).

**Cross-check baseline — CxG track (already computed this session, verify):**

| Pair | r (train) |
|---|---|
| `last_action_interval_s` / `previous_action_time_gap_s` | **0.864** |
| `last_box_entry_to_shot_s` / `first_box_entry_to_shot_s` | 0.472 |
| (all other pairs) | ≤ 0.42 |

**Cross-check baseline — CxG+ track (already computed this session, verify), pairs ≥ 0.5:**

| Pair | r (train) |
|---|---|
| `defensive_reset_index` / `rest_defence_reset_fraction` | **0.996** |
| `defensive_compactness` / `defensive_hull_area` | 0.917 |
| `pre_shot_receiver_space` / `shooter_space_previous_linked_event` | 0.862 |
| `defensive_centroid_x` / `defensive_line_depth` | 0.852 |
| `nearest_defender_distance_delta` / `shooter_space_previous_linked_event` | -0.792 |
| `defenders_between_ball_and_goal` / `shot_corridor_occlusion` | 0.789 |
| `estimated_goalface_occlusion` / `shot_corridor_occlusion` | 0.772 |
| `defensive_compactness` / `defensive_width` | 0.755 |
| `defensive_compactness` / `defensive_line_depth` | -0.753 |
| `defensive_hull_area` / `defensive_line_depth` | -0.725 |
| `defensive_hull_area` / `defensive_width` | 0.717 |
| `nearest_defender_distance_delta` / `pre_shot_receiver_space` | -0.694 |
| `defensive_compactness` / `min_defensive_compactness_sequence` | 0.670 |
| `defensive_hull_area` / `min_defensive_compactness_sequence` | 0.657 |
| `defensive_reset_index` / `gk_distance_to_shooter` | -0.609 |
| `gk_distance_to_shooter` / `rest_defence_reset_fraction` | -0.608 |
| (several more in the 0.5–0.6 range — re-derive the full list yourself) |

## Step 2: Redundancy resolution — only trim r ≥ 0.85 pairs, apply established precedent where it exists

**Only pairs at or above r = 0.85 (train) count as true redundancy** — this threshold is the one already established by Phase 2's defensive-profile clustering feature selection (`docs` note: Phase 2 pruned pairs at this exact bar). Everything between 0.5 and 0.85 is real covariance between conceptually distinct measurements and should be **kept, not trimmed** — same reasoning Phase 2 used (e.g. it deliberately kept `defenders_within_3m`/`defenders_within_5m` at r=0.74 as legitimately distinct).

**Two pairs already have established precedent from Phase 2 — apply the same call for consistency (Phase 2's original report is no longer stored locally, only on Google Drive; the relevant rows are reproduced here so you don't need to fetch it):**

| Pair | r (Phase 2, train) | Phase 2's call |
|---|---|---|
| `defensive_centroid_x` / `defensive_line_depth` | 0.852 | **Kept `defensive_line_depth`** (governed F2 primary depth name), dropped `defensive_centroid_x` |
| `defensive_compactness` / `defensive_hull_area` | 0.917 | **Dropped both**, in favour of the primitives `defensive_length`/`defensive_width` (compactness is a literal product of length×width, hull_area a correlated composite of the same footprint) |

Apply these same calls in this step: drop `defensive_centroid_x`, keep `defensive_line_depth`; drop `defensive_compactness` and `defensive_hull_area`, keep `defensive_width` (already independently qualified in the univariate pool at r=-0.0726/-0.0901/-0.1452 — check whether `defensive_length` is also present in the qualified pool; if not, note that only `defensive_width` survives from that primitive pair here, which is fine, just report it honestly).

**Two pairs have no established precedent — resolve with an explicit tie-break rule, and show your reasoning per pair:**

1. `last_action_interval_s` / `previous_action_time_gap_s` (r=0.864, CxG track)
2. `pre_shot_receiver_space` / `shooter_space_previous_linked_event` (r=0.862, CxG+ track)

**Tie-break rule:** prefer the feature with the higher `min(|r_train|, |r_val|, |r_test|)` target-correlation (i.e. the more robustly-signed one across splits); if that's a near-tie, prefer whichever name maps more directly to the governed E/F-family taxonomy's primary description in `src/opponent_adjusted/features/cxg/contracts.py` (check both features' family/description there and note which reads as more "primary" vs. derived). Document the reasoning for both calls in the report — don't just state the outcome.

**`nearest_defender_distance_delta` / `shooter_space_previous_linked_event` (r=-0.792) and `nearest_defender_distance_delta` / `pre_shot_receiver_space` (r=-0.694)** are both below 0.85 — keep all three features, do not trim, but note the three-way relationship in the report since `nearest_defender_distance_delta` is correlated with both halves of the pair already resolved above.

## Step 3: Deliverables — new governed tables

- `oam_analysis.cxg_feature_correlation_v1` — one row per unique pair per track: `track`, `feature_a`, `feature_b`, `r_train`, `n_train`, `is_redundant` (boolean, r≥0.85), `resolution` (`kept`/`dropped`/`kept_both_moderate`), `resolution_reason`. Typed contract, matching this project's existing pattern (see `defprofile/contracts.py` for the shape to follow).
- `oam_analysis.cxg_bivariate_candidate_pool_v1` — the final, post-redundancy-trim candidate list per track, with columns: `track`, `feature_family`, `column_name`, `qualification_reason` (`univariate_stable`, `deliberately_included_despite_weak_signal` for ODI, `categorical_proven_stable` for `defensive_profile_cluster`, `inherited_from_cxg_reverified` for the CxG-origin CxG+ candidates). This table is the explicit, auditable handoff artifact the next (bivariate) task should read from — don't let the next task re-derive the pool from scratch.

## Step 4: PCA — diagnostic only, scoped correctly

**Where PCA applies:** CxG+'s `defensive_360`/`line_shape_360` pool (post-redundancy-trim) — many of these features plausibly measure the same latent "defensive shape" construct from different angles (distance, density, line depth, width, hull area). Run PCA there to check whether the pool collapses into a small number of latent dimensions.

**Where it's a formality:** CxG's own 5-candidate pool (post-trim) is small and heterogeneous (timing gaps, shot position) — run PCA anyway for completeness/consistency, but don't force an interpretation if it just confirms near-independence (report that honestly, don't manufacture a finding).

**Method, matching Phase 2's existing preprocessing convention exactly (`defprofile/features.py`):**
- Median imputation (fit on train only) + `StandardScaler` (fit on train only), applied unchanged to validation/test.
- Fit PCA on the train split's post-redundancy-trimmed pool only.
- Report explained variance ratio per component, cumulative variance, and loadings for the top components that reach a meaningful cumulative threshold (report the actual cumulative-variance curve, don't pre-decide a "meaningful" cutoff — let the real numbers speak, e.g. report whichever number of components explains 80% and also just report the full scree curve).

**This is diagnostic only — explicitly do not:**
- Feed PCA components into the existing K-Means defensive-profile clustering (that's frozen, Phase 2 work — do not modify it).
- Use PCA components as replacement features anywhere downstream yet — that's a modeling decision for a later, separately-scoped task.
- Treat PCA as a filtering step — it does not remove any feature from the bivariate candidate pool table in Step 3.

**Deliverable:** `oam_analysis.cxg_pca_components_v1` (track, component_number, explained_variance_ratio, cumulative_variance_ratio) + `oam_analysis.cxg_pca_loadings_v1` (track, component_number, feature_name, loading).

## Step 5: Charts

- **Correlation heatmap, one per track** — this resurrects the `correlation_heatmap` chart type that was dropped during the univariate-only revert (it previously lived incorrectly bundled under `line_shape_360`'s bivariate-adjacent chart; now it has a correct home as this step's own chart, backed by `cxg_feature_correlation_v1`, not `cxg_correlation_v1` which stays retired). Add via the existing `CxGChartRenderer` pattern in `src/opponent_adjusted/analysis/cxg_charts.py` — a new `feature_family`-independent chart scoped by `track` instead (check whether the existing chart-spec/registry schema needs a `track` column added, or whether track can be encoded as a synthetic family name like `cxg_event_correlation`/`cxg_plus_correlation` — your call, but be consistent with the existing registry schema rather than inventing a parallel one).
- **PCA scree plot, one per track** — explained variance (and cumulative) per component, standard bar+line combo chart.
- Render both to local disk under a fresh `run_id` (`cxg-analysis-<UTC-timestamp>` convention) first; only upload to GCS + materialize the chart registry after confirming the registry delete-then-insert-by-run_id fix (already built and tested in the prior chart-registry task) is used — never `CREATE OR REPLACE` the registry.

## Step 6: Report

Write `audit_outputs/cxg_analysis/correlation_screen_pca/correlation_screen_pca_report.md`:
- Step 0's re-verified candidate pools per track (including the CxG-inherited-features-reverified-on-360-subset result)
- Full correlation matrix findings per track, the redundancy table (all pairs ≥0.85, with resolution + reasoning for every pair including the two open tie-break calls)
- Final `cxg_bivariate_candidate_pool_v1` contents, both tracks
- PCA results per track: explained variance curve, interpretation (real latent structure found, or confirmed near-independence — report honestly either way)
- Chart coverage confirmation (local + GCS, both tracks)
- Row-count reconciliation for every new table
- Full test suite pass count, confirm no regression

## What NOT to do

- Do not perform any bivariate interaction testing (feature × feature × target) — that is explicitly the next task.
- Do not drop any feature from the bivariate candidate pool for weak/no univariate signal — only true near-duplicate redundancy (r≥0.85) is grounds for trimming, and even then only one of the pair, never both unless Phase 2's own precedent already did that (the compactness/hull_area case).
- Do not modify the existing K-Means defensive-profile clustering, ODI features/tables, or any frozen S/E1-E12/F1-F15 feature code.
- Do not touch `cxg_analysis_event_v1`, `cxg_analysis_360_v1`, or `cxg_analysis_opponent_adjusted_v1` schemas.
- Do not reactivate `cxg_correlation_v1` (stays retired) — this task's correlation table is a new, correctly-scoped one (`cxg_feature_correlation_v1`), not a resurrection of the old one.
- Do not use PCA components as replacement/derived features anywhere yet.
- Do not `CREATE OR REPLACE` any chart registry table.

Report back with a summary, the file paths, and the final candidate pool sizes per track when complete.
