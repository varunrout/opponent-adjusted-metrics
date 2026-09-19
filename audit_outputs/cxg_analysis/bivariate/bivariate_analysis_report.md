# CxG / CxG+ Bivariate Interaction Testing

Pipeline stage: Summary Statistics -> EDA -> Univariate -> Correlation/Redundancy Screen -> PCA -> **Bivariate (this report, final analysis stage)**.

Run artifacts:
- `scripts/materialize_cxg_bivariate_interactions.py` -- fits every tier, writes both tables (INSERT-only, scoped delete-then-insert by `track`, never `CREATE OR REPLACE`).
- `src/opponent_adjusted/analysis/bivariate/{contracts.py, testing.py}` -- typed table contracts + the additive-vs-interaction LR-test engine, unit-tested (`tests/analysis/bivariate/test_testing.py`, 6 tests, synthetic ground truth).
- `scripts/materialize_cxg_bivariate_chart_registry.py` -- registers the 4 new chart rows under a fresh `run_id`, copy-forward pattern.
- Charts rendered/uploaded under run_id **`cxg-analysis-20260821T210549Z`**.

**Non-negotiable principle honored throughout:** every tested pair is recorded, including non-significant and failed-fit ones. Nothing is dropped from the output tables for a weak or null result -- see the `fit_status` / row-count reconciliation section below (182/182 pairs fit successfully; the column exists specifically so a future failed fit would still be visible, not silently absent).

---

## Step 0: candidate pool re-verification

Queried `oam_analysis.cxg_bivariate_candidate_pool_v1` live and compared against the task's given split.

| Track | Live count | Expected count | Match |
|---|---|---|---|
| `cxg_event` | 5 | 5 | Exact |
| `cxg_plus` | 18 | 18 | Exact |

No discrepancy -- proceeded without stopping. (`scripts/materialize_cxg_bivariate_interactions.py:verify_step0` hard-fails the run if this ever drifts.)

CxG track (5): `shot_x_sb`, `first_box_entry_to_shot_s`, `last_action_interval_s`, `last_box_entry_to_shot_s`, `regain_height_speed_interaction`.

CxG+ track (18): `last_action_interval_s`, `defenders_between_ball_and_goal`, `defensive_reset_index`, `nearest_defender_distance_delta`, `pre_shot_receiver_space`, `gk_distance_to_shooter`, `defensive_line_depth`, `defensive_width`, `estimated_goalface_occlusion`, `goal_mouth_defender_count`, `max_goal_exposure`, `min_defensive_compactness_sequence`, `shot_corridor_occlusion`, `visible_goal_angle_proxy`, `defensive_profile_cluster` (categorical, 4 non-null levels + a `null_cluster` level), `gk_odi`, `mean_backline_odi`, `nearest_defender_odi`.

---

## Step 1 (Tier 1): exhaustive within-track pairwise

Method: fit `is_goal ~ 1 + a + b` vs `is_goal ~ 1 + a + b + a:b` (logistic regression, train split), likelihood-ratio test on the interaction term. Categorical features (`defensive_profile_cluster`) are dummy-encoded (drop-first); the interaction term is then multi-df and `interaction_coef`/`interaction_se` are `NULL` by design (the LR test's `lr_stat`/p-value is still the joint significance of all interaction dummies). Benjamini-Hochberg FDR applied **within each track separately**.

| Track | Pairs tested | Raw p<0.05 | FDR p<0.10 | FDR p<0.10 AND validated on val split |
|---|---|---|---|---|
| `cxg_event` (C(5,2)=10) | 10 | 1 | 1 | 0 |
| `cxg_plus` (C(18,2)=153) | 153 | 24 | 13 | 4 |

All 163 pairs fit successfully (`fit_status = 'ok'`); none hit `insufficient_data` or `fit_failed`.

### CxG track: the one FDR survivor did not validate

| feature_a | feature_b | n_train | interaction_coef | p_raw | p_fdr | validated (val split) |
|---|---|---|---|---|---|---|
| `first_box_entry_to_shot_s` | `shot_x_sb` | 8,172 | +0.00177 | 0.0044 | 0.0441 | **False** |

This is the only CxG-track pair clearing FDR<0.10, but its interaction does not replicate on the validation split (wrong sign or p>=0.05 there -- `validates_on_split` failed). Recorded as a negative for downstream modeling purposes: promising on train, not confirmed.

### CxG+ track: 13 FDR survivors, 4 confirmed on validation

| feature_a | feature_b | n_train | p_raw | p_fdr | validated |
|---|---|---|---|---|---|
| `defensive_profile_cluster` | `visible_goal_angle_proxy` | 2,713 | 8.9e-13 | 1.4e-10 | **True** |
| `defensive_profile_cluster` | `shot_corridor_occlusion` | 2,713 | 3.1e-06 | 2.4e-04 | False |
| `defensive_profile_cluster` | `estimated_goalface_occlusion` | 2,713 | 8.5e-05 | 4.3e-03 | False |
| `defensive_width` | `last_action_interval_s` | 2,588 | 1.5e-04 | 5.8e-03 | False |
| `defensive_profile_cluster` | `goal_mouth_defender_count` | 2,713 | 2.1e-04 | 6.5e-03 | False |
| `defensive_profile_cluster` | `gk_distance_to_shooter` | 2,568 | 4.6e-04 | 1.16e-02 | **True** |
| `defensive_profile_cluster` | `defensive_reset_index` | 2,745 | 1.0e-03 | 2.21e-02 | False |
| `pre_shot_receiver_space` | `visible_goal_angle_proxy` | 1,824 | 1.2e-03 | 2.35e-02 | **True** |
| `defenders_between_ball_and_goal` | `defensive_profile_cluster` | 2,713 | 1.5e-03 | 2.61e-02 | False |
| `defensive_profile_cluster` | `gk_odi` | 1,838 | 1.8e-03 | 2.71e-02 | False |
| `defensive_line_depth` | `pre_shot_receiver_space` | 1,824 | 2.9e-03 | 3.65e-02 | **True** |
| `defensive_width` | `pre_shot_receiver_space` | 1,824 | 2.6e-03 | 3.65e-02 | False |
| `shot_corridor_occlusion` | `visible_goal_angle_proxy` | 2,713 | 7.6e-03 | 8.89e-02 | False |

**Four pairs are the only Tier 1 findings that clear both bars (FDR<0.10 on train AND replicate on validation):** `defensive_profile_cluster x visible_goal_angle_proxy`, `defensive_profile_cluster x gk_distance_to_shooter`, `pre_shot_receiver_space x visible_goal_angle_proxy`, `defensive_line_depth x pre_shot_receiver_space`. Notably `defensive_profile_cluster` (the Phase-2 K-Means archetype) is involved in 3 of the 4 -- its interaction with defensive-shape/geometry features is the strongest recurring Tier 1 signal in the CxG+ track. The other 9 FDR survivors are real train-split findings but are recorded as unconfirmed, per the "no promotion without validation" rule.

Full grid (all 163 raw + FDR p-values, including the 149 non-survivors) is in `oam_analysis.cxg_bivariate_interaction_v1` and rendered as `cxg_event_bivariate_tier1_significance_grid` / `cxg_plus_bivariate_tier1_significance_grid`.

---

## Step 2 (Tier 2): ODI trio x match-state

3 ODI features (`gk_odi`, `mean_backline_odi`, `nearest_defender_odi`) x 5 match-context features (`score_diff`, `game_state`, `late_game_leading`, `late_game_trailing`, `manpower_diff`, pulled from `oam_features.cxg_event_context_features`, not the trimmed pool) = 15 pairs. No FDR correction applied (task specifies FDR only for Tier 1's 163 tests).

| feature_a | feature_b | n_train | p_raw | validated |
|---|---|---|---|---|
| `gk_odi` | `manpower_diff` | 1,838 | 0.0145 | False |
| `nearest_defender_odi` | `manpower_diff` | 1,734 | 0.0292 | False |
| `nearest_defender_odi` | `late_game_leading` | 1,734 | 0.0327 | False |
| `nearest_defender_odi` | `score_diff` | 1,734 | 0.0831 | (not tested, below promotion bar) |
| ... 10 more pairs, all p_raw > 0.17 ... | | | | |

**None of the 15 pairs validate on the held-out split.** 3 clear raw p<0.05 on train (`gk_odi x manpower_diff` is the headline, lowest p) but this is exactly the kind of train-only signal the task's non-negotiable principle warns against promoting -- recorded as a negative for the ODI-conditional-on-match-state hypothesis. This is an honest null result: the ODI trio's weak univariate signal does not appear to be rescued by match-state conditioning, at least not for these 5 context features.

Stratified goal-rate table (ODI tercile x match-state tercile) for the headline pair `gk_odi x manpower_diff`:

| gk_odi tercile | manpower_diff tercile | n | goals | goal rate |
|---|---|---|---|---|
| low | low | 588 | 66 | 11.2% |
| mid | mid | 596 | 63 | 10.6% |
| high | high | 584 | 39 | 6.7% |
| (6 small off-diagonal cells, n=5-20 each, noisy) | | | | |

**Caveat:** `gk_odi` is ~94% exactly `0.0` (its zero-inflation is the same weak/unstable-signal pattern already documented in the correlation/PCA task). A tercile split on a near-constant column is necessarily approximate -- ties are broken by row order, not a meaningful magnitude boundary. `scripts/materialize_cxg_bivariate_interactions.py:_tercile` documents this; treat the "low/mid/high" gk_odi labels as roughly ordered, not precisely so. The dominant diagonal cells (low-low, mid-mid, high-high hold ~95% of train rows between them) show a mild monotonic decline in goal rate as both gk_odi and manpower_diff rise, consistent with the (non-validated) train-split raw signal, but this is descriptive, not confirmatory.

---

## Step 3 (Tier 3): null-cluster x shot geometry

### Blocker resolved: `oam_features.cxg_shot_geometry_features` has no feature columns

Confirmed live (consistent with the separately-closed `shot_geometry_buildup_backfill` investigation earlier this session): the table has only 5 lineage/metadata columns (`event_id`, `match_id`, `data_version`, `feature_version`, `materialized_at`) -- no `shot_distance_sb`, no `shot_angle_rad`, anywhere. Rather than block Tier 3 on a task premise that no longer matches live data, `shot_distance_sb` and `shot_angle_rad` were computed inline from `shot_x_sb`/`shot_y_sb` (always populated, `base_identity_target` family) using standard geometry:
- `shot_distance_sb = sqrt((120 - x)^2 + (40 - y)^2)`
- `shot_angle_rad` = angle subtended by the goal mouth (posts at x=120, y=36/44), via the law-of-cosines vector-angle formula.

Both are computed **inline in the analysis script only** -- not persisted as new Gold features, since this is a one-off statistical covariate for a single test, not a governed feature addition.

### Data-quality investigation: "suspiciously identical close-range coordinates" -- **not an artifact, confirmed genuine**

101 shots across all splits have `defensive_profile_cluster IS NULL` (geometry-ineligible, tagged `null_cluster`). Goal rate 42.6%, matching the earlier honest-opinion-review's figure. Of those 101:

- **50 shots (49.5%) sit at or within 0.1 units of the penalty spot** (`(108.0, 40.0)` exactly, or `(108.1, 40.1)`).
- Cross-referencing `oam_core.shots.shot_type_name` for these 101 event_ids: **exactly 50 are `shot_type_name = 'Penalty'`, 51 are `'Open Play'`** -- a perfect match to the coordinate-clustering count.

**Conclusion: the identical coordinates are real data, not a duplication bug or a placeholder default.** Penalty kicks are legitimately always taken from the same spot on a real pitch, so identical coordinates across 50 different event_ids, 6 different matches, and both goal/no-goal outcomes are exactly what genuine penalty-shot data should look like. This also explains *why* these shots are geometry-ineligible for the defensive-profile cluster: penalties have no meaningful defensive shape (no outfield defenders positioned normally, it's GK vs. shooter) -- clustering correctly excluded them rather than forcing a nonsensical shape assignment. The remaining 51 Open Play null-cluster shots have varied, non-clustered coordinates, consistent with genuinely missing 360 freeze-frame data for those specific events (unrelated to the "identical coordinates" question). Reported honestly: this is a real, sensible pattern in the data, not the kind of artifact the review's suspicion anticipated.

### Statistical test: cluster dummies jointly significant after controlling for geometry

Nested-model LR test (reduced: `is_goal ~ shot_distance_sb + shot_angle_rad`; full: reduced + `defensive_profile_cluster` dummies). This is a joint-significance test of the categorical term, not a two-way interaction -- it still populates the shared `cxg_bivariate_interaction_v1` schema (`feature_b` recorded as the synthetic label `shot_distance_sb+shot_angle_rad` to make the covariate set explicit), documented here as a deliberate, honest reuse of the generic LR-test row shape rather than a literal `a:b` interaction.

| feature_a | feature_b | n_train | lr_stat | p_raw |
|---|---|---|---|---|
| `defensive_profile_cluster` | `shot_distance_sb+shot_angle_rad` | 2,780 | 86.78 | 6.4e-18 |

**Cluster membership is highly significant even after controlling for shot distance and angle.** The defensive-profile archetype carries real information about goal probability beyond pure shot geometry -- consistent with Phase 2's original clustering rationale.

Stratified goal-rate table (cluster x shot-distance tercile):

| cluster | distance tercile | n | goals | goal rate |
|---|---|---|---|---|
| `null_cluster` | low (close) | 33 | 23 | **69.7%** |
| `cluster_3` | low | 48 | 26 | 54.2% |
| `cluster_2` | low | 193 | 39 | 20.2% |
| `cluster_1` | low | 516 | 103 | 20.0% |
| `cluster_0` | low | 137 | 12 | 8.8% |
| all clusters | high (far) | ~927 total | ~27 | 2.4-2.8% |
| all clusters | mid | ~926 total | ~136 | 6.2-12.1% |

`null_cluster`'s 69.7% goal rate in the close-range tercile is the highest cell in the table -- directly explained by the penalty-kick concentration confirmed above (penalties are short-range, high-conversion shots by construction). This is expected once the data-quality question is resolved, not a new surprising finding.

---

## Step 4 (Tier 4): cross-pool -- `last_action_interval_s` x CxG+-exclusive top performers

Per the task's explicit consistency rule, `shot_x_sb` is excluded here (it was redundancy-trimmed out of the final CxG+ pool against `gk_distance_to_shooter`, so it is excluded from this cross-pool comparison too, same as it would be from the CxG+ pool itself). That leaves `last_action_interval_s` as CxG's only own-pool survivor also present in the CxG+ 18-candidate pool.

Top CxG+-exclusive performers selected by `abs_signal` in `cxg_split_univariate_v1`, restricted to CxG+-pool members with adequate non-null support (hundreds-to-thousands of rows -- several higher-`abs_signal` candidates like `central_defenders_between_ball_and_goal` and `attackers_in_box` were excluded as unreliable, n=9-23): `visible_goal_angle_proxy`, `gk_distance_to_shooter`, `defensive_reset_index`.

**Important methodological note, documented rather than silently worked around:** all 3 of these exact pairs (`last_action_interval_s` x each) are also members of the CxG+ 18-pool and therefore already exhaustively tested in Tier 1's 153 pairs. Rather than refitting an identical model a second time (a pointless duplicate), Tier 4's rows reuse Tier 1's already-computed results, re-tagged `tier=4` -- same `n_train`, same coefficients, same p-values, same `fit_status`. This is a deliberate re-narration under the cross-pool framing the task asks for, not a second independent statistical test.

| feature_a | feature_b | p_raw | p_fdr (from Tier 1) |
|---|---|---|---|
| `last_action_interval_s` | `visible_goal_angle_proxy` | 0.0261 | 0.207 |
| `defensive_reset_index` | `last_action_interval_s` | 0.135 | 0.447 |
| `gk_distance_to_shooter` | `last_action_interval_s` | 0.335 | 0.657 |

None reach even nominal significance after correction. CxG's own strongest univariate-inherited feature does not show a robust cross-pool interaction with the CxG+-exclusive top performers.

---

## Step 5/6: tables and charts

### `oam_analysis.cxg_bivariate_interaction_v1`

182 rows total, INSERT-only via scoped delete-then-insert by `track` (`cxg_event` and `cxg_plus` deleted/reinserted independently, no other track's history touched).

| track | tier | rows |
|---|---|---|
| cxg_event | 1 | 10 |
| cxg_plus | 1 | 153 |
| cxg_plus | 2 | 15 |
| cxg_plus | 3 | 1 |
| cxg_plus | 4 | 3 |
| **total** | | **182** |

`fit_status`: 182/182 `'ok'`. Zero `insufficient_data`, zero `fit_failed` -- every planned test produced a usable result. The `fit_status` column (an addition beyond the task's literal column list, alongside `feature_a`/`feature_b` on the stratified table) exists specifically so a future failed fit would still show up as a recorded row rather than a silent gap; documented here as intentional schema extensions, not scope creep.

### `oam_analysis.cxg_bivariate_stratified_v1`

24 rows: 9 (Tier 2 headline, 3x3 gk_odi x manpower_diff tercile grid) + 15 (Tier 3, `defensive_profile_cluster` [5 levels including `null_cluster`] x shot-distance tercile).

### Charts

Registered under run_id **`cxg-analysis-20260821T210549Z`** via `scripts/materialize_cxg_bivariate_chart_registry.py` (copy-forward pattern, never `CREATE OR REPLACE`; the prior run's 31 rows were copied forward untouched and 4 new rows appended):

| chart_name | chart_type | feature_family |
|---|---|---|
| `cxg_event_bivariate_tier1_significance_grid` | `bivariate_significance_grid` | `cxg_event_bivariate` |
| `cxg_plus_bivariate_tier1_significance_grid` | `bivariate_significance_grid` | `cxg_plus_bivariate` |
| `cxg_plus_bivariate_tier2_stratified_bar` | `bivariate_stratified_bar` | `cxg_plus_bivariate` |
| `cxg_plus_bivariate_tier3_stratified_bar` | `bivariate_stratified_bar` | `cxg_plus_bivariate` |

All 35 charts for this run_id (31 carried forward + 4 new) rendered locally first (`--skip-upload`), verified, then uploaded to GCS (`gs://oam-varun-260819-artifacts/analysis/cxg/cxg-analysis-20260821T210549Z/rendered_charts/`) and registered in `cxg_rendered_chart_registry_v1` via the existing scoped delete-then-insert-by-run_id logic. `cxg_chart_registry_v1` now holds 5 distinct run_id batches (24/27/27/31/35 rows), all intact -- no run's history was overwritten.

An initial registration attempt used the same `chart_name` for both tracks' significance grids (`bivariate_tier1_significance_grid`), which collided on the file-naming key and silently overwrote one track's render with the other's -- caught by inspecting the rendered file count before uploading, fixed by track-prefixing chart names (matching the existing `cxg_event_correlation_heatmap` / `cxg_plus_correlation_heatmap` convention), and re-registered cleanly before any GCS upload occurred.

---

## Row-count reconciliation

| Table | Expected | Actual | Match |
|---|---|---|---|
| `cxg_bivariate_interaction_v1` | 10 + 153 + 15 + 1 + 3 = 182 | 182 | Yes |
| `cxg_bivariate_stratified_v1` | 9 + 15 = 24 | 24 | Yes |
| `cxg_chart_registry_v1` (new run_id) | 31 (carried forward) + 4 (new) = 35 | 35 | Yes |
| `cxg_rendered_chart_registry_v1` (new run_id) | 35 | 35 | Yes |

## Test suite

`python -m pytest -q` -> **223 passed** (217 prior baseline + 6 new `tests/analysis/bivariate/test_testing.py` tests covering: real-interaction detection, null-interaction non-significance, insufficient-data recorded-not-dropped, categorical multi-df interaction, and validation-split sign-checking both directions). No regressions.

---

## What was explicitly NOT done (per task constraints)

- No bivariate finding feeds into baseline modeling -- that remains a separate, unscoped, later task.
- No non-significant or failed pair was dropped from the output tables -- all 182 interaction tests and both stratified breakdowns are recorded regardless of outcome.
- The Phase 2 K-Means clustering, ODI feature code, and correlation/PCA tables were not modified -- read-only joins only.
- No track-mixing outside Tier 4's explicit, justified exception (and even there, no new fit was performed -- Tier 1 results were reused, not recomputed cross-track).
- No chart registry table was `CREATE OR REPLACE`d -- every write used the established scoped delete-then-insert pattern.
- No p<0.05 raw finding was reported as validated without both FDR correction (Tier 1) and validation-split confirmation.

---

## Summary for hand-off

**Confirmed, validated Tier 1 findings (4):** `defensive_profile_cluster x visible_goal_angle_proxy`, `defensive_profile_cluster x gk_distance_to_shooter`, `pre_shot_receiver_space x visible_goal_angle_proxy`, `defensive_line_depth x pre_shot_receiver_space` -- all CxG+ track, all clear FDR<0.10 on train and replicate on validation.

**Tier 3's strongest finding:** defensive-profile-cluster membership remains highly significant (p=6.4e-18) even after controlling for shot distance/angle -- the archetype carries real predictive signal beyond geometry.

**Tier 2 and Tier 4: clean negative results.** Neither the ODI-trio-x-match-state hypothesis nor the cross-pool `last_action_interval_s` pairing produced anything that survives validation -- both recorded in full as documented negatives, not omitted.

**Data-quality resolution:** the "suspiciously identical coordinates" concern is fully resolved as a non-issue -- confirmed genuine penalty-kick data, not a duplication artifact.
