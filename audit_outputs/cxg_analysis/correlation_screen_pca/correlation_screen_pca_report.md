# Correlation/Redundancy Screen + PCA — Report

**Status: complete.** New governed pipeline stage, inserted between Univariate and Bivariate: **Summary Statistics → EDA → Univariate → Correlation/Redundancy Screen (this task) → PCA (this task) → Bivariate (separate, next task — not built here)**. No bivariate interaction testing performed. No feature dropped for weak/no univariate signal — only true near-duplicate redundancy (train |r| ≥ 0.85) was ever grounds for trimming.

`python -m pytest -q` → **217 passed** (208 baseline + 9 new tests for `corrpca/features.py`'s pool logic), no regression.

---

## 1. Step 0 — candidate pool re-verification

**CxG track (`cxg_event`, full 15,737-shot population):** re-ran the sign-stable + `min(|r|)≥0.05` query against `cxg_split_univariate_v1` directly. Confirmed **exactly 6 candidates**, matching the given list exactly: `shot_x_sb`, `last_action_interval_s`, `previous_action_time_gap_s`, `last_box_entry_to_shot_s`, `regain_height_speed_interaction`, `first_box_entry_to_shot_s`.

**CxG+ track native numeric candidates:** the same query against the `cxg_plus` track initially returned 21 rows, not 18 — 3 extra (`attackers_in_box`, `box_numerical_balance_delta`, `defenders_in_box`) appeared because their `validation`-split point-biserial correlation is `NaN` (BigQuery `CORR()` returns `NaN`, not `NULL`, when the underlying values are too sparse to define a correlation — consistent with these F3 box-occupation columns' well-documented ~97% null rate from Phase 2). A `NaN` value is not sign-stable by definition, so these 3 are correctly excluded, narrowing to **exactly 18**, matching the given list exactly.

**CxG+ prerequisite check (does it matter — yes, materially):** re-ran the 6 CxG-origin candidates' point-biserial correlation specifically within `cxg_plus_360_model_matrix_v1`'s own train/validation/test splits (the 3,960-shot 360-only cohort), not the full-population numbers. Only **3 of 6 hold up**:

| Feature | train | validation | test | Holds? |
|---|---|---|---|---|
| `shot_x_sb` | 0.193 | 0.208 | 0.154 | **Yes** (min\|r\|=0.154) |
| `last_action_interval_s` | -0.108 | -0.090 | -0.088 | **Yes** (min\|r\|=0.088) |
| `previous_action_time_gap_s` | -0.094 | -0.083 | -0.071 | **Yes** (min\|r\|=0.071) |
| `last_box_entry_to_shot_s` | -0.089 | **-0.033** | -0.105 | No — validation min\|r\|=0.033 < 0.05 |
| `regain_height_speed_interaction` | +0.035 | **-0.0002** | +0.168 | No — sign flips (near-zero/negative validation vs. positive train/test) |
| `first_box_entry_to_shot_s` | -0.069 | **-0.029** | -0.106 | No — validation min\|r\|=0.029 < 0.05 |

The 3 that fail remain fully valid, untouched candidates in **CxG's own pool** (full-population qualification stands regardless) — they are only excluded from feeding into CxG+'s pool. This confirms the task's premise that the 360-tracked matches aren't a random sample: signal that's robust across the full 15,737-shot population is not automatically robust within the smaller, competition-skewed 3,960-shot subset.

---

## 2. Step 1 — pairwise correlation, train-split only, per track

**CxG track: 15 pairs (6 choose 2), verified against the given baseline — exact match.**

| Pair | r (train), given | r (train), computed |
|---|---|---|
| `last_action_interval_s` / `previous_action_time_gap_s` | 0.864 | **0.8639** ✓ |
| `last_box_entry_to_shot_s` / `first_box_entry_to_shot_s` | 0.472 | **0.4724** ✓ |
| (all other 13 pairs) | ≤ 0.42 | max = 0.4145 (`shot_x_sb`/`last_box_entry_to_shot_s`) ✓ |

**CxG+ track: 276 pairs (24 choose 2, including the 3 ODI candidates), verified against the given baseline — all 16 given rows match to 4 decimal places.**

**Finding beyond the given baseline:** the given table (pairs ≥ 0.5) predates the Step 0 reverification above — it does not include `shot_x_sb` in the correlation matrix at all. Once `shot_x_sb` is correctly carried forward as a reverified candidate, **two more pairs cross the 0.85 threshold**, neither of which appears in the given baseline:

| Pair | r (train) |
|---|---|
| `gk_distance_to_shooter` / `shot_x_sb` | **-0.8998** |
| `last_action_interval_s` / `previous_action_time_gap_s` (recomputed within the 360-cohort) | **0.8516** (vs. 0.8639 in the full CxG population — same pair, close but distinct value) |

Full ≥0.5 list (33 pairs total; the 16 given + these 2 + 15 more in the 0.5–0.6 range not enumerated in the given table) is in `cxg_feature_correlation_v1`.

---

## 3. Step 2 — redundancy resolution (train \|r\| ≥ 0.85 only)

**6 redundant pairs total for CxG+, 1 for CxG. Every pair's resolution and reasoning:**

| Track | Pair | r | Resolution | Reasoning |
|---|---|---|---|---|
| cxg_event | `last_action_interval_s` / `previous_action_time_gap_s` | 0.8639 | Drop `previous_action_time_gap_s` | **Tie-break, no precedent.** `last_action_interval_s` has higher `min(\|r_train\|,\|r_val\|,\|r_test\|)` (0.0922 vs 0.0842). Also genuinely different governed families — E5 "Possession Tempo Context" vs. E10 "Immediate Pre-Shot Sequence Context" — not a naming coincidence. |
| cxg_plus | `defensive_reset_index` / `rest_defence_reset_fraction` | 0.9959 | Drop `rest_defence_reset_fraction` | **Tie-break, no precedent** (highest r in the entire matrix). `defensive_reset_index` has higher min\|r\| (0.1706 vs 0.1678). |
| cxg_plus | `defensive_compactness` / `defensive_hull_area` | 0.9167 | **Drop both** | **Phase 2 precedent applied.** `defensive_compactness = defensive_width × defensive_length` (literal product); `defensive_hull_area` a correlated composite of the same footprint. Keep the primitive `defensive_width` (already independently qualified at train r = -0.0726 to -0.1452 across splits). `defensive_length` is **not** in this pool's qualified candidates — only `defensive_width` survives from that primitive pair, exactly as anticipated; reported honestly rather than papered over. |
| cxg_plus | `gk_distance_to_shooter` / `shot_x_sb` | -0.8998 | Drop `shot_x_sb` (from CxG+ pool only) | **Tie-break, no precedent — newly surfaced pair** (§2). `gk_distance_to_shooter` has higher min\|r\| within the 360-cohort (0.2132 vs 0.1544). Physically sensible: GK positioning and shot x-position are mechanically linked. `shot_x_sb` is **not** dropped from CxG's own pool — this redundancy is specific to the 360-cohort. |
| cxg_plus | `pre_shot_receiver_space` / `shooter_space_previous_linked_event` | 0.8624 | Drop `shooter_space_previous_linked_event` | **Tie-break, no precedent.** Near-tied on min\|r\| (0.0822 vs 0.0829 — within noise), so resolved via taxonomy primacy: both are F10 "Shooter / Receiver Space Evolution", but `pre_shot_receiver_space` is a direct state measurement while `shooter_space_previous_linked_event` is an explicitly derived/relational lookback (references a prior linked event) — kept the more primary one. |
| cxg_plus | `defensive_centroid_x` / `defensive_line_depth` | 0.8524 | Drop `defensive_centroid_x` | **Phase 2 precedent applied.** Kept the governed F2 primary depth name. |
| cxg_plus | `last_action_interval_s` / `previous_action_time_gap_s` (360-cohort) | 0.8516 | Drop `previous_action_time_gap_s` | Same underlying pair as the cxg_event resolution, same call for consistency — CxG+-population-specific univariate check also favours `last_action_interval_s` (min\|r\|=0.088 vs 0.071). |

**Three-way relationship noted, no trimming action** (both below 0.85): `nearest_defender_distance_delta` correlates with both halves of the resolved `pre_shot_receiver_space`/`shooter_space_previous_linked_event` pair — with `shooter_space_previous_linked_event` at r=-0.7919 and with `pre_shot_receiver_space` at r=-0.6936. All three features individually stay under 0.85 with each other, so none of these three pairs is itself grounds for trimming. `shooter_space_previous_linked_event` is still removed from the final pool — but because of its own ≥0.85 pair with `pre_shot_receiver_space`, not because of its (moderate) relationship with `nearest_defender_distance_delta`.

---

## 4. Final `cxg_bivariate_candidate_pool_v1` (Step 3 handoff artifact)

**CxG track: 5 features** (started at 6, dropped `previous_action_time_gap_s`):
`shot_x_sb`, `last_action_interval_s`, `last_box_entry_to_shot_s`, `regain_height_speed_interaction`, `first_box_entry_to_shot_s`.

**CxG+ track: 18 features** (started at 25 pre-trim: 3 reverified-inherited + 18 native + 1 categorical + 3 ODI; dropped 7 via redundancy: `rest_defence_reset_fraction`, `defensive_compactness`, `defensive_hull_area`, `shot_x_sb`, `shooter_space_previous_linked_event`, `defensive_centroid_x`, `previous_action_time_gap_s`):

| qualification_reason | Features |
|---|---|
| `inherited_from_cxg_reverified` (1) | `last_action_interval_s` |
| `univariate_stable` (13) | `visible_goal_angle_proxy`, `gk_distance_to_shooter`, `defensive_reset_index`, `goal_mouth_defender_count`, `estimated_goalface_occlusion`, `defenders_between_ball_and_goal`, `defensive_line_depth`, `shot_corridor_occlusion`, `min_defensive_compactness_sequence`, `pre_shot_receiver_space`, `nearest_defender_distance_delta`, `defensive_width`, `max_goal_exposure` |
| `categorical_proven_stable` (1) | `defensive_profile_cluster` |
| `deliberately_included_despite_weak_signal` (3) | `nearest_defender_odi`, `mean_backline_odi`, `gk_odi` |

Row-count confirmed: `SELECT track, COUNT(*)` → `cxg_event=5, cxg_plus=18` — exact match.

---

## 5. PCA (diagnostic only — does not filter the pool table above)

Method matches Phase 2's `defprofile/features.py` convention exactly: median imputation + `StandardScaler`, both fit on train only, PCA fit on train's post-redundancy-trimmed pool.

### CxG track (5 features) — confirmed near-independence, reported honestly, no manufactured finding

| Component | Explained variance | Cumulative |
|---|---|---|
| PC1 | 33.0% | 33.0% |
| PC2 | 24.2% | 57.1% |
| PC3 | 19.6% | 76.7% |
| PC4 | 13.5% | 90.3% |
| PC5 | 9.7% | 100.0% |

**4 of 5 components needed for 80% cumulative variance** — essentially no dimensionality reduction available; this is the small, heterogeneous pool behaving exactly as expected (timing-gap features, shot position, a momentum proxy — genuinely different measurements). PC3 loads almost entirely on `regain_height_speed_interaction` alone (loading 0.933, next-highest 0.213) — confirms it is close to orthogonal to everything else in the pool, consistent with its own weak/context-only univariate signal. This is a formality run, as expected — no interpretation forced.

### CxG+ track (17 numeric features) — real, interpretable latent structure found

| Component | Explained variance | Cumulative |
|---|---|---|
| PC1 | 23.9% | 23.9% |
| PC2 | 15.9% | 39.8% |
| PC3 | 9.6% | 49.5% |
| PC4 | 7.6% | 57.0% |
| PC5 | 6.5% | 63.5% |
| PC6 | 5.8% | 69.4% |
| PC7 | 5.3% | 74.7% |
| PC8 | 4.1% | 78.8% |
| PC9 | 4.0% | **82.9%** |
| PC10–PC17 | (declining) | 100% |

**9 of 17 components needed for 80% cumulative variance** — a real, moderate collapse (roughly half), not extreme but genuine. Top-2 component loadings:

- **PC1** (positive: `shot_corridor_occlusion` 0.40, `defenders_between_ball_and_goal` 0.39, `estimated_goalface_occlusion` 0.36, `gk_distance_to_shooter` 0.34; negative: `defensive_reset_index` -0.34, `visible_goal_angle_proxy` -0.33) — reads as a **"goal exposure / defensive coverage"** axis: occlusion and defender-count measures load one way, open-angle measures load the other. A sensible, physically coherent construct.
- **PC2** (positive: `defensive_line_depth` 0.44; negative: `defensive_width` -0.36, `min_defensive_compactness_sequence` -0.35, `max_goal_exposure` -0.33) — reads as a **"deep vs. high-and-wide block shape"** axis.

This is genuine latent structure in the defensive-shape-heavy portion of the pool, as the task anticipated. **Not fed into the existing K-Means defensive-profile clustering, not used as a replacement feature anywhere** — purely diagnostic, per the task's explicit scope boundary.

---

## 6. Chart coverage

New chart types added to `CxGChartRenderer` (`src/opponent_adjusted/analysis/cxg_charts.py`): `feature_correlation_heatmap` and `pca_scree`. Track is encoded as a **synthetic `feature_family`** (`cxg_event_correlation`, `cxg_plus_correlation`, `cxg_event_pca`, `cxg_plus_pca`) — no new column added to the existing chart-registry schema, consistent with every other chart type already registered there. Explicitly **not** a resurrection of the old `heatmap`/`cxg_correlation_v1` chart type, which stays retired — the dispatch branch for it is untouched, and no registry row references it.

4 new charts registered under fresh `run_id = cxg-analysis-20260821T174050Z` (existing 27 rows from the latest prior run copied forward via `register_chart_registry_for_run.py`, which auto-detects the latest run and prunes any row whose backing table no longer exists — none needed pruning this time). Rendered locally first, then uploaded — **used the already-fixed delete-then-insert-by-`run_id` registry pattern throughout, never `CREATE OR REPLACE`.**

| | Local | GCS |
|---|---|---|
| `cxg_event_correlation_heatmap` | .html (9.7KB) + .png (84KB) | uploaded |
| `cxg_plus_correlation_heatmap` | .html (21KB) + .png (175KB) | uploaded |
| `cxg_event_pca_scree` | .html (8.5KB) + .png (34KB) | uploaded |
| `cxg_plus_pca_scree` | .html (9.0KB) + .png (41KB) | uploaded |

Total for this run: 31 charts (27 carried forward + 4 new) = 63 GCS objects (31×2 + 1 manifest), confirmed via direct listing. `cxg_rendered_chart_registry_v1` confirmed to hold all 4 run_ids untouched: `cxg-analysis-20260820T201934Z`=24, `cxg-analysis-20260821T125853Z`=27, `cxg-analysis-20260821T153226Z`=27, `cxg-analysis-20260821T174050Z`=31 — no data loss, registry fix continues to work correctly.

---

## 7. Row-count reconciliation, all 4 new tables

| Table | Rows | Reconciliation |
|---|---|---|
| `cxg_feature_correlation_v1` | 291 | 15 (cxg_event, 6 choose 2) + 276 (cxg_plus, 24 choose 2) = 291 ✓ |
| `cxg_bivariate_candidate_pool_v1` | 23 | 5 (cxg_event) + 18 (cxg_plus) = 23 ✓ |
| `cxg_pca_components_v1` | 22 | 5 (cxg_event, 5 features → 5 components) + 17 (cxg_plus, 17 features → 17 components) = 22 ✓ |
| `cxg_pca_loadings_v1` | 173 | 20 (cxg_event, 4 components needed for 80% × 5 features) + 153 (cxg_plus, 9 components × 17 features) = 173 ✓ |
| Redundant pairs found | cxg_event: 1/15, cxg_plus: 6/276 | Matches §3 exactly |

All tables created via `INSERT`-only (never `CREATE OR REPLACE`) — no existing family's rows in any other analysis table were touched.

---

## 8. Confirmations

- No bivariate interaction testing performed anywhere in this task.
- No feature dropped for weak/no univariate signal — the ODI trio's inclusion despite 0/3 sign-stability is the explicit proof point (it survives every step here, precisely to be tested in bivariate).
- K-Means defensive-profile clustering, ODI features/tables, frozen S/E1-E12/F1-F15 feature code — **not modified**.
- `cxg_analysis_event_v1`, `cxg_analysis_360_v1`, `cxg_analysis_opponent_adjusted_v1` schemas — **not touched**.
- `cxg_correlation_v1` — **stays retired**, not reactivated; this task's table is the newly and correctly scoped `cxg_feature_correlation_v1`.
- PCA components — diagnostic only, not fed into K-Means, not used as replacement features anywhere.
- Chart registry — every write used the tested delete-then-insert-by-`run_id` pattern; zero `CREATE OR REPLACE` calls.

---

## File paths

- Contracts: `src/opponent_adjusted/analysis/corrpca/contracts.py`
- Candidate pools + redundancy resolution (with full reasoning): `src/opponent_adjusted/analysis/corrpca/features.py`
- Materializer: `scripts/materialize_cxg_correlation_pca.py`
- Chart additions: `src/opponent_adjusted/analysis/cxg_charts.py` (`_feature_correlation_heatmap`, `_pca_scree`, `_write_feature_correlation_png`)
- Tests: `tests/analysis/corrpca/test_features.py`
- This report: `audit_outputs/cxg_analysis/correlation_screen_pca/correlation_screen_pca_report.md`
- Charts: `audit_outputs/cxg_analysis/cxg-analysis-20260821T174050Z/rendered_charts/{cxg_event,cxg_plus}_{correlation_heatmap,pca_scree}.{html,png}`
