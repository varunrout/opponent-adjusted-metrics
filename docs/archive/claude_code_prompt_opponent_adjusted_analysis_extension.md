# Task: Extend CxG+ Analysis to the New `opponent_adjusted` Feature Family (ODI + Defensive Profile Clusters)

## Context

Silver v1 Data Engineering acceptance is now formally CLOSED (`audit_outputs/silver_acceptance/silver_acceptance_closure_report.md`) — every downstream table can be treated as confirmed, not assumed.

Two new opponent-adjustment feature sources exist and are validated:
- `oam_analysis.cxg_odi_features_v1` (3,960 rows — nearest-defender ODI, mean back-line ODI, GK ODI, each with explicit `*_null_reason` columns)
- `oam_analysis.cxg_defensive_profile_clusters_v1` (3,960 rows — categorical `cluster_label` 0-3, plus `cluster_label_null_reason` for 101 geometry-ineligible rows, plus `split` and `cluster_model_version`)

Both are keyed on `event_id` and join cleanly to `cxg_analysis_360_v1` / `cxg_plus_360_model_matrix_v1`.

**Critical scoping constraint — read `docs/cxg_split_policy_and_parallel_plan.md` in full before starting.** On 21 Aug, the project deliberately reverted ALL bivariate/multivariate/correlation/model-spec analysis output back to univariate-only, specifically because bivariate work needs deliberate planning first (game-state/opponent-context interaction design was called out explicitly) and because full-dataset feature selection was found to risk overfitting — feature promotion must be split-aware (train-only), not full-dataset, going forward. **This task extends the analysis only as far as univariate target analysis, train-split-only. Do not build bivariate, correlation, multivariate, or model-spec outputs for these two features or for any other family — that scope boundary is intentional and still in force.**

Follow the existing governance style: typed contracts, validation before anything is called done, evidence under `audit_outputs/`, no BigQuery table considered real until reconciliation checks pass. No model training, calibration, or promotion. Do not touch Phase 1/2 ODI/defprofile code or tables, frozen S/E1-E12 feature code, or `three_sixty_context.py` F-family derivations.

## Family classification (locked)

Register a **new formal feature family: `opponent_adjusted`** — distinct from `defensive_360`/`goalkeeper_360`/`line_shape_360`/`event_context`. This reflects that ODI and defensive-profile-clusters are conceptually different from raw positional geometry: they are derived quality/structure signals (rolling defensive performance, clustered shape archetype), not direct per-shot geometric measurements. Add this family to whatever taxonomy registry/contract governs family membership (check `features/cxg/contracts.py` and however `cxg_context_taxonomy_v3` — referenced repeatedly in prior analysis docs — is defined/enforced; extend it consistently, don't create a parallel ungoverned label).

Members of `opponent_adjusted`:
- `nearest_defender_odi` (numeric, from `cxg_odi_features_v1`)
- `mean_backline_odi` (numeric)
- `gk_odi` (numeric)
- `defensive_profile_cluster` (categorical, from `cxg_defensive_profile_clusters_v1.cluster_label`)

Population: 360-eligible cohort only (3,960 shots) — same population as every other 360-family feature (`defensive_360`, `goalkeeper_360`, `line_shape_360`).

## Build: inventory through univariate, train-split-only

Follow the exact same structure, output tables, and column contracts already established for every other family in `docs/16_CXG_ANALYSIS_PLAN_LOCKED` (Steps 1-3 and 6 specifically — Feature Inventory, Summary Statistics, Feature-Level EDA, Univariate Target Analysis). Reuse existing derivation code/patterns wherever the existing family pipelines already do this (check `src/opponent_adjusted/analysis/` and whatever scripts materialize `cxg_feature_inventory_v1`, `cxg_null_profile_v1`, `cxg_summary_stats_v1`, `cxg_eda_distribution_bins_v1`, `cxg_univariate_target_v1` today) — extend them to include the new family, don't build a parallel one-off pipeline.

**1. Feature inventory** (`oam_analysis.cxg_feature_inventory_v1`).
Add rows for all 4 `opponent_adjusted` features: source table, analysis surface, family, data type, is_candidate_feature, is_360_feature=true, row/non-null/null counts, unique_count, example values, eligibility_note (document the cold-start and geometry-ineligibility null reasons explicitly here — this is exactly what this column is for), leakage_screen_initial (confirm and state explicitly: both features are strictly pre-shot / trailing-window, no self-shot leakage, per the Phase 1/2 validation reports — restate that reasoning here, don't just assert it).

**2. Null profile + summary statistics** (`cxg_null_profile_v1`, `cxg_summary_stats_v1`).
Same treatment as every other family. For null profile specifically: distinguish the *documented* null reasons (`cold_start_lt_15min`, `no_freeze_frame`, `shootout_stage_excluded`, `geometry_ineligible_all_features_null`, etc.) from any *unexplained* null — there should be none, since Phase 1/2 already proved every null is reasoned, but this step makes that visible in the governed inventory format rather than only in the standalone Phase 1/2 reports.

**3. EDA distribution bins** (`cxg_eda_distribution_bins_v1`).
Numeric distribution bins for the 3 ODI features; categorical distribution for `defensive_profile_cluster` (4 categories + null).

**4. Univariate target analysis — train-split-only (`cxg_univariate_target_v1`).**
This is the step that actually answers "does this carry signal." **Must be computed on the train split only**, using `oam_analysis.cxg_match_splits_v1` / `cxg_plus_360_model_matrix_v1` to identify train-split shots — this is the same split-aware discipline the split policy doc requires for any feature-promotion-relevant analysis, and these two features are exactly the kind of thing that policy was written to protect against overfitting on. Compute: non-null count, null_pct, target_rate_non_null vs null, bucket-based goal rate deciles (numeric features) or goal rate per category (`defensive_profile_cluster`), lift, univariate AUC-or-proxy, monotonicity where applicable, signal_strength_bucket, stability_flag, recommended_action — matching the existing `cxg_univariate_target_v1` column contract exactly.

Report the actual numbers plainly in the summary: does nearest-defender ODI show a real goal-rate gradient across deciles on train? Do the 4 defensive-profile clusters show train-split goal-rate separation consistent with the descriptive full-cohort numbers already reported in the Phase 2 report (6.1%-14.7%), or does it look different when restricted to train only? This is genuinely an open question — do not assume the answer.

## What NOT to do

- Do not build bivariate, cross-family correlation, multivariate/permutation, or model-spec outputs — for these two features or for reactivating any other family's already-reverted analysis. That scope decision stands until the project separately decides to resume it.
- Do not compute univariate analysis on the full dataset — train-split only, per the locked split policy.
- Do not touch Phase 1/2 ODI/defprofile code or tables, frozen S/E1-E12 feature code, `three_sixty_context.py`, or anything Silver/oam_core-related from the just-closed acceptance task.
- Do not add these features to any candidate model spec.

## Deliverables checklist

- [ ] `opponent_adjusted` family registered in the governed taxonomy (not a parallel ungoverned label)
- [ ] Feature inventory rows for all 4 features, with real eligibility/leakage documentation
- [ ] Null profile + summary statistics rows
- [ ] EDA distribution bins
- [ ] Train-split-only univariate target analysis rows, with an honest report of whether signal is present
- [ ] Validation report under `audit_outputs/cxg_analysis/` confirming row counts/reconciliation for each new table extension
- [ ] No bivariate/multivariate/model-spec work, no full-dataset (non-train-split) univariate computation

Report back with a summary when complete, including the actual univariate signal findings — that's the point of this task.
