# Task: Backfill EDA + Univariate Coverage for `shot_geometry` and `buildup` Families

## Context

The governed feature-family taxonomy has 8 families total in the analysis layer: `base_identity_target`, `shot_geometry`, `event_context`, `buildup`, `defensive_360`, `goalkeeper_360`, `line_shape_360`, `opponent_adjusted`. An audit found that **6 of 8 have full EDA + univariate coverage** (`cxg_feature_inventory_v1`, `cxg_null_profile_v1`, `cxg_summary_stats_v1`, `cxg_eda_distribution_bins_v1`, `cxg_univariate_target_v1`/`cxg_split_univariate_v1` rows, plus rendered charts local+GCS) — but **`shot_geometry` and `buildup` have zero rows in any analysis table and zero charts anywhere**, despite both having real, fully-populated Gold tables in `oam_features`:

- `cxg_shot_geometry_features`: 15,737 rows, 8 columns (`visible_goal_angle_delta`, `visible_goal_angle_proxy`, `phase_location`, plus base/id columns) — confirmed live via `audit_outputs/cxg_null_report/cxg_feature_null_report.md`, generated 2026-08-20.
- `cxg_buildup_features`: 15,737 rows, 33 columns (possession-origin/tempo/transition features like `possession_start_x`, `possession_age_s`, `high_regain`, `regain_to_box_entry_s`, etc.) — same source, confirmed live.

Both tables are real and populated. This is not a missing-data problem — it's a gap in the analysis/EDA layer that never processed them. Root causes found by code trace (verify these against current code before assuming they're still accurate — code may have moved since this audit):

1. **`render_cxg_feature_eda_appendix.py`, `AppendixRenderer.features()`** (~line 103): has a hardcoded `AND feature_family NOT IN ('buildup', 'shot_geometry')` filter in the SQL that selects which features to render per-feature EDA charts for. No comment or rationale in the code for why. This alone explains why neither family has per-feature appendix charts.
2. **Possible naming mismatch**: `scripts/materialize_cxg_feature_family_tables.py` (the script whose `_classify()` function assigns raw JSON feature keys to a family) uses the family key `"geometry"` for what becomes table `cxg_shot_geometry_features`, while `src/opponent_adjusted/analysis/cxg.py`'s `FAMILY_TABLES`/`event_families` and `cxg_charts.py`'s `FAMILY_SURFACE` all use the key `"shot_geometry"`. If `cxg_feature_inventory_v1` rows were ever seeded using the `"geometry"` name, any downstream filter/join keyed on `"shot_geometry"` would silently return nothing. Verify whether this mismatch is real and current, or whether it's already consistent and the appendix exclusion is the only blocker.
3. **`buildup` classification**: in the same materializer script, `_classify()` has no code path that ever returns `"buildup"` — every feature key resolves to `event_context`/`goalkeeper_360`/`line_shape_360`/`defensive_360`/`geometry`, or raises `ValueError`. Yet `cxg_buildup_features` demonstrably has 33 real populated columns per the null report. This means either the materializer has a `buildup`-classifying branch that wasn't found in this audit (re-check the full file, don't assume the audit's read was complete), or `cxg_buildup_features` was populated by a different mechanism/script/version than the one currently in the repo. **Figure out which is true before writing any fix** — do not guess.

## What "done" looks like

By the end of this task, `shot_geometry` and `buildup` must have **exactly the same shape of coverage** the other 6 families already have, at the same depth (univariate only — no bivariate/correlation/multivariate, matching the existing split-policy scope boundary that's already in force for every other family):

1. `cxg_feature_inventory_v1` — one row per feature column in both families (8 candidate columns for `shot_geometry` minus id/metadata columns; 33 minus id/metadata for `buildup` — compute the exact expected feature-only count from the null report's column list, don't assume it's the full column count since some are `event_id`/`data_version`/etc metadata, not features).
2. `cxg_null_profile_v1` — same rows, matching null counts already visible in the null report (e.g. `visible_goal_angle_delta` 77.75% null, `regain_to_box_entry_s` 71.49% null, etc. — these are real eligibility gaps already documented, not new problems to solve).
3. `cxg_summary_stats_v1` — summary stats for every feature in both families.
4. `cxg_eda_distribution_bins_v1` — quantile bins for numeric features, category bins for STRING/BOOL features (matching existing convention).
5. `cxg_univariate_target_v1` and `cxg_split_univariate_v1` — **train-split-only computation**, exactly like the `opponent_adjusted` extension task before this one. Do not use full-dataset numbers for feature promotion signal — this is a locked convention (see `docs/cxg_split_policy_and_parallel_plan.md`'s status note and the existing `opponent_adjusted` extension report for the pattern to replicate).
6. Chart coverage: standard 4-chart family-overview set (`null_profile_bar`, `summary_box`, `eda_histogram`, `target_lift_bar`) for both families via `cxg_charts.py`, plus per-feature EDA appendix charts via `render_cxg_feature_eda_appendix.py` (once the hardcoded exclusion is removed) — both local (`audit_outputs/cxg_analysis/<new-run-id>/`) and uploaded to GCS, using the registry-fix pattern (delete-then-insert scoped to run_id) already built and tested in the prior chart-registry-fix task — do not reintroduce the `CREATE OR REPLACE` overwrite bug.

## Steps

### 1. Investigate first — don't fix blind

- Read the full current `scripts/materialize_cxg_feature_family_tables.py`, confirm whether `_classify()` really has no `buildup` branch, or whether the repo state has moved since the audit. Report what you find.
- Query `cxg_shot_geometry_features` and `cxg_buildup_features` directly — confirm row counts, column lists, and whether `data_version`/`feature_version` match the currently-pinned values used everywhere else (`data_version = b0bc9f22dd77c206ddedc1d742893b3bbe64baec`, check `feature_version` against `DEFAULT_FEATURE_VERSION = "cxg_gold_v1_e13_f1_f15"` in `cxg.py`). If either table's `feature_version`/`data_version` is stale or doesn't match `cxg_training_matrix_v1`'s current pin, flag this clearly — do not silently join against a stale table.
- Query `cxg_feature_inventory_v1` directly for `WHERE feature_family IN ('geometry', 'shot_geometry', 'buildup')` — report exactly what's there today (expect zero rows, but confirm rather than assume, and check both the `geometry`/`shot_geometry` spelling to settle the naming-mismatch question definitively).
- Check `cxg_training_matrix_v1` (the view built by `_create_training_view` in the materializer) — does it currently join in `cxg_shot_geometry_features`/`cxg_buildup_features` at all? If the view's `FAMILY_TABLES` dict also uses `"geometry"` instead of `"shot_geometry"`, the join key mismatch could mean columns from `cxg_shot_geometry_features` aren't even reaching `cxg_analysis_event_v1` today. Trace this all the way through before concluding what's broken.

### 2. Fix root cause(s), not just the symptom

- If it's a naming mismatch: standardize on `"shot_geometry"` everywhere (matches the analysis-layer convention already used by `cxg.py`/`cxg_charts.py`/`render_cxg_feature_eda_appendix.py`) — fix the materializer script's `FAMILY_TABLES`/`_classify()` return value, not the other direction, since three files already agree on `"shot_geometry"` and only one disagrees.
- If `buildup` truly has no classification path in the current materializer: determine how `cxg_buildup_features` got populated with real data despite that (check `bq` job history / table `last_modified` timestamp if accessible, or ask before assuming) — this matters because if the table was built by an old/orphaned version of the script, blindly re-running today's materializer could `CREATE OR REPLACE` it into an empty table. **Do not re-run the materializer against `buildup` until you're certain doing so won't destroy the existing 33-column populated table.** If in doubt, back up current `cxg_buildup_features` (e.g. `CREATE TABLE cxg_buildup_features_backup_<date> AS SELECT * FROM cxg_buildup_features`) before touching anything upstream of it.
- Remove the hardcoded `feature_family NOT IN ('buildup', 'shot_geometry')` exclusion in `render_cxg_feature_eda_appendix.py` once the underlying data path is confirmed sound — not before.

### 3. Materialize the analysis-layer rows

Extend `cxg_feature_inventory_v1`, `cxg_null_profile_v1`, `cxg_summary_stats_v1`, `cxg_eda_distribution_bins_v1`, `cxg_univariate_target_v1`, `cxg_split_univariate_v1` for both families — same INSERT-only pattern used in the `opponent_adjusted` extension task (never `CREATE OR REPLACE`, which would blow away the other 6 families' rows). Use a fresh `run_id` for any chart-registry-touching step, following the existing UTC-timestamp convention.

### 4. Render and publish charts

- `cxg_charts.py`: family-overview charts for `shot_geometry` and `buildup` (4 each: null_profile_bar, summary_box, eda_histogram, target_lift_bar). No pitch/correlation chart types — same scope boundary as every prior family task.
- `render_cxg_feature_eda_appendix.py`: per-feature EDA charts for every feature in both families, after removing the exclusion.
- Watch for the same numeric-vs-categorical misclassification risk already found and fixed for `defensive_profile_cluster` (`distinct_count > 2` heuristic) — check each `shot_geometry`/`buildup` feature's `data_type`/`distinct_count` and verify the right chart branch is taken, especially STRING columns like `phase_location`, `possession_start_zone`, `previous_action_type`, `set_piece_category`, `direct_vs_second_phase`, `restart_vs_live_regain`, `phase_control_state`, `phase_directness_bucket`.
- Upload to GCS using the already-fixed delete-then-insert registry pattern (`tests/analysis/test_cxg_charts_registry.py` / `tests/scripts/test_render_cxg_feature_eda_appendix_registry.py` cover this — run them, confirm still passing, don't regress).
- Confirm via direct GCS listing (byte-size check against local, same pattern as the last publication task) — don't just trust the upload call didn't raise.

### 5. Validation and reconciliation

- Row-count deltas for every one of the 6 analysis tables, before/after, matching expected feature counts computed from the null report's column lists (exclude `event_id`/`match_id`/`data_version`/`feature_version`/`materialized_at`/`is_goal`/`outcome_name`-type metadata/target columns — only count real feature columns).
- Confirm no existing family's rows were touched (spot-check counts for `base_identity_target`/`event_context`/`defensive_360`/`goalkeeper_360`/`line_shape_360`/`opponent_adjusted` are unchanged before/after).
- Train-split-only univariate signal report for both families, same honest-read style as the `opponent_adjusted` extension report — actual point-biserial correlations, quintile/category goal rates, explicit stable-vs-unstable-across-splits verdict per feature or feature group. Do not just report train numbers as if they were validated.
- Full test suite pass count (`python -m pytest -q`), confirm no regression against whatever the current baseline count is.

### 6. Report

Write `audit_outputs/cxg_analysis/shot_geometry_buildup_backfill/shot_geometry_buildup_backfill_report.md` covering:
- Root-cause finding for both families (naming mismatch confirmed/refuted for `shot_geometry`; classification-path mystery resolved for `buildup` — how did the table get populated if `_classify()` has no path to it)
- Fix applied, with before/after evidence (not just a description of the diff)
- Row-count reconciliation for all 6 analysis tables
- Chart coverage confirmation (local + GCS, both renderers, byte-size verified)
- Train-split-only univariate findings for every feature in both families, honest signal/no-signal verdict
- Full test suite pass count
- Explicit confirmation that no other family's data/charts/registry rows were altered

## What NOT to do

- Do not touch `defensive_360`, `goalkeeper_360`, `line_shape_360`, `event_context`, `base_identity_target`, or `opponent_adjusted` rows/charts/registry entries.
- Do not build or reactivate bivariate/correlation/multivariate/model-spec work for any family — univariate only, matching the locked split-policy scope.
- Do not re-run the materializer against `cxg_buildup_features` without first confirming it won't destroy existing populated data (back it up first if there's any doubt).
- Do not use `CREATE OR REPLACE` for any analysis-table insert — additive/scoped only, per the discipline already established in the `opponent_adjusted` extension and chart-registry-fix tasks.
- Do not train, calibrate, or promote any model.
- Do not silently paper over the `buildup` classification mystery — if you can't determine how the table was populated, say so clearly in the report rather than guessing.

Report back with a summary and file paths when complete.
