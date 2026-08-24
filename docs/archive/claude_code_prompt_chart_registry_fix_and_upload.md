# Task: Fix Chart Registry Overwrite Bug, Then Upload `opponent_adjusted` Charts to GCP

## Context

The prior task (chart coverage for the `opponent_adjusted` family) rendered 8 new charts (4 family-overview + 4 per-feature EDA appendix) successfully to local disk under `audit_outputs/cxg_analysis/cxg-analysis-20260821T125853Z/`, but deliberately used `--skip-upload` on both renderers rather than push to GCP, because a real bug was found: `CxGChartRenderer._materialize_render_registry()` (in `src/opponent_adjusted/analysis/cxg_charts.py`) does `CREATE OR REPLACE TABLE cxg_rendered_chart_registry_v1` populated from **only the current run's rows** — meaning any upload would silently delete every other run's chart-registry history from that table. The equivalent method in `scripts/render_cxg_feature_eda_appendix.py` (`materialize_registry`, feeding `cxg_feature_eda_chart_registry_v1`) has the same `CREATE OR REPLACE TABLE` pattern and needs the same fix.

Nothing has been uploaded to GCS yet for this family, and `cxg_rendered_chart_registry_v1`/`cxg_feature_eda_chart_registry_v1` in BigQuery do not yet know about the new charts. This task fixes the underlying bug, then completes the actual GCP publication.

## Part 1: Fix the overwrite bug in both registry-materialization methods

**`CxGChartRenderer._materialize_render_registry()`** (`src/opponent_adjusted/analysis/cxg_charts.py`) and **`AppendixRenderer.materialize_registry()`** (`scripts/render_cxg_feature_eda_appendix.py`): both currently do a blind `CREATE OR REPLACE TABLE ... AS SELECT ... UNION ALL ...` built only from the current run's `rendered`/`charts` list — destroying every previously-recorded run's rows in that table.

Fix both to be additive per run_id, not destructive:
- Preferred approach: `DELETE FROM {table} WHERE run_id = @run_id` (parameterized, not string-interpolated) followed by an `INSERT INTO {table} (...) VALUES (...)` (or `INSERT ... SELECT` from a temp/staging pattern) for the current run's rows only — this makes a re-run of the *same* run_id idempotent (delete-then-insert) while never touching other run_ids' rows.
- If the table doesn't exist yet, create it first with an explicit schema (don't rely on `CREATE TABLE AS SELECT` inferring types from a single run's data only).
- Apply the identical fix pattern to both files — they should not diverge in approach.
- Add or extend a test that proves this: materialize registry rows for a fake run_id A, then for run_id B, then assert run_id A's rows are still present and unchanged. This is the regression test that would have caught the original bug.
- Run the full test suite (`python -m pytest -q`) and confirm no regression, report the real pass count.

**Do not delete or alter any existing row in either registry table for any prior run_id as part of testing this fix** — if you need to test destructively, use a temp/scratch table or mock the BigQuery client, not the real `oam_analysis` tables.

## Part 2: Verify current registry state, then upload

1. Before uploading, query `cxg_rendered_chart_registry_v1` and `cxg_feature_eda_chart_registry_v1` and report their current row counts and which `run_id`s exist — confirm whether the earlier `cxg-analysis-20260820T201934Z` run's rows are still present (they should be, since no upload/materialize happened for the new run yet) or already missing (which would mean damage happened before this fix — report clearly which case it is, don't assume).
2. With the fix in place, run both renderers again for the `opponent_adjusted`-inclusive `cxg-analysis-20260821T125853Z` run **without** `--skip-upload`: this uploads the 27 family-overview chart files (HTML+PNG, but only the ones not already uploaded under this run_id if the renderer supports skipping re-upload of unchanged files — otherwise re-uploading all 27 is acceptable, just confirm none are corrupted/truncated) plus the appendix's 158 feature charts, plus both manifests, to `gs://oam-varun-260819-artifacts/analysis/cxg/cxg-analysis-20260821T125853Z/`.
3. After upload, confirm via direct GCS listing (not just trusting the upload call didn't error): object count under `.../rendered_charts/` and `.../eda_appendix/` for this run_id matches the local file count (27×2 + manifest = 55 objects; 158×2 + manifest = 317 objects — confirm exact expected counts against what's actually on local disk first, don't assume these numbers, recompute from the actual local file listing).
4. Confirm `cxg_rendered_chart_registry_v1` now has rows for **both** `cxg-analysis-20260820T201934Z` (24, unchanged) and `cxg-analysis-20260821T125853Z` (27, new) — this is the proof the fix works, not just an assertion.
5. Confirm `cxg_feature_eda_chart_registry_v1` similarly has both the pre-existing 154-row run and the new 158-row run coexisting.

## Part 3: Report

Write `audit_outputs/cxg_analysis/opponent_adjusted_extension/gcp_publication_report.md`:
- The bug found and the fix applied (code diff summary, not full diff)
- The new regression test added
- Full test suite pass count
- Registry row counts before and after, per run_id, proving no data loss
- GCS object counts confirmed via direct listing, both local-vs-uploaded reconciliation
- Explicit confirmation: local artifacts (already existed) + BigQuery tables (already existed) + GCS artifacts (new, this task) + chart registries (fixed and now correctly populated for both runs) are all now consistent — this is the "landed in GCP and local" closure the user asked for.

## What NOT to do

- Do not touch analysis-table content (`cxg_feature_inventory_v1`, `cxg_univariate_target_v1`, etc.) — this task is registry-mechanism and upload only.
- Do not touch Phase 1/2 ODI/defprofile code/tables, frozen S/E1-E12, `three_sixty_context.py`, or Silver/oam_core.
- Do not delete or overwrite any prior run's GCS objects or registry rows.
- Do not skip the "before" state check in Part 2 step 1 — you need to know whether damage already happened before claiming the fix resolved anything.

Report back with a summary when complete.
