# Task: Local + GCS Cleanup (Post Univariate-Phase Housekeeping)

## Context

Project state check confirmed EDA + univariate analysis is complete for all real feature families (`base_identity_target`, `event_context`, `defensive_360`, `goalkeeper_360`, `line_shape_360`, `opponent_adjusted` — `shot_geometry`/`buildup` are legacy labels with no live feature content, correctly empty, confirmed via a separate closed investigation). Before bivariate work starts, this is a housekeeping pass: delete local scratch files that are no longer needed (all reasoning docs already live on Google Drive), and remove one now-redundant GCS prefix. **This task performs deletions — follow the "what NOT to do" section exactly, and stop and ask if anything doesn't match what's described below.**

This sandbox's own filesystem access has a mount write-permission issue that has repeatedly blocked `rm` on these exact paths ("Operation not permitted") across multiple prior sessions — that's why this is being handed to you rather than done directly. Use your normal direct filesystem access.

## Part 1: Delete local `.md` analysis reports (all already backed up to Drive)

All 11 of the following `.md` files have been uploaded in full to Google Drive, folder `OpponentAdjustedMetrics/Reports/` (confirmed present via the Drive API — you don't need to re-verify the upload, just the local deletion). Local copies are pure duplicates at this point; the project convention going forward is **no `.md` analysis reports live locally — only charts (PNG/HTML), data tables (JSON/CSV), and SQL stay in `audit_outputs/`.**

Delete these 11 files:
```
audit_outputs/cxg_analysis/cxg-analysis-20260820T201934Z/split_analysis/cxg_split_analysis_report.md
audit_outputs/cxg_analysis/cxg_analysis_materialization_validation.md
audit_outputs/cxg_analysis/defprofile_phase2/defprofile_phase2_validation_report.md
audit_outputs/cxg_analysis/odi_defprofile_spike/freeze_frame_identity_report.md
audit_outputs/cxg_analysis/odi_phase1/odi_phase1_validation_report.md
audit_outputs/cxg_analysis/opponent_adjusted_extension/gcp_publication_report.md
audit_outputs/cxg_analysis/opponent_adjusted_extension/opponent_adjusted_analysis_extension_report.md
audit_outputs/cxg_analysis/opponent_adjusted_extension/opponent_adjusted_chart_coverage_validation.md
audit_outputs/cxg_analysis/shot_geometry_buildup_backfill/shot_geometry_buildup_backfill_report.md
audit_outputs/cxg_null_report/cxg_feature_null_report.md
audit_outputs/silver_acceptance/silver_acceptance_closure_report.md
```

After deleting, run `find audit_outputs -name "*.md" -type f` and confirm it returns nothing.

**Note:** `audit_outputs/cxg_null_report/cxg_feature_null_report.json` (the JSON sibling of the deleted null-report `.md`) should stay — JSON data files are kept locally, only `.md` narrative reports move to Drive.

## Part 2: Delete repo-root scratch file

Delete `tmp_patch_split.py` (repo root) — confirmed dead scratch code, already approved for deletion, never a real module import path. Confirm with `git status` or `ls` that it's gone and nothing else in the repo root references it (`grep -rn "tmp_patch_split" --include="*.py" .` should return nothing after deletion — also check before deleting, to be safe).

## Part 3: Sweep build/test caches

Delete these cache directories (all safe, regenerated automatically by pytest/ruff on next run, never version-controlled):
```
.pytest_cache/
.ruff_cache/
scripts/__pycache__/
src/opponent_adjusted/__pycache__/
tests/__pycache__/
tests/analysis/__pycache__/
tests/features/__pycache__/
tests/ingestion/__pycache__/
tests/models/__pycache__/
tests/scripts/__pycache__/
tests/storage/__pycache__/
```
(Also sweep any other `__pycache__` dirs found via `find . -type d -name "__pycache__"` that weren't listed above — the list above is what existed as of the last check, but re-glob to catch any new ones.)

After sweeping, run `python -m pytest -q` once to confirm the suite still passes (this will regenerate `.pytest_cache/`, which is expected and fine — the point was removing stale cache content, not preventing it from ever existing again).

## Part 4: Delete redundant GCS prefix (old, marker-order-defective Silver publication)

**Context:** Silver v1 had a `_SUCCESS`-marker-ordering defect that was fixed and republished under a new prefix (`silver_acceptance_closure_report.md`, already on Drive, has full detail if you need to re-read the history). Both prefixes have coexisted since:
- **Old (defective marker ordering, superseded):** `gs://oam-varun-260819-data/staged/statsbomb/b0bc9f22dd77c206ddedc1d742893b3bbe64baec/statsbomb_silver_v1_2/` — 72 objects, `_SUCCESS` dated 2026-08-19.
- **New (compliant, canonical — `oam_core` is reconciled against this one):** `gs://oam-varun-260819-data/staged/statsbomb/b0bc9f22dd77c206ddedc1d742893b3bbe64baec/statsbomb_silver_v1_2-remediated/` — 72 objects, `_SUCCESS` dated 2026-08-21.

User has explicitly confirmed: **delete the old prefix.** It's fully superseded — `oam_core`'s 18 governed tables were already reconciled against the remediated prefix (every table came back `skipped_existing`, meaning byte-identical row volume), so nothing downstream reads from the old prefix and no data is uniquely at risk.

**Before deleting, verify the safety conditions still hold — don't just trust this document:**
1. Confirm the remediated prefix (`statsbomb_silver_v1_2-remediated/`) still exists and has all 72 objects with a valid `_SUCCESS`.
2. Confirm `oam_core`'s tables currently reconcile against `data_version = b0bc9f22dd77c206ddedc1d742893b3bbe64baec` (spot-check row counts against what's in `silver_acceptance_closure_report.md`'s table, e.g. `events` = 2,156,823, `shots` = 15,737).
3. Only then delete the old prefix: `gsutil -m rm -r gs://oam-varun-260819-data/staged/statsbomb/b0bc9f22dd77c206ddedc1d742893b3bbe64baec/statsbomb_silver_v1_2/` (or equivalent via the `google-cloud-storage` Python client if `gsutil` isn't available).
4. Confirm via a fresh listing that the old prefix is now empty/gone and the remediated prefix is untouched.

**If either safety condition in step 1/2 doesn't hold, stop and report back — do not delete.**

## Part 5: Report

Write a short summary (no need for a formal report file for this one, a chat summary is fine) confirming:
- All 11 `.md` files deleted, `find audit_outputs -name "*.md"` returns empty
- `tmp_patch_split.py` deleted, no references remain
- All cache dirs swept, `pytest -q` still passes post-sweep (report the pass count)
- Old Silver GCS prefix deleted (or, if safety checks failed, exactly what didn't check out and why you stopped)

## What NOT to do

- Do not delete `audit_outputs/cxg_null_report/cxg_feature_null_report.json` — only the `.md` sibling.
- Do not touch any `.png`/`.html`/`.json`/`.csv`/`.sql` file anywhere under `audit_outputs/` — those stay local per the project convention.
- Do not delete `docs/archive/` (already-archived prompt docs from a prior cleanup — those are fine where they are, not part of this task).
- Do not delete the **remediated** Silver GCS prefix — only the old, superseded one.
- Do not touch any BigQuery table — that part of cleanup (dropping an orphaned duplicate `oam_core.cxg_shot_features`) was already done directly, not part of this task.
- Do not delete anything not explicitly listed above without stopping to ask first.

Report back with a summary when complete.
