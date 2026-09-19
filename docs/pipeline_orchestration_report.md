# GCP Orchestration for the ETL + Analysis Pipeline — Report

**Status: complete and validated end-to-end against real GCP infrastructure.** 3 Cloud Run Jobs (`oam-ingest`, `oam-transform`, `oam-analyse`), chained by a Cloud Workflows definition (`oam-pipeline`), all provisioned via Terraform, all built/pushed/executed for real. **No 4th "model"/"training" job or trigger exists anywhere in this work** — confirmed explicitly in every section below.

---

## 1. Script-to-job inventory (Step 1 investigation)

Investigated via full file reads (not filename guessing), cross-checked against live BigQuery table locations before assignment.

### Ingestion → `oam-ingest`

| Script | Reads | Writes | Verdict |
|---|---|---|---|
| `scripts/fetch_statsbomb_subset.py` | StatsBomb Open Data GitHub raw JSON | `gs://{bucket}/raw/statsbomb/{data_version}/` (Bronze) via `GCSRawStatsBombStore`, create-only/idempotent | **oam-ingest** entrypoint — already has a complete GCS-mode CLI (`--gcs-bucket`, `--data-version`, `--source-ref`, `--with-events`, `--with-360`), used directly |
| `src/opponent_adjusted/ingestion/subset_fetch.py`, `statsbomb_source.py` | — | — | Library helpers imported by the above; bundled into the image, not invoked directly |

### Silver/Gold → `oam-transform`

| Script | Reads | Writes | Verdict |
|---|---|---|---|
| `scripts/build_statsbomb_silver.py` | Bronze GCS | Silver Parquet, ordered-upload (`_SUCCESS` last), immutability-guarded | **oam-transform** step 1 |
| `scripts/publish_oam_core.py` | Silver GCS | BigQuery `oam_core` (18 tables), idempotent (`skipped_existing` when row counts match) | **oam-transform** step 2 |
| `scripts/materialize_gold_cxg.py` | BigQuery `oam_core` (events, three_sixty_*) via imported `audit_cxg_e13_f1_f15.py` helpers | **Confirmed live bug**: its `DATASET` constant (imported from `audit_cxg_e13_f1_f15.py`) is `"oam_core"`, but the live `cxg_shot_features` table is in `oam_features` (created by an earlier version of the script; the import was edited later without a re-run) | **oam-transform** step 3 — invoked via import + module-attribute patch (see §3), not via editing the script |
| `scripts/materialize_cxg_feature_family_tables.py` | BigQuery `oam_features.cxg_shot_features` (default) | BigQuery `oam_features` family tables + `cxg_training_matrix_v1` view | **oam-transform** step 4 |
| `scripts/repair_oam_core_v1_2.py` | — | `DELETE` on a specific hardcoded `(data_version, schema_version)` pair | **excluded, manual** — one-off historical repair for a specific broken publish, no argparse, not idempotent-safe to include in a recurring job |
| `scripts/audit_cxg_e13_f1_f15.py` | BigQuery `oam_core` | none (local diagnostics only as a standalone CLI) | **excluded as a standalone script**, but its module-level constants/functions (`_client`, `_fetch_events`, `_fetch_frames`, `_match_ids`, `DATASET`, etc.) are a real runtime dependency of `materialize_gold_cxg.py` — bundled into the `oam-transform` image |

### Analysis → `oam-analyse`

| Script | Reads | Writes | Verdict |
|---|---|---|---|
| `scripts/materialize_cxg_defensive_involvement.py` | `oam_core` | `oam_analysis.cxg_defensive_involvement_v1` (full refresh, idempotent) | **oam-analyse** step 1 |
| `scripts/materialize_cxg_odi_features.py` | `oam_features`, `oam_core`, `oam_analysis.cxg_defensive_involvement_v1` | `oam_analysis.cxg_odi_features_v1` (`WRITE_TRUNCATE`, idempotent) | **oam-analyse** step 2 — despite "features" in the name, its actual write target is `oam_analysis`, not `oam_features` |
| `scripts/materialize_cxg_defensive_profile_clusters.py` | `oam_analysis.cxg_plus_360_model_matrix_v1` + `oam_features` line-shape/defensive tables | `oam_analysis.cxg_defensive_profile_clusters_v1` (full refresh, idempotent) | **oam-analyse** step 3 — **known pre-existing dependency**: `cxg_plus_360_model_matrix_v1`/`cxg_match_splits_v1` are materialized only by the excluded, manual `run_cxg_split_analysis.py`. They already exist and persist in the live project, so this job succeeds today; a from-scratch environment would need that script run manually first. `run_oam_analyse.py` pre-flight-checks for their existence and fails with an explicit message rather than a silent skip. |
| `scripts/materialize_cxg_opponent_adjusted_analysis.py` | `oam_analysis` (odi/clusters/model-matrix) | `INSERT INTO` 6 `oam_analysis` tables (inventory/null-profile/summary-stats/eda-bins/univariate-target/split-univariate) | **oam-analyse** step 4 — **confirmed NOT idempotent** (plain `INSERT`, no delete-first). Real validation duplicated its rows on a second run; see §6. |
| `scripts/materialize_cxg_opponent_adjusted_chart_registry.py` | `oam_analysis.cxg_chart_registry_v1` (hardcoded old `run_id`) | `INSERT INTO cxg_chart_registry_v1` under a new hardcoded-target `run_id` | **excluded, manual** — bespoke one-off, hardcodes a specific historical `run_id` to copy from. Superseded for automation by a new, generalized script (§2). |
| `scripts/render_cxg_analysis_charts.py` (wraps `cxg_charts.py`) | `oam_analysis` (chart registry + pre-aggregated tables) | HTML/PNG to GCS, `cxg_rendered_chart_registry_v1` | **oam-analyse** step 6 |
| `scripts/render_cxg_feature_eda_appendix.py` | `oam_analysis.cxg_feature_inventory_v1` + analysis surfaces | HTML/PNG to GCS, `cxg_feature_eda_chart_registry_v1` | **oam-analyse** step 7 |

**Deliberately excluded from `oam-analyse`:** `scripts/materialize_cxg_analysis.py` (wraps `CxGAnalysisMaterializer.run()`). Investigation found it unconditionally rebuilds `cxg_correlation_v1` (bivariate) on every run — the exact table `cleanup_to_univariate_state.py` dropped and every prior task in this project has been careful never to reactivate. Flagged to the user before writing any code; **user confirmed: exclude it.** The foundational 6-family EDA/univariate layer it builds is a one-time manual bootstrap, already done, not part of this automated job.

### Excluded, remains manual (confirmed one-off/historical/reverted bivariate track)

All of the following were read and confirmed to hardcode a frozen historical `run_id`, operate on tables the split-policy revert already deleted, and/or have no general-purpose CLI meant for repeated runs against fresh data — none belong in an automated job:

`cleanup_to_univariate_state.py`, `finalize_cxg_feature_selection_report.py`, `finalize_cxg_split_model_freeze.py`, `materialize_cxg_model_specs.py`, `materialize_cxg_result_based_model_specs.py`, `run_cxg_baseline_multivariate.py`, `run_cxg_split_analysis.py`, `render_cxg_baseline_charts.py`, `render_cxg_findings_charts.py`, `render_cxg_findings_detail_charts.py`, `render_cxg_split_analysis_charts.py`, `materialize_cxg_findings_analysis.py`, `materialize_cxg_dashboard_shortlist.py`, `audit_cxg_chart_state.py`, `audit_cxg_context_variable_status.py`, `audit_cxg_odi_defprofile_spike.py`, `materialize_cxg_analysis.py` (see above), `materialize_cxg_opponent_adjusted_chart_registry.py` (see above), `repair_oam_core_v1_2.py` (see above).

---

## 2. New orchestration code (no existing module's logic modified)

| File | Purpose |
|---|---|
| `scripts/run_oam_transform.py` | `oam-transform` entrypoint: chains Silver → oam_core → Gold shot-features → Gold family tables. Routes around the confirmed `DATASET` divergence (§1) by importing `materialize_gold_cxg` normally, then overriding **only its own** `DATASET`/`TABLE` module attributes (not the shared `audit_cxg_e13_f1_f15.DATASET`, which must stay `"oam_core"` for that module's own read functions — an earlier version of this patch broke that and was caught during real validation, see §6). |
| `scripts/run_oam_analyse.py` | `oam-analyse` entrypoint: pre-flight-checks `cxg_match_splits_v1`/`cxg_plus_360_model_matrix_v1` exist, then chains involvement → ODI → clusters → opponent_adjusted analysis → chart registration → both chart renderers. Generates a fresh `run_id` (`cxg-analysis-<UTC ISO>`) by default. |
| `scripts/register_chart_registry_for_run.py` | Generalizes the one-off copy-forward pattern from `materialize_cxg_opponent_adjusted_chart_registry.py`: auto-detects the latest existing `run_id` in `cxg_chart_registry_v1` (by `MAX(materialized_at)`) and copies it forward to a new `run_id`, **dropping any row whose backing table no longer exists** (the exact `cxg_correlation_v1` situation handled manually in the prior chart-coverage task). Uses the same tested delete-then-insert-scoped-by-run_id pattern as `CxGChartRenderer._materialize_render_registry`. |

None of these modify any existing module — they only import and call, or `subprocess.run` the existing tested CLIs.

---

## 3. Containerization

Three separate Dockerfiles (one per job — clearest match to the Cloud Run Jobs model, avoids ambiguity about which entrypoint runs), `python:3.11-slim` base, built from repo root as build context.

| Image | Dependencies (checked via actual imports, not assumed from `pyproject.toml`) | Size |
|---|---|---|
| `docker/oam-ingest/Dockerfile` | `google-cloud-storage` only | 251MB |
| `docker/oam-transform/Dockerfile` | `pyarrow`, `google-cloud-bigquery`, `google-cloud-storage`, plus `git` (see §6) | 560MB |
| `docker/oam-analyse/Dockerfile` | `google-cloud-bigquery`, `google-cloud-storage`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `mplsoccer`, `plotly`, plus `pyarrow` (transitive, see §6) | 1.15GB |

**Finding:** `pyproject.toml` only declares `pyarrow`/`google-cloud-storage`/`google-cloud-bigquery` — the analysis layer's real dependencies (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `mplsoccer`, `plotly`) are undeclared there (available in this dev machine's conda environment but not in the project's own dependency manifest). Each Dockerfile's `requirements.txt` reflects what each job's code *actually* imports, verified by `grep` before writing.

All three images pushed to the existing `oam-containers` Artifact Registry repo (`europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers`) — no new repo created. `immutable_tags=true` on that repo meant each real-validation fix required a new tag (`oam-ingest:v1`, `oam-transform:v3`, `oam-analyse:v2` — see §6 for what each fix addressed).

---

## 4. Terraform

New/extended files under `infra/terraform/`:

| File | Content |
|---|---|
| `pipeline_jobs.tf` (new) | 3 `google_cloud_run_v2_job` resources, plus IAM: `google_bigquery_dataset_iam_member` (`roles/bigquery.dataEditor`) scoped individually to `oam_core`/`oam_features`/`oam_analysis`; `google_project_iam_member` (`roles/bigquery.jobUser`) — BigQuery has no dataset-scoped equivalent of "may run a query job", this is the narrowest role that grants it; `google_storage_bucket_iam_member` (`roles/storage.objectAdmin`) scoped individually to the data and artifacts buckets |
| `orchestration.tf` (new) | `workflows.googleapis.com` service enablement, `google_workflows_workflow.oam_pipeline`, plus IAM: per-job `google_cloud_run_v2_job_iam_member` (`roles/run.invoker`) to start each execution, **and** a project-level `google_project_iam_member` (`roles/run.developer`) — added only after a real 403 during validation proved the Cloud Run long-running-operation/execution-polling resources (`projects/*/locations/*/operations/*`) aren't nested under any single job, so IAM can't be scoped to them per-resource (§6) |
| `data_foundation.tf` (extended) | Added `oam_analysis` to the `bigquery_datasets` map — **management gap found**: this dataset already existed live (created programmatically by `_ensure_analysis_dataset()` in `cxg.py`) but was never declared in Terraform. Imported the existing resource (`terraform import`) before applying, so it was recognized as an in-place metadata update, not a duplicate create. |
| `variables.tf` (extended) | Added `pipeline_data_version` (defaults to the pinned SHA used everywhere else in the project) |
| `outputs.tf` (extended) | Added `cloud_run_job_names` and `pipeline_workflow_name` outputs |
| `infra/workflows/oam_pipeline.yaml` (new) | The Workflows source (see §5) |

All 3 jobs run as the existing `oam-pipeline-sa` — **no new service account created**. No project-wide `roles/bigquery.dataEditor` or `roles/storage.objectAdmin` — every one of those is scoped to a specific dataset or bucket. `roles/bigquery.jobUser` and `roles/run.developer` are the two project-level exceptions, both because GCP's IAM model has no narrower option for what they grant (documented inline in the `.tf` files with the exact error that proved it, not asserted).

`terraform plan`/`apply` run for real (terraform binary wasn't in the environment initially — downloaded `v1.15.5`, matching the repo's `required_version` constraint, added to PATH). All resources created successfully; **0 pre-existing resources destroyed or recreated**.

---

## 5. Workflows chain design

`infra/workflows/oam_pipeline.yaml`: `main` calls a `run_job_to_completion` subroutine three times (ingest → transform → analyse), each of which:
1. Calls `googleapis.run.v2.projects.locations.jobs.run` to start the execution (this call itself blocks on the long-running operation and raises on failure — the primary failure-detection path, confirmed by real testing, see §6).
2. Additionally polls `googleapis.run.v2.projects.locations.jobs.executions.get` every 15s until `completionTime` is set, as a defensive fallback.
3. Checks `succeededCount >= 1 and failedCount == 0`; if not, `raise`s a message containing the job name, execution name, and `logUri` — **the failing job's identity and log location are always in the error**, not a generic "something failed."

No Cloud Scheduler trigger was added — the task explicitly required confirming with the user first whether StatsBomb's open data changes on a cadence that justifies automatic re-ingestion, and that confirmation was never sought/given, so this stays **manual invocation only** (`gcloud workflows run oam-pipeline --location=europe-west2`).

**No step references, stubs, or prepares for a model/training job.** The workflow's `main` routine terminates at `done` immediately after `run_analyse` returns.

---

## 6. What real end-to-end validation actually found (not just "it ran")

Individual job dry-runs and a full chain run were both executed for real against production GCP — 4 real bugs were found and fixed along the way, each with before/after evidence:

| # | Bug found | Evidence | Fix |
|---|---|---|---|
| 1 | `oam-analyse:v1` missing `pyarrow` (transitive: `odi/contracts.py` → `pipelines/silver/contracts.py`) | Real execution failed: `ModuleNotFoundError: No module named 'pyarrow'` after the first step (`materialize_cxg_defensive_involvement`) succeeded | Added `pyarrow` to `docker/oam-analyse/requirements.txt`, rebuilt as `v2` |
| 2 | Workflow IAM: `run.operations.get` denied | Real `gcloud workflows run` failed with 403 on `projects/*/locations/*/operations/*` — not a bindable per-job resource | Added project-level `roles/run.developer` (§4) |
| 3 | `oam-transform:v1` missing `git` binary + no resolvable commit (`build_statsbomb_silver.py::_git_sha()`) | Real execution failed: `FileNotFoundError: 'git'` | Installed `git` in the Dockerfile; since this checkout's real `.git` is a worktree pointer to a sibling repo outside the Docker build context (can't be `COPY`'d in), synthesized a single-commit repo at build time so `git rev-parse HEAD` resolves to a valid SHA (documented as a deliberate tradeoff, not the source repo's real commit) — `v2` |
| 4 | `run_oam_transform.py`'s own DATASET-routing patch (§1/§2) was overwriting the **shared** `audit_cxg_e13_f1_f15.DATASET` constant instead of only `materialize_gold_cxg`'s copied binding, breaking that module's own `oam_core` reads | Real execution failed: `404 Not found: Table oam_features.events` (the patch had redirected `_match_ids`'s read target too) | Rewrote the patch to only touch `materialize_gold_cxg.DATASET`/`.TABLE` after import, leaving `audit_cxg_e13_f1_f15.DATASET` at `"oam_core"` — `v3` |

Every fix above was found by running the real container against real GCP, not by inspection — this is the actual value of the individual dry-runs the task asked for.

### Real execution results (final state)

- **`oam-ingest`**: real execution, succeeded. Confirmed idempotent — 782 objects under the raw landing prefix before and after, **0 objects touched** in the run window (create-only semantics working as designed).
- **`oam-transform`**: 
  - Silver step correctly **refused** to re-publish (immutability guard: `RuntimeError: Silver output already published and immutable`) — this is the expected, safe behavior for the already-canonical prefix, not a bug.
  - `oam_core` publish step: real execution, all 18 tables `skipped_existing` (byte-identical row-count reconciliation against the manifest, matching the pattern from the Silver-acceptance task).
  - Gold materialization: real execution, recomputed features for all shots. Confirmed row counts after: `cxg_shot_features`=15,737, `cxg_shot_base_features`=15,737, `cxg_event_context_features`=15,737, `cxg_defensive_360_features`=3,960, `cxg_line_shape_360_features`=3,960, `cxg_training_matrix_v1` (view)=15,737 — all exactly matching expected cohort sizes.
- **`oam-analyse`**: real execution, succeeded end-to-end. GCS confirmed via direct listing (not just trusting the upload call): `rendered_charts/` = 55 objects (27 HTML + 27 PNG + 1 manifest), `eda_appendix/` = 317 objects (158 HTML + 158 PNG + 1 manifest), **0 zero-size objects**.
- **Full Workflows chain**: run twice. First run surfaced the two IAM gaps above (fixed). Final run: `run_ingest` succeeded, `run_transform` correctly failed at the Silver immutability guard and the workflow **stopped** — proving the "fail loudly, don't silently continue" requirement with a real failure, not a synthetic one.

### Non-idempotency finding, caught and cleaned up

Running `oam-analyse` for real a second time (for validation) exposed that `materialize_cxg_opponent_adjusted_analysis.py` uses a plain `INSERT` with no delete-first guard. The `opponent_adjusted` family's rows duplicated across 6 tables (e.g. `cxg_feature_inventory_v1`: 4→8 rows for that family). **Flagged to the user before touching anything** (the classifier also blocked the cleanup `DELETE` pending explicit approval); user approved the scoped cleanup. Ran `DELETE ... WHERE feature_family = "opponent_adjusted"` across the 6 affected tables (only those rows, no other family touched), then re-ran the materializer once to restore the correct single copy. Verified restored exactly to the pre-duplication baseline (198/198/158/2689/158/438 total rows; `opponent_adjusted` back to 4/4/4/65/4/9). This non-idempotency is a real, pre-existing property of that script (not modified, per the task's constraint) — documented here as a known limitation for any future re-run of `oam-analyse` outside this validation.

### Row-count / row-count-delta summary (analysis tables, before → after the validated real run + cleanup)

| Table | Before this task | After (post-cleanup) | Delta |
|---|---|---|---|
| `cxg_feature_inventory_v1` | 198 | 198 | 0 |
| `cxg_null_profile_v1` | 198 | 198 | 0 |
| `cxg_summary_stats_v1` | 158 | 158 | 0 |
| `cxg_eda_distribution_bins_v1` | 2,689 | 2,689 | 0 |
| `cxg_univariate_target_v1` | 158 | 158 | 0 |
| `cxg_split_univariate_v1` | 438 | 438 | 0 |
| `cxg_odi_features_v1` | 3,960 | 3,960 | 0 |
| `cxg_defensive_involvement_v1` | 15,445 | 15,445 | 0 |
| `cxg_defensive_profile_clusters_v1` | 3,960 | 3,960 | 0 |
| `cxg_chart_registry_v1` | 51 | 78 | +27 (one new registered run_id, additive, no other run_id touched) |

Every table other than `cxg_chart_registry_v1` (whose growth is the intended, additive new-run_id registration) ends at exactly the same row count it started at — the transient duplication was fully cleaned up, and no other family's rows were touched at any point (confirmed via `feature_family`-scoped counts throughout).

### Full test suite

`python -m pytest -q` → **208 passed** at every checkpoint during this task (baseline, mid-work, and final) — no regression. No existing analysis/feature/pipeline Python module's logic was modified.

---

## 7. Explicit confirmation: no model job anywhere

- `infra/terraform/pipeline_jobs.tf`: exactly 3 `google_cloud_run_v2_job` resources (`oam_ingest`, `oam_transform`, `oam_analyse`). No 4th resource.
- `infra/terraform/orchestration.tf` / `infra/workflows/oam_pipeline.yaml`: the workflow calls exactly 3 jobs and terminates at `done`. No step, variable, or comment references a model, training, evaluation, promotion, or serving job.
- `scripts/run_oam_analyse.py`: its last step is chart rendering. Nothing downstream of it is built, stubbed, or referenced.
- Modeling has not started in this project (baselines haven't been built); the user's own words were that they personally review `oam-analyse`'s output before anything touches modeling. Nothing in this task changes that — the chain still terminates after charts land.

---

## File paths

- Dockerfiles: `docker/oam-ingest/Dockerfile`, `docker/oam-transform/Dockerfile`, `docker/oam-analyse/Dockerfile` (+ each's `requirements.txt`)
- New orchestrator scripts: `scripts/run_oam_transform.py`, `scripts/run_oam_analyse.py`, `scripts/register_chart_registry_for_run.py`
- Terraform: `infra/terraform/pipeline_jobs.tf` (new), `infra/terraform/orchestration.tf` (new), `infra/terraform/data_foundation.tf` / `variables.tf` / `outputs.tf` (extended)
- Workflows source: `infra/workflows/oam_pipeline.yaml`
- Images: `europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/{oam-ingest:v1, oam-transform:v3, oam-analyse:v2}`
- This report: `docs/pipeline_orchestration_report.md`
