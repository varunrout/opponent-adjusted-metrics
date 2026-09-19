# Task: Build Real GCP Orchestration for the ETL + Analysis Pipeline (3 Jobs, Not 4)

## Context and goal

Everything in this project currently runs as manually-invoked Python scripts (via Claude Code / local CLI). Nothing in GCP triggers itself. The Terraform already shows the intent: `infra/terraform/runtime_identity.tf` provisions `oam-pipeline-sa` with a description naming "Cloud Run jobs for ingestion, feature builds, training, evaluation, promotion, serving builds" — but no actual Cloud Run Job, Cloud Workflows, or Cloud Scheduler resource exists yet. `infra/terraform/artifact_registry.tf` already provisions an Artifact Registry repo (`oam-containers`, DOCKER format, immutable tags) and `services.tf` already enables `run.googleapis.com`/`artifactregistry.googleapis.com`. Build on top of what's there, don't duplicate it.

**Goal: three chained Cloud Run Jobs, orchestrated by Cloud Workflows, triggered by completion (not a clock):**

1. **`oam-ingest`** — raw StatsBomb open data → Bronze (GCS)
2. **`oam-transform`** — Bronze → Silver → Gold (BigQuery `oam_core` + `oam_features`)
3. **`oam-analyse`** — Gold → EDA/univariate analysis tables + charts (BigQuery `oam_analysis` + GCS chart artifacts)

**Explicitly out of scope: a 4th "model" job.** Modeling doesn't exist yet in this project (baselines haven't started) and must never auto-fire — the user's own words: after `oam-analyse` produces charts/tables, they personally review and decide what to select before anything touches modeling. Do not build, stub, or even name a model job/trigger. The Workflows chain terminates after `oam-analyse`.

## Step 1: Investigate real script I/O before designing anything — do not guess from filenames

This repo has many scripts (`scripts/*.py`) and it would be a mistake to assign them to jobs based on name-guessing. Read each of the following and determine its actual CLI args, inputs (what tables/files it reads), outputs (what it writes), and true dependency order:

- Ingestion candidates: `scripts/fetch_statsbomb_subset.py`, `src/opponent_adjusted/ingestion/subset_fetch.py`, `src/opponent_adjusted/ingestion/statsbomb_source.py`
- Silver/Gold candidates: `scripts/build_statsbomb_silver.py`, `scripts/publish_oam_core.py`, `scripts/materialize_gold_cxg.py`, `scripts/materialize_cxg_feature_family_tables.py`, `scripts/repair_oam_core_v1_2.py` (check if this is a one-off historical repair, not part of the regular pipeline — likely excluded from the job, confirm)
- Analysis candidates: `scripts/materialize_cxg_analysis.py`, `scripts/render_cxg_analysis_charts.py`, `scripts/render_cxg_feature_eda_appendix.py`

**Important boundary question to resolve during investigation, don't assume:** scripts like `scripts/materialize_cxg_odi_features.py`, `scripts/materialize_cxg_defensive_involvement.py`, `scripts/materialize_cxg_defensive_profile_clusters.py`, `scripts/materialize_cxg_opponent_adjusted_analysis.py`, `scripts/materialize_cxg_opponent_adjusted_chart_registry.py` sit ambiguously between "Gold feature engineering" (ODI/cluster labels are features) and "Analysis" (their outputs get charted). Determine which BigQuery dataset each one's *primary output* lands in (`oam_features` = belongs in `oam-transform`; `oam_analysis` = belongs in `oam-analyse`) and assign accordingly — do not put a script in both jobs, and do not guess based on the word "analysis" in a filename alone (e.g. `materialize_cxg_opponent_adjusted_analysis.py` may still be a Gold-layer feature step despite the name — check its actual `CREATE TABLE`/`INSERT` targets).

**Also check:** `scripts/cleanup_to_univariate_state.py`, `scripts/finalize_cxg_feature_selection_report.py`, `scripts/finalize_cxg_split_model_freeze.py`, `scripts/materialize_cxg_model_specs.py`, `scripts/materialize_cxg_result_based_model_specs.py`, `scripts/run_cxg_baseline_multivariate.py`, `scripts/run_cxg_split_analysis.py`, `scripts/render_cxg_baseline_charts.py`, `scripts/render_cxg_findings_charts.py`, `scripts/render_cxg_findings_detail_charts.py`, `scripts/render_cxg_split_analysis_charts.py`, `scripts/materialize_cxg_findings_analysis.py`, `scripts/materialize_cxg_dashboard_shortlist.py` — these look like one-off/historical/bivariate-track scripts run manually during specific past sessions (some reference reverted bivariate work per `docs/cxg_split_policy_and_parallel_plan.md`'s status note). **These almost certainly do NOT belong in any of the 3 automated jobs** — confirm this by checking what they do, and explicitly list them as "excluded, remains manually-invoked" in your report rather than silently omitting them.

Write a short inventory table (script → real inputs/outputs → assigned job or "excluded, manual") before writing any Dockerfile or Terraform. Show this to confirm your assignment is right before proceeding — this determines the shape of every container that follows.

## Step 2: Containerize each job

- One `Dockerfile` per job (or one multi-stage Dockerfile producing 3 images — your call, but keep image size sane; this project's dependencies are `pyarrow`, `google-cloud-storage`, `google-cloud-bigquery`, per `pyproject.toml`, nothing exotic).
- Each container's entrypoint should accept the project's existing parameter conventions as env vars or CLI args: `--data-version`, `--silver-schema-version` (transform only), `--feature-version` (transform/analyse), `--run-id` (analyse — use the existing UTC-timestamp convention, e.g. `cxg-analysis-<UTC ISO>`), matching how these are already passed to the underlying scripts today. Don't invent a new parameter scheme.
- Push to the existing Artifact Registry repo (`oam-containers`, `europe-west2`) — don't create a second repo.
- Local build/test: confirm each image builds and the entrypoint at least runs `--help`/argument validation successfully before wiring up Cloud Run — don't ship an untested container.

## Step 3: Terraform for the 3 Cloud Run Jobs

Add new `.tf` files (follow the existing file-per-concern convention: `runtime_identity.tf`, `artifact_registry.tf`, `services.tf`, `data_foundation.tf`, `variables.tf`, `outputs.tf` already exist — add e.g. `pipeline_jobs.tf`, `orchestration.tf`).

- `google_cloud_run_v2_job` resources for `oam-ingest`, `oam-transform`, `oam-analyse`. Use `oam-pipeline-sa` (already provisioned) as the runtime service account — do not create a new one.
- Confirm `oam-pipeline-sa`'s current IAM bindings (check for existing `google_project_iam_member`/`google_bigquery_dataset_iam_member` resources) and add whatever's missing: BigQuery Data Editor scoped to `oam_core`/`oam_features`/`oam_analysis` (not project-wide `roles/bigquery.dataEditor` unless that's already the existing pattern — match whatever precedent exists), GCS object admin scoped to the raw/silver/artifacts buckets/prefixes this pipeline touches, not broader.
- Enable `run.googleapis.com` (already enabled) — add `workflows.googleapis.com` and, only if you end up needing scheduled ingestion, `cloudscheduler.googleapis.com`, following the existing `google_project_service` resource pattern in `services.tf`.

## Step 4: Cloud Workflows orchestration

- One Workflows definition chaining `oam-ingest` → `oam-transform` → `oam-analyse`, triggered on completion (each step calls the next only after the prior job's execution succeeds — use the Workflows Cloud Run connector's execution-status polling, not a blind fire-and-continue).
- The workflow should be callable manually (via `gcloud workflows run` or Console) for now. **Do not add a Cloud Scheduler trigger unless you first confirm with the user whether StatsBomb's open data actually changes on a cadence that justifies automatic re-ingestion** — default to manual invocation only.
- The workflow must terminate cleanly after `oam-analyse` succeeds. No step should reference, stub, or prepare for a model job.
- On any step failure, the workflow should fail loudly (surface the failing job's logs/execution ID) rather than silently continuing to the next step.

## Step 5: Validate end-to-end

- Dry-run each Cloud Run Job individually first (`gcloud run jobs execute --dry-run` if supported, or a real execution against a scratch/test scope if safe) before wiring the full chain.
- Run the full Workflows chain once, end-to-end, and confirm: `oam_core`/`oam_features` tables land correctly, `oam_analysis` tables + GCS chart artifacts land correctly, using the project's existing verification patterns (row-count reconciliation, `_SUCCESS`-style completion signals if applicable, GCS byte-size checks — matching the discipline already used in this project's Silver/chart-publication tasks).
- Confirm `python -m pytest -q` still passes (baseline is 208 as of the last check) — this task shouldn't touch any existing analysis/feature code, only add orchestration around it.

## Step 6: Report

Write `docs/pipeline_orchestration_report.md` covering:
- The script-to-job inventory table from Step 1, including the explicit "excluded, remains manual" list
- Dockerfile/image structure chosen and why
- Terraform resources added, IAM bindings applied (with reasoning for scope)
- Workflows chain design, how failure is surfaced
- End-to-end validation results (real row counts / GCS confirmations from the test run, not just "it ran")
- Explicit confirmation: no model job or model-triggering step was built, stubbed, or referenced anywhere

## What NOT to do

- Do not build, stub, or name a 4th "model"/"training" job or any Workflows step that could fire one — this is the hardest boundary in this task, respect it exactly.
- Do not add a Cloud Scheduler recurring trigger without first surfacing the question to the user — default to manual invocation of the Workflows chain.
- Do not create a new service account, new Artifact Registry repo, or duplicate any Terraform resource that already exists — extend what's there.
- Do not assign a script to a job based on its filename alone — verify its actual BigQuery write target first (Step 1 is not optional busywork, it determines correctness).
- Do not grant project-wide IAM roles where dataset/bucket-scoped roles would do — match or tighten the existing precedent, don't loosen it.
- Do not modify any existing analysis/feature/pipeline Python module's logic — this task is purely containerization + orchestration around what already exists and is tested.

Report back with a summary and the file paths (Dockerfiles, new `.tf` files, Workflows YAML) when complete.
