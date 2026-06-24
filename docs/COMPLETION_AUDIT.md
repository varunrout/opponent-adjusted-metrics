# Completion Audit

This audit records the current project state and the work required to make `opponent-adjusted-metrics` complete, reproducible, and portfolio-ready.

## Completion definition

The project is complete when a fresh clone can:

1. Start the local database.
2. Run migrations.
3. Fetch or locate the configured StatsBomb Open Data subset.
4. Ingest competitions, matches, events, and normalised detail tables without duplicate rows.
5. Build model-ready feature tables for CxG, CxA, and CxT.
6. Train and evaluate CxG, CxA, and CxT with leakage checks and grouped validation.
7. Generate model artefacts, metadata, reports, charts, and player/team aggregates.
8. Serve real model-backed API responses.
9. Run tests, linting, type checks, security checks, and smoke checks in CI.
10. Present a clear dashboard and README with honest limitations.

## Current status summary

| Area | Status | Notes |
| --- | --- | --- |
| Repository structure | Partial | The core package exists, but status documents are stale and some completion claims are too broad. |
| Dependency management | Mostly complete | Poetry is configured. CI should validate dependency installation and lock consistency. |
| Database schema | Mostly complete | SQLAlchemy and Alembic are present. Needs migration smoke tests in CI. |
| Local database | Partial | Docker Compose support exists. Needs documented end-to-end smoke path. |
| Data ingestion | Partial | Scripts exist, but completion requires idempotency checks, row-count reports, and resume-safe execution. |
| Normalisation | Partial | Event normalisation exists, but needs documented coverage and fixture tests across event types. |
| Feature generation | Partial | Feature scripts exist. Needs formal feature contracts and data quality checks. |
| CxG modelling | Partial | Modelling artefacts and reports exist, but the completion path should be standardised into one runnable pipeline. |
| CxG neutralisation | Partial | Methodology is documented, but outputs should be generated, tested, and exposed through API. |
| CxA modelling | Partial | Analysis and modelling work exist. Needs a final public methodology name, reproducible pipeline, and leaderboard outputs. |
| CxT modelling | Needs remediation | Current reporting flags potential leakage. This must be fixed before release. |
| API | Partial | Health and aggregate endpoints exist. CxG prediction currently needs a real model-backed path. |
| Dashboard | Partial | Dashboard work exists, but it needs a clean run path, screenshots, and integration with generated outputs. |
| Evaluation | Partial | Reports exist. Needs one standard evaluation protocol used by all metrics. |
| Tests | Partial | Tests exist. Coverage needs to include ingestion, features, modelling contracts, API, and smoke paths. |
| CI/CD | Missing | No GitHub Actions workflows are currently part of the repo. |
| Documentation | Partial | README and status docs exist, but they need to be made consistent and honest. |

## Main risks

### 1. Documentation drift

Several documents describe the project as infrastructure-complete, while later files indicate modelling, prediction, dashboard, and analysis work has expanded. The repo needs one truthful project state.

### 2. API placeholder

The `/predict/cxg` endpoint currently needs a model-backed implementation before the API can be described as complete.

### 3. CxT leakage

The CxT report flags a feature leakage risk. The completion plan must remove leakage-prone features from the completion model and regenerate CxT evaluation.

### 4. Reproducibility gap

The repo has many scripts and reports, but a reviewer needs one reliable route from fresh clone to usable outputs.

### 5. Portfolio readability

The project has strong technical content, but it needs a clearer story: what the system does, what metrics are implemented, what results were achieved, and what limitations remain.

## Required PR sequence

1. Completion audit and roadmap.
2. CI/CD and repository hygiene.
3. Reproducible data ingestion.
4. Feature store and quality checks.
5. CxG end-to-end completion.
6. CxA completion.
7. CxT leakage fix and completion.
8. API prediction completion.
9. Dashboard and football storytelling layer.
10. Final release packaging.

## Release readiness checklist

- [ ] Fresh clone quickstart tested.
- [ ] Docker Postgres starts reliably.
- [ ] Alembic migrations run in CI.
- [ ] StatsBomb subset ingestion is idempotent.
- [ ] Feature contracts exist for CxG, CxA, and CxT.
- [ ] CxG produces trained artefacts, neutralised scores, aggregates, and reports.
- [ ] CxA produces action-chain outputs and player leaderboards.
- [ ] CxT is retrained without leakage-prone features.
- [ ] API returns real predictions.
- [ ] Dashboard displays generated outputs.
- [ ] CI runs tests, linting, formatting checks, type checks, and security checks.
- [ ] README includes architecture, run path, outputs, limitations, and screenshots.
- [ ] `v1.0.0` release notes are written.
