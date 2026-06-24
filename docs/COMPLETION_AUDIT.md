# Completion Audit

This audit separates what is already usable from what still needs completion before the repository can be described as a complete opponent-adjusted football analytics project.

## Current judgement

`opponent-adjusted-metrics` has a strong foundation, but it is not yet complete as an end-to-end project. The repo contains meaningful architecture, modelling work, reports, scripts and dashboard material, but the project story is inconsistent and several interfaces still need production-grade wiring.

The immediate completion objective is not to add unrelated features. The objective is to make the existing system reproducible, tested, documented and runnable from a fresh clone.

## Status by area

| Area | Status | Evidence / notes | Completion need |
| --- | --- | --- | --- |
| Repository structure | Mostly complete | Package layout, scripts, docs, tests and outputs exist. | Add CI, contribution docs, release discipline and consistent status docs. |
| Dependency management | Mostly complete | Poetry config exists. | Ensure lock/install behaviour is validated in CI. |
| Database schema | Mostly complete | SQLAlchemy models and Alembic migration exist. | Add migration smoke test against Postgres in CI. |
| StatsBomb ingestion | Partial | Fetch, ingest and normalise scripts exist. | Prove fresh database ingestion works end to end and is idempotent. |
| Feature engineering | Partial to mostly complete | CxG/CxA/CxT feature modules exist. | Add feature contracts, leakage checks and fixture-based tests. |
| CxG modelling | Partial to mostly complete | CxG modelling artifacts and reports exist. | Consolidate training, evaluation, neutralisation, registry and API inference into one reproducible flow. |
| CxA modelling | Partial | CxA analysis and newer methodology docs exist. | Choose final public methodology, complete action-sequence scoring and player aggregation. |
| CxT modelling | Partial | CxT evaluation exists. | Resolve leakage warning, retrain/evaluate and update report. |
| API | Partial | Health/model/aggregate routes exist, but CxG prediction still returns a placeholder `501`. | Implement real artifact loading and feature-contract inference. |
| Dashboard | Partial | Dashboard files and requirements exist. | Wire to stable outputs, add screenshots and dashboard smoke test. |
| Reports | Partial | Several generated reports exist. | Standardise report locations, remove stale claims and add final model cards. |
| Testing | Partial | Unit and e2e tests exist. | Add CI coverage for lint, formatting, typing, unit tests, DB smoke and API smoke. |
| CI/CD | Missing | No workflow files were found during audit. | Add GitHub Actions workflows. |
| Documentation | Inconsistent | README and project status overstate completion in places. | Rewrite around a truthful v1 completion roadmap. |

## Blocking issues before calling the project complete

1. Prediction API still contains a placeholder CxG endpoint.
2. CxT report flags potential leakage from `xt_delta` in the completion model.
3. Project status documentation is stale and conflicts with later modelling artifacts.
4. CI/CD is not present, so reproducibility is not enforced.
5. Feature contracts are not yet treated as first-class artifacts.
6. Fresh-clone instructions need to be verified by automated smoke tests.
7. README should distinguish implemented, partial and planned metric layers.

## Completion definition

The repository is complete when a reviewer can:

1. Clone the repo.
2. Start Postgres with Docker Compose.
3. Run migrations.
4. Fetch a StatsBomb subset.
5. Ingest and normalise events.
6. Build feature tables.
7. Train and evaluate CxG.
8. Generate neutralised/opponent-adjusted predictions.
9. Train or run completed CxA and CxT pipelines.
10. Query real model outputs through the API.
11. Open a dashboard backed by generated outputs.
12. See CI passing for lint, tests, type checks and smoke tests.
13. Read model cards and limitations without stale or contradictory claims.

## Recommended completion order

1. Documentation audit and roadmap.
2. CI/CD and repository hygiene.
3. Data ingestion reproducibility.
4. Feature contracts and data-quality gates.
5. CxG end-to-end completion.
6. API inference completion.
7. CxA completion.
8. CxT leakage fix and completion.
9. Dashboard/reporting completion.
10. v1 release packaging.
