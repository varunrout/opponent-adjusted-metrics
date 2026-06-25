# Completion Audit

This audit records the current repository state after the documentation and generated-output cleanup.

## Current Judgement

`opponent-adjusted-metrics` has a reproducible CxG baseline path, CI quality gates, fixture-backed ingestion tests, feature contracts, and API-backed CxG prediction from exported artifacts. It should not be described as a complete v1 football analytics product.

Generated reports and model outputs are intentionally not committed. They must be regenerated locally from source code and configs.

## Status By Area

| Area | Status | Evidence / notes | Completion need |
| --- | --- | --- | --- |
| Repository structure | Mostly complete | Package layout, scripts, tests, docs, configs, and migrations exist. | Keep release discipline and avoid recommitting generated outputs. |
| Dependency management | Mostly complete | Poetry config exists and validation commands run through Poetry. | Keep lock/install behavior covered by CI. |
| Database schema | Mostly complete | SQLAlchemy models and Alembic migrations exist. | Broaden Postgres smoke coverage. |
| StatsBomb ingestion | Mostly complete | Ingestion foundation and fixture-backed tests exist. | Prove full fresh-clone ingestion against Docker Postgres. |
| Feature engineering | Mostly complete for CxG, partial for CxA/CxT | Feature contracts exist for all metric families; CxG has the stable runner. | Enforce contracts across unfinished CxA/CxT paths. |
| CxG modelling | Baseline complete | `scripts/run_cxg_pipeline.py` and `scripts/run_cxg_end_to_end.py` build features, train, validate, and export artifacts. | Future calibration, monitoring, and richer slice validation. |
| CxG outputs | Reproducible, not tracked | `outputs/` and `feature_store/` are ignored by Git. | Regenerate outputs when needed; commit only curated small examples by explicit decision. |
| CxA modelling | Not complete | Planning and contracts exist; old generated/completion reports were removed. | Complete reproducible sequence scoring, validation, aggregates, and model card. |
| CxT modelling | Not complete | Leakage guardrail note exists; old generated/completion reports were removed. | Complete leakage-sensitive validation and regenerate final outputs. |
| API | Mostly complete for CxG prediction, partial overall | `/predict/cxg` can load the emitted CxG artifact. | Broader registry-backed aggregate serving and API coverage. |
| Dashboard | Not complete | Dashboard code exists, but it is not a stable v1 output surface. | Wire dashboard to stable regenerated outputs and add smoke coverage/screenshots. |
| CI/CD | Mostly complete | Quality gates exist for lint, formatting, typing, and tests. | Expand database/release workflows. |
| Documentation | Clean source-of-truth set | Historical generated reports and obsolete summaries were removed from `docs/`. | Keep docs aligned with implemented behavior. |

## Not Complete Yet

The repository is not v1 complete until:

1. Full fresh-clone ingestion is validated against Docker Postgres.
2. CxG calibration/monitoring refinements are complete beyond the baseline runner.
3. Registry-backed aggregate serving is finalized.
4. CxA has reproducible scoring, validation, aggregate outputs, and model-card documentation.
5. CxT has leakage-sensitive validation, regenerated outputs, and model-card documentation.
6. Dashboard views run against stable regenerated outputs.
7. v1 release notes and packaging are published.

## Documentation Cleanup Decision

Historical generated reports, obsolete implementation summaries, old modelling result summaries, and documents claiming CxA/CxT completion were removed from `docs/`. The remaining docs should either describe current reproducible behavior or clearly identify future/planned work.

## Completion Definition

The repository reaches v1 complete when a reviewer can:

1. Clone the repo.
2. Start Postgres with Docker Compose.
3. Run migrations.
4. Fetch a StatsBomb subset.
5. Ingest and normalise events.
6. Build feature tables.
7. Train and validate CxG with the baseline runner.
8. Regenerate CxG artifacts and outputs.
9. Query real CxG predictions through the API.
10. Run completed CxA and CxT pipelines without unresolved validation warnings.
11. Open a dashboard backed by stable regenerated outputs.
12. See CI passing for lint, tests, type checks, and smoke tests.
13. Read model cards and limitations without stale or contradictory claims.
