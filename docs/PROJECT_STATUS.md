# Project Status

This document is the source of truth for the current completion state of `opponent-adjusted-metrics`.

## Current status

The repository has a strong foundation for an opponent-adjusted football analytics system, but it should not yet be described as fully complete or production-ready.

The project currently contains:

- Python package structure for football analytics workflows.
- PostgreSQL-oriented database models and migrations.
- StatsBomb ingestion and normalisation scripts.
- CxG, CxA and CxT modelling/analysis modules.
- API routes for health, model metadata and aggregates.
- Dashboard/reporting material.
- Unit and e2e-style tests.

The project still needs final completion work in CI/CD, reproducibility, model-contract validation, API inference, CxT leakage resolution and documentation consistency.

## Implementation status matrix

| Area | Status | Notes |
| --- | --- | --- |
| Project structure | Mostly complete | Good package and script layout exists. |
| Dependency management | Mostly complete | Poetry is configured. CI install validation still needed. |
| Database schema | Mostly complete | SQLAlchemy/Alembic foundation exists. Needs automated Postgres smoke test. |
| Data ingestion | Partial | Scripts exist, but fresh-clone reproducibility and idempotency need verification. |
| Event normalisation | Partial | Normalisation path exists. Needs row-count and integrity reporting. |
| Feature engineering | Partial | Feature modules exist. Needs formal contracts and leakage gates. |
| CxG modelling | Mostly complete | One-command end-to-end training/evaluation/export now exists; further calibration and production monitoring remain future work. |
| CxG neutralisation | Mostly complete | End-to-end runner exports raw, neutral and opponent-adjusted shot scores plus player/team aggregates. |
| CxA modelling | Partial | Analysis and methodology exist. Needs final public methodology and reproducible sequence scoring. |
| CxT modelling | Partial | Evaluation exists, but leakage warning must be resolved. |
| API | Partial | Health/model/aggregate routes exist and `/predict/cxg` can load the emitted CxG artifact; registry-backed aggregate serving still needs completion. |
| Dashboard | Partial | Dashboard material exists. Needs stable output contracts and screenshots. |
| Tests | Partial | Tests exist. Needs CI enforcement and broader smoke coverage. |
| CI/CD | Missing | GitHub Actions workflows need to be added. |
| Documentation | Partial | Documentation is extensive but inconsistent in completion claims. |

## Completion blockers

The repository cannot be treated as complete until these are resolved:

1. CI/CD workflows are added and passing.
2. Fresh-clone setup is tested against Docker Postgres.
3. Feature contracts exist for CxG, CxA and CxT.
4. CxG needs further calibration/monitoring beyond the reproducible baseline path.
5. Registry-backed aggregate serving needs final wiring beyond file exports.
6. CxT leakage warning is fixed and the report is regenerated.
7. README, project status and implementation docs are aligned.
8. Dashboard reads stable generated outputs.
9. Model cards and limitations are present for each metric family.

## Active roadmap

The completion roadmap is tracked in [`docs/ROADMAP.md`](./ROADMAP.md).

Recommended PR sequence:

1. Audit and roadmap.
2. CI/CD and repository hygiene.
3. Data ingestion reproducibility.
4. Feature contracts and quality gates.
5. CxG end-to-end completion (baseline runner implemented; calibration refinements remain).
6. API inference completion.
7. CxA completion.
8. CxT leakage fix and completion.
9. Dashboard and storytelling.
10. v1 release packaging.

## Definition of complete

The project reaches v1 complete when a reviewer can:

1. Clone the repository.
2. Install dependencies.
3. Start Postgres.
4. Run migrations.
5. Fetch a configured StatsBomb subset.
6. Ingest and normalise events.
7. Build feature tables.
8. Train and evaluate CxG.
9. Generate neutralised/opponent-adjusted outputs.
10. Run CxA and CxT pipelines without unresolved leakage warnings.
11. Query real predictions through the API.
12. Open a dashboard backed by generated outputs.
13. See CI passing on pull requests.
14. Read consistent documentation that separates implemented work from future extensions.

## Known limitations

- StatsBomb Open Data coverage is limited and competition-dependent.
- Some pressure and opponent-quality signals are proxies rather than tracking-data measures.
- CxA and CxT require careful validation because sequence/action attribution is sensitive to label design.
- CxT must be revalidated after leakage-prone features are removed.
- API inference depends on stable model artifact and feature-contract handling.

## Next milestone

The next milestone after this status cleanup is to add CI/CD workflows and repository hygiene so all later completion work lands through tested pull requests.
