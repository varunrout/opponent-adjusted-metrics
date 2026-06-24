# Project Status

This document is the source of truth for the current implementation status of `opponent-adjusted-metrics`.

## Current status: completion programme in progress

The repository has a strong foundation for an end-to-end football analytics system, but it should not yet be presented as a fully complete production project. The codebase includes database infrastructure, ingestion scripts, feature engineering modules, modelling work, reports, API scaffolding, and dashboard assets. The next phase is to make the system reproducible, tested, model-backed, and consistent from a fresh clone.

For the full completion plan, see:

- `docs/COMPLETION_AUDIT.md`
- `docs/ROADMAP.md`

## Status by area

| Area | Status | Notes |
| --- | --- | --- |
| Project structure | Mostly complete | Package structure, scripts, docs, tests, and outputs are present. Needs final organisation and release packaging. |
| Dependency management | Mostly complete | Poetry is configured. CI needs to validate install, lint, tests, and type checks. |
| Database schema | Mostly complete | SQLAlchemy models and Alembic migrations exist. Needs migration smoke testing in CI. |
| Local Postgres | Partial | Docker Compose support exists. Needs documented end-to-end smoke path. |
| Data ingestion | Partial | Scripts exist. Needs idempotency checks, row-count reporting, and fixture coverage. |
| Event normalisation | Partial | Normalisation scripts exist. Needs coverage report across relevant StatsBomb event types. |
| Feature engineering | Partial | Geometry, context, pressure, opponent, CxA, and CxT feature work exists. Needs feature contracts and quality gates. |
| CxG | Partial | Modelling work and reports exist. Needs one standard pipeline that trains, evaluates, neutralises, aggregates, and registers outputs. |
| CxA | Partial | Analysis and sequence-aware work exists. Needs final methodology naming, reproducible training, and leaderboards. |
| CxT | Needs remediation | Existing reporting flags possible leakage. Completion requires feature review and regenerated evaluation. |
| API | Partial | Health, model version, and aggregate endpoints exist. `/predict/cxg` still needs real model-backed inference. |
| Dashboard | Partial | Dashboard assets exist. Needs clean run instructions and integration with generated outputs. |
| Tests | Partial | Test coverage exists but needs expansion across smoke paths, API, feature contracts, and pipelines. |
| CI/CD | Missing | GitHub Actions workflows need to be added. |
| Documentation | Partial | Several docs exist, but status, roadmap, README, and reports need consistency. |

## Implemented foundations

### Project infrastructure

- Poetry-based dependency management.
- Python package under `src/opponent_adjusted`.
- Environment configuration through Pydantic settings.
- Logging, timing, and batching utilities.
- Makefile with common development and pipeline commands.

### Database layer

- SQLAlchemy models for competitions, teams, players, matches, events, raw events, possessions, shots, shot features, opponent profiles, model registry, predictions, aggregates, and evaluation metrics.
- Alembic migration setup.
- Local Postgres support through Docker Compose.

### Data and feature layer

- StatsBomb data loading and ingestion scripts.
- Event normalisation scripts.
- Shot feature generation.
- Opponent profile generation.
- CxA and CxT feature modules.

### Modelling and analysis layer

- CxG modelling modules and reports.
- CxA analysis and modelling modules.
- CxT modelling and evaluation modules.
- Generated reports and charts in the repository outputs.

### API and presentation layer

- FastAPI service scaffold.
- Health endpoint.
- Model version endpoint.
- Player and team aggregate endpoints.
- Dashboard folder and assets.

## Known gaps before release

### 1. Fresh-clone reproducibility

The project needs one documented route from fresh clone to generated outputs. The target path is:

```bash
cp .env.example .env
make db-up
make migrate-up
make fetch-data
make ingest-all
make normalize-events
make build-features
make build-profiles
make run-cxg-pipeline
```

This path should be tested and documented.

### 2. CI/CD

The repo needs GitHub Actions workflows for:

- unit tests
- linting
- formatting checks
- type checks
- security scanning
- Docker/Postgres smoke tests
- documentation checks

### 3. Feature contracts

CxG, CxA, and CxT need explicit feature contracts that define:

- required columns
- target columns
- nullable columns
- categorical columns
- columns excluded from inference
- leakage-sensitive columns

### 4. CxG completion

CxG should become the first fully completed metric. Completion requires:

- baseline model
- contextual model
- grouped validation
- held-out evaluation
- calibration metrics
- neutralised scores
- player and team aggregates
- model registry entry
- final report
- model-backed API prediction

### 5. CxA completion

CxA should be positioned as sequence-adjusted CxA. Completion requires:

- action sequence builder
- shot creation model
- downstream quality model
- credit distribution
- player/team leaderboards
- example sequences
- final methodology report

### 6. CxT completion

CxT must be remediated before release. Completion requires:

- removal of leakage-prone features from completion modelling
- separate completion and value-gain feature sets
- regenerated evaluation
- slice analysis
- player/team leaderboards
- final report without unresolved leakage warning

### 7. API completion

The `/predict/cxg` endpoint must load a real artefact, apply the feature contract, and return model-backed predictions.

### 8. Dashboard completion

The dashboard should read generated outputs and present:

- CxG leaderboards
- CxA leaderboards
- CxT leaderboards
- player profiles
- team profiles
- shot/action explorer views

## Release readiness checklist

- [ ] CI workflows are green.
- [ ] Docker database smoke path works.
- [ ] Alembic migrations are tested.
- [ ] Ingestion is idempotent.
- [ ] Feature contracts exist.
- [ ] CxG is complete end-to-end.
- [ ] CxA is complete end-to-end.
- [ ] CxT leakage is fixed.
- [ ] API returns real predictions.
- [ ] Dashboard displays generated outputs.
- [ ] README includes final architecture and screenshots.
- [ ] Model cards exist for CxG, CxA, and CxT.
- [ ] `v1.0.0` release notes are prepared.

## Next PRs

1. CI/CD and repository hygiene.
2. Reproducible ingestion.
3. Feature store and quality checks.
4. CxG end-to-end completion.
5. API prediction completion.
6. CxA completion.
7. CxT leakage fix and completion.
8. Dashboard and release packaging.
