# Completion Roadmap

This roadmap turns `opponent-adjusted-metrics` into a complete, reproducible and portfolio-ready football analytics project.

## v1 goal

Build an end-to-end system that ingests StatsBomb Open Data, stores normalised football events, builds contextual and opponent-aware features, trains CxG/CxA/CxT models, evaluates and neutralises outputs, exposes predictions through an API, and presents player/team insights through reports and dashboard views.

## PR delivery plan

### PR 1: Audit and roadmap

**Goal:** make project status honest and remove ambiguity.

Deliverables:

- `docs/COMPLETION_AUDIT.md`
- `docs/ROADMAP.md`
- Updated `docs/PROJECT_STATUS.md`
- Clear implementation status matrix
- Completion criteria for each metric layer

Acceptance criteria:

- No module is described as complete unless it is runnable and tested.
- Placeholder and partial areas are explicitly marked.
- Future PRs have clear ownership and scope.

### PR 2: CI/CD and repository hygiene

**Goal:** enforce quality on every future PR.

Deliverables:

- GitHub Actions CI workflow
- Security/dependency workflow
- Docker/Postgres smoke workflow
- Pre-commit configuration
- Contributing guide
- Pull request template

Acceptance criteria:

- Lint, formatting, type checks and tests run on pull requests.
- Database migration smoke test runs against Postgres.
- Contributors have a standard checklist.

### PR 3: Data ingestion reproducibility

**Goal:** make the raw-data pipeline runnable from a fresh clone.

Deliverables:

- Deterministic StatsBomb subset configuration
- Fresh database ingestion path
- Idempotent ingestion checks
- Normalisation validation report
- Fixture-based ingestion tests

Acceptance criteria:

- `make db-up && make migrate-up && make fetch-data && make ingest-all && make normalize-events` can run without manual file edits.
- Re-running ingestion does not duplicate data.
- Row-count and integrity summaries are produced.

### PR 4: Feature contracts and quality gates

**Goal:** make model inputs stable and leakage-aware.

Deliverables:

- Feature contracts for CxG, CxA and CxT
- Required, optional and forbidden columns
- Data quality checks for missingness, invalid coordinates, duplicates and target leakage
- Feature-store output conventions

Acceptance criteria:

- Each modelling script validates its input contract before training or scoring.
- Leakage-prone columns are blocked from inference features.
- Reports are generated for feature quality.

### PR 5: Complete CxG end to end

**Goal:** make CxG the flagship completed metric.

Deliverables:

- Baseline geometry model
- Contextual/opponent-aware model
- Match-grouped validation
- Held-out competition evaluation
- Calibration metrics
- Neutral CxG and opponent-adjusted deltas
- Player/team aggregates
- Model registry entry
- CxG model card

Acceptance criteria:

- One command trains, evaluates and exports CxG outputs.
- Predictions and aggregates are reproducible.
- Model card includes limitations and validation design.

### PR 6: API inference completion

**Goal:** replace placeholder API inference with real model-backed predictions.

Deliverables:

- CxG model artifact loader
- Feature-contract validation at request time
- Real `/predict/cxg` response
- Batch scoring endpoint or runner
- API tests with fixture model
- OpenAPI examples

Acceptance criteria:

- `/predict/cxg` no longer returns `501` when a model artifact is configured.
- Invalid requests fail with clear validation errors.
- Health and aggregate endpoints are covered by tests.

### PR 7: Complete CxA

**Goal:** produce meaningful creator and sequence attribution outputs.

Deliverables:

- Action sequence builder for passes, carries and dribbles
- Shot-creation and shot-quality modelling
- Credit distribution across final action, pre-assist and earlier actions
- Player/team CxA aggregates
- CxA model card and football interpretation report

Acceptance criteria:

- CxA outputs identify creators beyond traditional assists.
- Sequence-level examples are documented.
- Methodology avoids overclaiming terms such as `true xA` unless precisely defined.

### PR 8: Fix and complete CxT

**Goal:** remove leakage risk and make CxT defensible.

Deliverables:

- Remove leakage-prone completion features
- Rebuild completion and xT-gain models
- Slice evaluation by pressure, zone and action type
- Player/team CxT aggregates
- Updated CxT report and model card

Acceptance criteria:

- CxT report no longer contains unresolved leakage warnings.
- Completion and value models are evaluated separately.
- CxT outputs are interpretable by action type.

### PR 9: Dashboard and storytelling

**Goal:** make the project understandable to non-technical football audiences.

Deliverables:

- Dashboard wired to stable output files
- CxG, CxA and CxT leaderboards
- Player/team profile pages
- Shot/action explorer
- README screenshots
- Football interpretation guide

Acceptance criteria:

- Dashboard runs from generated outputs.
- README explains what a football analyst can learn from the project.
- At least one player, team and match case study is included.

### PR 10: v1 release packaging

**Goal:** mark the project as complete and reproducible.

Deliverables:

- Final README rewrite
- Architecture diagram
- Data-flow diagram
- Model cards
- Limitations page
- Docker Compose full-stack instructions
- Tagged release notes

Acceptance criteria:

- Fresh-clone path is documented and tested.
- CI is passing.
- v1 release notes clearly distinguish implemented work from future extensions.

## Completion checklist

- [ ] CI passes for lint, formatting, typing and tests.
- [ ] Docker Postgres smoke test passes.
- [ ] Fresh ingestion path is documented.
- [ ] Feature contracts exist for all metric families.
- [ ] CxG end-to-end pipeline is complete.
- [ ] API returns real predictions.
- [ ] CxA outputs are complete and documented.
- [ ] CxT leakage warning is resolved.
- [ ] Dashboard runs on generated outputs.
- [ ] README and project status are consistent.
- [ ] v1 release notes are published.
