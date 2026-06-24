# Completion Roadmap

This roadmap turns `opponent-adjusted-metrics` from a strong prototype into a complete, reproducible football analytics project.

## North star

Build an end-to-end system that ingests StatsBomb Open Data, stores reusable football event data, builds contextual and opponent-adjusted features, trains CxG, CxA, and CxT models, evaluates them rigorously, serves model outputs through an API, and presents player/team insights through reports and a dashboard.

## Delivery principles

- Every major change lands through a focused PR.
- Each PR must be independently reviewable.
- Documentation must reflect what the code actually does.
- No model is described as production-ready until it has reproducible training, evaluation, artefacts, and tests.
- Known limitations are documented rather than hidden.
- CxT cannot be considered complete until leakage-prone features are removed and the report is regenerated.

## Phase 1: Project truth and engineering foundation

### PR 1: Completion audit and roadmap

Goal: make the current state honest and reviewable.

Deliverables:

- `docs/COMPLETION_AUDIT.md`
- `docs/ROADMAP.md`
- Updated `docs/PROJECT_STATUS.md`
- README status section that separates implemented, partial, and planned work

Acceptance criteria:

- The repo has one clear completion definition.
- Stale completion claims are removed or softened.
- The remaining PR sequence is explicit.

### PR 2: CI/CD and repository hygiene

Goal: make every future change automatically checked.

Deliverables:

- GitHub Actions CI for tests, linting, formatting, and type checks
- Security workflow with CodeQL
- Docker/Postgres smoke workflow
- Markdown/docs validation workflow
- `.pre-commit-config.yaml`
- `CONTRIBUTING.md`

Acceptance criteria:

- Pull requests run automated checks.
- The local developer workflow matches the CI workflow.
- The repo has a repeatable quality gate.

## Phase 2: Reproducible data layer

### PR 3: Data ingestion completion

Goal: make the StatsBomb data path reproducible from a fresh database.

Deliverables:

- Config-driven competition subset
- Deterministic fetch command
- Idempotent competition, match, event, and normalisation scripts
- Ingestion row-count report
- Fixture-based ingestion smoke tests

Acceptance criteria:

- `make db-up migrate-up fetch-data ingest-all normalize-events` works from a fresh clone.
- Rerunning ingestion does not duplicate rows.
- A row-count report is generated.

### PR 4: Feature store and data quality checks

Goal: make modelling inputs stable and auditable.

Deliverables:

- Feature-store directories for CxG, CxA, and CxT
- Feature contracts for each metric
- Data quality checks for missingness, invalid coordinates, duplicates, target leakage, and split integrity
- Quality reports saved under `outputs/reports/data_quality/`

Acceptance criteria:

- Every model consumes a documented feature contract.
- Data quality failures are visible before training.
- Leakage checks are part of the workflow.

## Phase 3: Core metric completion

### PR 5: CxG end-to-end completion

Goal: make CxG the flagship completed metric.

Deliverables:

- Baseline geometry model
- Contextual model
- Calibration metrics
- Match-level grouped validation
- Held-out competition evaluation
- Neutralised CxG generation
- Player/team aggregates
- Model metadata and registry persistence
- Final CxG report and charts

Acceptance criteria:

- `make run-cxg-pipeline` produces model artefacts, predictions, neutralised scores, reports, and aggregates.
- CxG report includes limitations and comparison against provider xG where available.

### PR 6: Sequence-adjusted CxA completion

Goal: complete CxA as a sequence-aware creation metric.

Deliverables:

- Action sequence builder for passes, carries, and dribbles
- Shot creation model
- Downstream shot quality model
- Credit distribution across key action, pre-assist, and earlier contributors
- Player and team leaderboards
- CxA methodology report

Acceptance criteria:

- CxA outputs explain creators beyond final assists.
- The public naming avoids overclaiming and uses `sequence-adjusted CxA`.
- Player leaderboards and example sequences are generated.

### PR 7: CxT leakage fix and completion

Goal: make CxT technically defensible.

Deliverables:

- Remove leakage-prone features from completion modelling
- Separate completion and value-gain feature sets
- Pass, carry, dribble, and combined CxT outputs
- Slice evaluation by pressure, zone, final-third entry, and action type
- Regenerated CxT report without unresolved leakage warning

Acceptance criteria:

- CxT report contains no unresolved leakage warning.
- The model has grouped validation and documented limitations.
- Player and team CxT leaderboards are generated.

## Phase 4: Product layer

### PR 8: API prediction completion

Goal: make the API return real model-backed predictions.

Deliverables:

- Model artefact loader
- Feature contract validation
- Real `/predict/cxg` endpoint
- Batch prediction endpoint
- Aggregate endpoints for CxG, CxA, and CxT outputs
- API tests with fixture model artefacts

Acceptance criteria:

- `/predict/cxg` no longer returns 501 when a model artefact exists.
- API responses include model version and prediction metadata.
- API tests pass in CI.

### PR 9: Dashboard and football storytelling

Goal: make the outputs understandable to non-technical reviewers.

Deliverables:

- Dashboard run path
- Model overview page
- CxG, CxA, and CxT leaderboards
- Player profile page
- Team profile page
- Shot/action explorer
- README screenshots
- Example football case studies

Acceptance criteria:

- A reviewer can see model outputs visually without reading code.
- Dashboard data sources are generated by the pipeline.

## Phase 5: Release packaging

### PR 10: Final `v1.0.0` release preparation

Goal: make the repo complete and presentable.

Deliverables:

- Final README rewrite
- Architecture diagram
- Data flow diagram
- Model cards for CxG, CxA, and CxT
- Limitations and ethics section
- Docker Compose full stack
- Release notes
- `CHANGELOG.md`

Acceptance criteria:

- Fresh clone instructions are tested.
- CI is green.
- Project is tagged as `v1.0.0`.
- README accurately represents the final implementation.

## Recommended merge order

1. Audit and roadmap
2. CI/CD and repo hygiene
3. Data ingestion completion
4. Feature store and quality checks
5. CxG completion
6. API CxG prediction
7. CxA completion
8. CxT leakage fix
9. Dashboard and storytelling
10. Final release packaging

This order makes the repo safer to review because it first fixes project truth, then infrastructure, then modelling, then product presentation.
