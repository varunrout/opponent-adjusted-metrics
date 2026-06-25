# Completion Roadmap

This roadmap tracks the path from the current CxG baseline-complete repository to a complete, reproducible v1 football analytics project.

## v1 Goal

Build an end-to-end system that ingests StatsBomb Open Data, stores normalised football events, builds contextual and opponent-aware features, trains and validates CxG/CxA/CxT metrics, exposes CxG predictions through an API, and presents player/team insights through reports and dashboard views.

## Completed Phases

### Phase 1: Audit And Status Alignment

Completed:

- Project status audit.
- Roadmap creation.
- Honest implementation matrix.
- Completion criteria for each metric layer.

### Phase 2: CI/CD And Repository Hygiene

Completed or mostly complete:

- CI quality gates for linting, formatting, typing, and tests.
- Repository contribution guidance.
- Git tracking rules for generated outputs.

Remaining refinements:

- Broader database smoke workflows.
- Release-oriented automation.

### Phase 3: Data Ingestion Reproducibility

Completed or mostly complete:

- StatsBomb ingestion foundation.
- Deterministic subset configuration.
- Fixture-backed ingestion tests.

Remaining refinements:

- Full fresh-clone Docker/Postgres walkthrough validation.
- Expanded row-count and integrity reports.

### Phase 4: Feature Contracts And Quality Gates

Completed or mostly complete:

- Feature contracts for CxG, CxA, and CxT.
- Contract validation utility.
- Leakage-aware expectations for model inputs.

Remaining refinements:

- Broader enforcement across every non-CxG modelling path.
- Generated feature-quality reports.

### Phase 5: CxG Baseline End To End

Completed:

- CxG feature pipeline command.
- CxG end-to-end baseline runner.
- Baseline model training and evaluation.
- Neutral/opponent-adjusted scoring outputs.
- Player/team aggregate exports.
- Model artifact and metadata export.
- CxG model card.
- API-backed CxG prediction from the emitted artifact.

Future CxG refinements:

- Calibration improvements beyond the current baseline.
- Monitoring and drift checks.
- Richer slice validation and production-readiness reporting.
- Registry-backed aggregate serving.

## Remaining Phases

### Phase 6: CxA Completion

Goal: produce defensible creator and sequence-attribution outputs.

Deliverables:

- Reproducible action sequence builder for passes, carries, and dribbles.
- Shot-creation and shot-quality modelling.
- Credit distribution across final action, pre-assist, and earlier actions.
- Player/team CxA aggregates.
- CxA model card and football interpretation report.

Acceptance criteria:

- CxA outputs identify creators beyond traditional assists.
- Sequence-level examples are documented.
- Methodology avoids overclaiming terms such as `true xA` unless precisely defined.

### Phase 7: CxT Completion

Goal: remove leakage risk and make CxT defensible.

Deliverables:

- Remove leakage-prone completion features.
- Rebuild completion and xT-gain models.
- Slice evaluation by pressure, zone, and action type.
- Player/team CxT aggregates.
- Updated CxT report and model card.

Acceptance criteria:

- CxT report no longer contains unresolved leakage warnings.
- Completion and value models are evaluated separately.
- CxT outputs are interpretable by action type.

### Phase 8: Dashboard And Storytelling

Goal: make generated outputs useful to football audiences.

Deliverables:

- Dashboard wired to stable generated-output files.
- CxG, CxA, and CxT leaderboards once each metric is ready.
- Player/team profile pages.
- Shot/action explorer.
- README screenshots.
- Football interpretation guide.

Acceptance criteria:

- Dashboard runs from regenerated outputs.
- README explains what a football analyst can learn from the project.
- At least one player, team, and match case study is included.

### Phase 9: v1 Release Packaging

Goal: mark the project as complete and reproducible without overclaiming.

Deliverables:

- Final README review.
- Architecture diagram.
- Data-flow diagram.
- Model cards for completed metric families.
- Limitations page.
- Docker Compose full-stack instructions.
- Tagged release notes.

Acceptance criteria:

- Fresh-clone path is documented and tested.
- CI is passing.
- v1 release notes distinguish implemented work from future extensions.

## Completion Checklist

- [x] CI quality gates exist for lint, formatting, typing, and tests.
- [ ] Docker Postgres smoke test is fully validated.
- [x] Fresh ingestion path has fixture-backed tests.
- [x] Feature contracts exist for CxG, CxA, and CxT.
- [x] CxG baseline end-to-end pipeline is complete.
- [x] API returns real CxG predictions from exported artifacts.
- [ ] CxG calibration/monitoring refinements are complete.
- [ ] CxA outputs are complete and documented.
- [ ] CxT leakage warning is resolved.
- [ ] Dashboard runs on stable generated outputs.
- [ ] README and project status remain consistent.
- [ ] v1 release notes are published.
