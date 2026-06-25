# Project Status

This document is the current source of truth for the completion state of `opponent-adjusted-metrics`.

## Current Status

The repository has moved beyond a prototype foundation. It now includes CI workflows, fixture-backed StatsBomb ingestion tests, feature contracts, a reproducible CxG end-to-end baseline runner, CxG artifact and metadata export, CxG model-card documentation, and API-backed CxG prediction.

Generated CxG outputs are intentionally not committed. Run `scripts/run_cxg_pipeline.py` and `scripts/run_cxg_end_to_end.py` to regenerate feature and modelling outputs locally.

It should still not be described as a complete v1 product. CxG is baseline-complete and mostly complete as a metric family, while CxA, CxT, dashboard integration, and final packaging remain partial.

## Implementation Status Matrix

| Area | Status | Notes |
| --- | --- | --- |
| Project structure | Mostly complete | Package, scripts, tests, docs, configs, and migrations are in place. |
| Dependency management | Mostly complete | Poetry is configured and validated by CI. |
| Database schema | Mostly complete | SQLAlchemy/Alembic foundation exists; broader Postgres smoke coverage can still improve confidence. |
| Data ingestion | Mostly complete | StatsBomb ingestion foundation exists with fixture-backed tests; full fresh-clone data-volume validation remains useful future work. |
| Event normalisation | Partial | Normalisation path exists; row-count and integrity reporting can be expanded. |
| Feature engineering | Mostly complete for CxG, partial for CxA/CxT | Feature contracts exist; CxG has the most stable feature path. |
| CxG modelling | Mostly complete | One-command baseline training/evaluation/export exists. Future work is calibration, monitoring, and richer production validation. |
| CxG neutralisation | Mostly complete | Runner exports raw, neutral, and opponent-adjusted shot scores plus aggregates. |
| CxA modelling | Partial | Planning, contracts and exploratory code exist, but final reproducible sequence scoring and public metric validation are not complete. |
| CxT modelling | Partial | Modelling/evaluation code exists, but leakage-sensitive design and final validation remain open. |
| API | Mostly complete for CxG prediction, partial overall | `/predict/cxg` can load the emitted artifact; broader registry-backed aggregate serving and API coverage remain future work. |
| Dashboard | Partial | Dashboard material exists but is not yet wired to stable v1 output contracts. |
| Tests | Mostly complete for current CxG foundation | CI runs quality gates and focused tests; broader integration coverage remains valuable. |
| CI/CD | Mostly complete | Quality workflows exist; deeper database and release workflows can be expanded later. |
| Documentation | Mostly complete for current status | Historical generated reports, obsolete summaries, and CxA/CxT completion claims were removed from `docs/`. |

## Remaining Blockers For v1

1. Fresh-clone setup should be tested against Docker Postgres end to end.
2. CxG should receive calibration, monitoring, and richer slice-validation refinements beyond the baseline runner.
3. Registry-backed aggregate serving should be completed beyond file exports.
4. CxA needs final methodology, reproducible sequence scoring, validation, and model-card documentation.
5. CxT needs leakage-sensitive redesign/validation and regenerated reports.
6. Dashboard views need stable generated-output contracts and screenshots.
7. v1 release packaging, limitations, and release notes need to be finalized.

## Active Roadmap

The completion roadmap is tracked in [docs/ROADMAP.md](./ROADMAP.md).

Recommended remaining sequence:

1. CxG calibration, monitoring, and production-readiness refinements.
2. CxA completion.
3. CxT leakage-sensitive completion.
4. Dashboard and storytelling.
5. v1 release packaging.

## Definition Of v1 Complete

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
10. Run CxA and CxT pipelines without unresolved leakage or validation warnings.
11. Query real CxG predictions through the API.
12. Open a dashboard backed by generated outputs.
13. See CI passing on pull requests.
14. Read consistent documentation that separates implemented work from future extensions.

## Known Limitations

- StatsBomb Open Data coverage is limited and competition-dependent.
- Pressure and opponent-quality signals are event-data proxies rather than tracking-data measures.
- The current CxG path is a reproducible baseline, not the final calibrated production model.
- CxA and CxT are sensitive to sequence windows, attribution labels, and leakage controls.
- Dashboard and aggregate-serving surfaces are not yet stable v1 interfaces.
- Generated outputs are reproducible artifacts and are intentionally not tracked by Git.
- Historical generated reports were removed from `docs/` so the repository has one honest documentation state.

## Next Milestone

The next milestone is to harden CxG beyond the baseline runner while keeping CxA, CxT, dashboard work, and v1 packaging clearly scoped as remaining work.
