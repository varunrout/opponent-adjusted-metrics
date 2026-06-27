# Changelog

All notable changes to this project will be documented here.

This project follows a practical completion roadmap before the first stable release.

## Unreleased

### Added

- Pull request quality workflow for linting, formatting, type checking and tests.
- Database smoke workflow for Postgres migration validation.
- Security workflow with dependency audit and CodeQL.
- Pre-commit configuration.
- Pull request template.
- Contribution guide.

## v1.0.0 - 2026-06-27

### Modelling

- Added reproducible CxG feature, training, validation, metadata, and API-backed inference paths.
- Added baseline CxA feature, modelling, attribution, aggregate, and reporting paths.
- Added leakage-safe baseline CxT with deterministic zone/grid threat values.
- Added CxT player, team, sequence, zone-transition, top-action, and interpretation outputs.
- Kept generated outputs under ignored `feature_store/` and `outputs/` paths.

### Dashboard

- Added Streamlit dashboard v1 at `app/streamlit_app.py`.
- Added contract-driven dashboard data loading from `configs/dashboard/v1_dashboard_contract.json`.
- Added graceful missing-output handling for clean checkouts.
- Added guided storytelling, v1 scope banner, metric explanations, and dashboard demo walkthrough.

### Documentation

- Added v1 release notes.
- Added reviewer quickstart.
- Added release checklist.
- Added dashboard design, storytelling, release scope, and demo walkthrough docs.
- Updated README for portfolio-ready v1 review.

### Testing/CI

- Added tests for CxG, CxA, CxT, dashboard contract, dashboard data loading, storytelling, and release packaging.
- Maintained linting, formatting, non-e2e tests, and targeted mypy validation commands.

### Known Limitations

- v1 is a portfolio/demo release, not a production deployment.
- Generated model outputs are not committed and must be regenerated locally.
- Baseline models should not be read as production-grade calibration claims.
- Tracking data is not required or included.

### Deferred Work

- CxT+.
- Contextual CxT.
- Advanced CxT.
- OD-CxT and OD-CxT+.
- Live data ingestion.
- Production deployment, monitoring, and model registry workflows.
