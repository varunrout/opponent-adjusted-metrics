# V1 Release Checklist

Use this checklist before creating the v1.0.0 tag after the release PR is merged.

## Required

- [ ] Tests pass locally with `poetry run pytest -v -m "not e2e"`.
- [ ] Lint passes with `poetry run ruff check src scripts tests app`.
- [ ] Formatting check passes with `poetry run black --check src scripts tests app`.
- [ ] Targeted mypy command passes.
- [ ] CI passes on the release PR.
- [ ] Dashboard runs with `make dashboard`.
- [ ] README links to release notes, reviewer quickstart, and dashboard walkthrough.
- [ ] Release notes are updated at `docs/releases/v1.0.0.md`.
- [ ] Changelog includes a `v1.0.0` section.
- [ ] Reviewer quickstart exists at `docs/releases/v1_reviewer_quickstart.md`.
- [ ] Generated outputs are not committed.
- [ ] v1 limitations are documented.
- [ ] Deferred work is documented.

## Post-PR43 Reproducibility Checks

- [ ] `make reproduce-v1` completes on a clean local checkout.
- [ ] `make ingestion-report` writes `outputs/reports/ingestion/db_status.json`.
- [ ] `action_features` table is populated.
- [ ] `shot_features` possession/context fields are populated.
- [ ] `model_registry` contains CxG, CxA, and CxT model rows.
- [ ] `shot_predictions` is populated by CxG.
- [ ] `action_predictions` is populated by CxA.
- [ ] `action_threat_predictions` is populated from real generated CxT rows.
- [ ] `aggregates_sequence` is populated.
- [ ] No generated parquet, CSV, joblib, SQLite DB, `feature_store/`, or `outputs/` files are committed.

## Deferred After V1

- [ ] CxT+.
- [ ] Contextual CxT.
- [ ] Advanced CxT.
- [ ] OD-CxT and OD-CxT+.
- [ ] Production deployment.
- [ ] Live data ingestion.
- [ ] Tracking-data workflows.

## After Merge

- [ ] Confirm the default branch is green.
- [ ] Create the `v1.0.0` tag.
- [ ] Publish release notes from `docs/releases/v1.0.0.md`.
