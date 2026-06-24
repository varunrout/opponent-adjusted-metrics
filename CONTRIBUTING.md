# Contributing

This project is being completed through focused pull requests. Each PR should have a single purpose and should leave the repository more reproducible than before.

## Local setup

```bash
poetry install --with dev
cp .env.example .env
make db-up
make migrate-up
```

## Quality checks

Run these before opening a PR:

```bash
poetry run ruff check src scripts tests
poetry run black --check src scripts tests
poetry run mypy src/opponent_adjusted/api/schemas.py src/opponent_adjusted/features/cxg/context.py src/opponent_adjusted/features/cxg/geometry.py src/opponent_adjusted/features/context.py src/opponent_adjusted/features/geometry.py
poetry run pytest -v
```

Full-repo MyPy currently has a known typing backlog. New stable modules should be added to the typed-core MyPy scope once they are clean.

E2E tests currently require local StatsBomb data and database setup, so normal CI excludes them until fixture-backed E2E tests are added. `pip-audit` is currently advisory while vulnerable dependency upgrades are handled in a separate hardening PR.

For database-related changes, also run:

```bash
make db-up
make migrate-up
```

## Pull request expectations

A good PR should include:

- A clear summary of the football/data problem being solved.
- Tests or smoke checks for new behaviour.
- Updated docs when commands, outputs or methodology change.
- Model metrics when training or evaluation logic changes.
- Explicit notes on data assumptions and leakage risks.

## Modelling rules

- Keep train/test splits at match level or higher when match context can leak across rows.
- Do not use target-derived columns as model features.
- Save model metadata with feature lists, target definition, split design and evaluation date.
- Treat provider xG as a benchmark, not as proof that a new model is automatically better.
- Any known leakage warning must be resolved before a model is described as complete.

## Documentation rules

- Avoid claiming production readiness unless the code is runnable, tested and documented.
- Separate implemented work from planned extensions.
- Keep README, project status and model cards consistent.
