# Contributing

This project is completed through focused pull requests. Each PR should have one clear purpose and should be small enough to review without reconstructing the entire project history.

## Local setup

```bash
poetry install
cp .env.example .env
```

For database-backed work:

```bash
make db-up
make migrate-up
```

## Development checks

Before opening a PR, run:

```bash
make format
make lint
make type-check
make test
```

To run coverage:

```bash
make test-cov
```

## Pre-commit hooks

Install hooks with:

```bash
poetry run pre-commit install
```

Run all hooks manually with:

```bash
poetry run pre-commit run --all-files
```

## PR expectations

Each PR should include:

- a clear summary
- motivation for the change
- files or modules changed
- validation performed
- known limitations or follow-up work

## Branch naming

Use a prefix that describes the type of work:

```text
completion/<topic>
engineering/<topic>
data/<topic>
features/<topic>
modeling/<topic>
api/<topic>
dashboard/<topic>
docs/<topic>
```

## Testing expectations

Use the smallest reliable test for the change:

- pure utility change: unit tests
- ingestion change: fixture-backed ingestion test
- feature change: feature contract or schema test
- model change: lightweight fixture or smoke test
- API change: FastAPI test client
- migration change: Docker/Postgres smoke test

## Documentation expectations

Update docs when behaviour changes. Do not describe future work as implemented. If a feature is partial, label it as partial and link to the relevant roadmap item.
