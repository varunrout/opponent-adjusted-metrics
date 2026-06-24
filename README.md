# Opponent-Adjusted Football Metrics

An end-to-end football analytics project for building contextual and opponent-adjusted metrics from StatsBomb Open Data.

The project is being completed as a full reproducible system: data ingestion, PostgreSQL storage, feature engineering, CxG/CxA/CxT modelling, neutralisation, evaluation, API serving, and dashboard/reporting.

## Current status

This repository has a strong technical foundation, but it is currently under a completion programme rather than a final `v1.0.0` release.

For the live completion plan, see:

- `docs/COMPLETION_AUDIT.md`
- `docs/ROADMAP.md`
- `docs/PROJECT_STATUS.md`

## What the project is designed to do

The final system should:

1. Fetch and ingest a configured StatsBomb Open Data subset.
2. Store normalised match, event, shot, possession, player, and team data in PostgreSQL.
3. Build feature tables for CxG, CxA, and CxT.
4. Train contextual and opponent-adjusted models.
5. Generate neutralised player and team outputs.
6. Evaluate calibration, discrimination, slice performance, and leakage risk.
7. Serve predictions and aggregates through FastAPI.
8. Present football-facing outputs through reports and dashboard views.

## Metrics

### CxG: Contextual Expected Goals

Shot probability accounting for:

- shot geometry
- pressure
- game state
- possession context
- opponent defensive profile
- neutralised opponent context

### CxA: Sequence-adjusted Expected Assists

Creation value assigned across attacking sequences, not only the final pass. The target completion work is to account for:

- passes
- carries
- dribbles
- final actions
- pre-assists
- earlier progression actions
- downstream shot creation and shot quality

### CxT: Contextual Expected Threat

Action value for ball progression actions. The current completion work includes removing leakage-prone features from the completion model and regenerating CxT evaluation before release.

### C-OBV: Contextual On-Ball Value

Planned extension after CxG, CxA, and CxT are complete.

## Repository structure

```text
opponent-adjusted-metrics/
├── alembic/                  # Database migrations
├── dashboard/                # Streamlit dashboard assets
├── docs/                     # Methodology, reports, roadmap, status
├── scripts/                  # CLI entrypoints for ingestion, features, modelling
├── src/opponent_adjusted/    # Main Python package
│   ├── api/                  # FastAPI service and schemas
│   ├── analysis/             # Analysis modules
│   ├── db/                   # SQLAlchemy models and sessions
│   ├── features/             # Feature engineering
│   ├── ingestion/            # StatsBomb loading and ingestion
│   ├── modeling/             # CxG, CxA, CxT modelling
│   ├── pipelines/            # Pipeline orchestration modules
│   ├── prediction/           # Batch prediction and scoring utilities
│   └── utils/                # Shared utilities
├── tests/                    # Unit and integration tests
├── Makefile                  # Common project commands
├── docker-compose.yml        # Local Postgres service
└── pyproject.toml            # Poetry dependency configuration
```

## Quick start

### Requirements

- Python 3.11+
- Poetry
- Docker and Docker Compose
- PostgreSQL 12+ if not using Docker

### Install

```bash
poetry install
cp .env.example .env
```

### Start local database

```bash
make db-up
make migrate-up
```

### Fetch and ingest data

```bash
make fetch-data
make ingest-all
make normalize-events
```

### Build features

```bash
make build-features VERSION=v1
make build-profiles VERSION=v1
```

### Run modelling pipelines

```bash
make run-cxg-pipeline
make run-cxa-pipeline
make run-cxt-pipeline
```

### Start API

```bash
make api
```

Then open:

```text
http://localhost:8000/docs
```

## Development

```bash
make test
make lint
make type-check
```

Format code:

```bash
make format
```

## API endpoints

Current API scaffold includes:

- `GET /health`
- `GET /models/cxg/version`
- `POST /predict/cxg`
- `GET /aggregates/player`
- `GET /aggregates/team`

The prediction endpoint is part of the completion roadmap and must be backed by a real model artefact before release.

## Data source

This project uses StatsBomb Open Data. Usage of StatsBomb data should follow the StatsBomb open data licence and attribution requirements.

## Completion roadmap

The repository is being completed through focused PRs:

1. Completion audit and roadmap.
2. CI/CD and repository hygiene.
3. Reproducible data ingestion.
4. Feature store and data quality checks.
5. CxG end-to-end completion.
6. Sequence-adjusted CxA completion.
7. CxT leakage fix and completion.
8. API prediction completion.
9. Dashboard and football storytelling.
10. Final `v1.0.0` release packaging.

## Citation

```bibtex
@software{opponent_adjusted_metrics,
  title = {Opponent-Adjusted Football Metrics},
  author = {Varun Rout},
  year = {2026},
  url = {https://github.com/varunrout/opponent-adjusted-metrics}
}
```

## Acknowledgements

- StatsBomb for providing open football event data.
- The football analytics community for methodological foundations and public discussion around xG, xA, xT, and possession value modelling.
