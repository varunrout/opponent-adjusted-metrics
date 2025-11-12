# Opponent-Adjusted Football Metrics

An end-to-end system for building contextual, opponent-adjusted football metrics (CxG, CxA, CxT, C-OBV) using **StatsBomb Open Data** and **PostgreSQL**.

## Overview

This project implements advanced football analytics metrics that account for:
- **Shot context**: Game state, possession patterns, pressure
- **Opponent quality**: Defensive strength, zone-specific ratings
- **Geometric factors**: Distance, angle, centrality
- **Neutralization**: Isolating player/team performance from contextual effects

### Metrics Implemented

1. **CxG (Contextual Expected Goals)**: Shot probability accounting for context and opponent strength
2. **CxA (Contextual Expected Assists)**: Pass value considering completion probability and downstream shot generation (planned)
3. **CxT (Contextual Expected Threat)**: Action value based on state transitions (planned)
4. **C-OBV (Contextual On-Ball Value)**: Comprehensive action value using MDP framework (planned)

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 12+
- Poetry (for dependency management)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/varunrout/opponent-adjusted-metrics.git
cd opponent-adjusted-metrics
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

4. Initialize the database:
```bash
# Create the database in PostgreSQL
createdb opponent_metrics

# Run migrations
poetry run alembic upgrade head
```

### Data Pipeline

The complete pipeline consists of several stages:

#### 1. Data Ingestion

Ingest StatsBomb Open Data for the specified competitions:

```bash
# Ingest competitions
poetry run python scripts/ingest_competitions.py

# Ingest matches
poetry run python scripts/ingest_matches.py

# Ingest events and normalize
poetry run python scripts/ingest_events.py
```

#### 2. Feature Engineering

Build features for shots:

```bash
# Build shot features (geometry, context, pressure)
poetry run python scripts/build_shot_features.py --version v1

# Build opponent defensive profiles
poetry run python scripts/build_opponent_profiles.py --version v1
```

#### 3. Model Training

Train the CxG model:

```bash
# Train CxG model
poetry run python scripts/train_cxg.py --features v1 --version cxg_v1

# Generate neutralized predictions
poetry run python scripts/neutralize_cxg.py --model cxg_v1 --features v1
```

#### 4. Evaluation

Evaluate and generate reports:

```bash
# Evaluate model performance
poetry run python scripts/evaluate_cxg.py --model cxg_v1 --features v1

# Export reports
poetry run python scripts/export_reports.py --model cxg_v1 --features v1
```

### API Service

Start the inference API:

```bash
poetry run uvicorn opponent_adjusted.api.service:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health` - Health check
- `GET /models/cxg/version` - Get current model version
- `POST /predict/cxg` - Predict CxG for a shot
- `GET /aggregates/player?model=cxg_v1&limit=50` - Get player aggregates
- `GET /aggregates/team?model=cxg_v1&limit=50` - Get team aggregates

## Project Structure

```
opponent-adjusted-metrics/
├── src/opponent_adjusted/       # Main package
│   ├── config.py                # Configuration
│   ├── db/                      # Database models and session
│   ├── ingestion/               # Data ingestion modules
│   ├── features/                # Feature engineering
│   ├── modeling/                # Model training and prediction
│   ├── evaluation/              # Evaluation and metrics
│   ├── pipelines/               # End-to-end pipelines
│   ├── api/                     # FastAPI service
│   └── utils/                   # Utility functions
├── scripts/                     # Executable scripts
├── alembic/                     # Database migrations
├── tests/                       # Unit tests
├── reports/                     # Generated reports
└── db/                          # Database seeds
```

## Data Sources

This project uses **StatsBomb Open Data**, specifically:
- FIFA World Cup 2018
- UEFA Euro 2020
- FIFA World Cup 2022
- UEFA Euro 2024

Data is automatically discovered and ingested from the StatsBomb open data repository.

## Database Schema

Key tables:
- `competitions`, `teams`, `players` - Reference data
- `matches`, `events`, `raw_events` - Match data
- `shots`, `shot_features` - Shot-level data and features
- `opponent_def_profile` - Opponent defensive ratings
- `model_registry` - Model versions and metadata
- `shot_predictions` - Raw and neutralized predictions
- `aggregates_player`, `aggregates_team` - Aggregated metrics
- `evaluation_metrics` - Model performance metrics

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Format code
poetry run black src/ scripts/ tests/

# Lint
poetry run ruff check src/ scripts/ tests/

# Type checking
poetry run mypy src/
```

## Methodology

### CxG Model

The CxG (Contextual Expected Goals) model uses:

**Input Features:**
- **Geometry**: Distance, angle, centrality
- **Context**: Score differential, minute bucket, possession stats
- **Pressure**: Under pressure flag, recent defensive actions, composite score
- **Opponent**: Global defensive rating, zone-specific rating, block rate
- **Baseline**: StatsBomb xG

**Model**: LightGBM classifier with post-hoc isotonic calibration

**Neutralization**: 
- Replace contextual and opponent features with reference values
- Reference: Tied game (0-0), minute 55, no pressure, average opponent
- Captures inherent shot quality independent of context

### Acceptance Criteria

- Brier score improvement over baseline xG
- Calibration (ECE) ≤ 0.06 overall, ≤ 0.05 under pressure
- Mean opponent-adjusted diff within ±0.005
- Slice-based performance on pressure, opponent strength, game state

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project uses StatsBomb Open Data under their [open data license](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

## Citation

If you use this project in your research, please cite:

```
@software{opponent_adjusted_metrics,
  title = {Opponent-Adjusted Football Metrics},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/varunrout/opponent-adjusted-metrics}
}
```

## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for providing open football event data
- The football analytics community for methodological foundations

