.PHONY: help install migrate-create migrate-up migrate-down \
	db-up db-down db-logs db-psql \
	ingest-competitions ingest-matches ingest-events \
	normalize-events \
	build-features build-profiles build-cxa-baselines train-cxg neutralize evaluate reports \
	fetch-data api test lint format clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup
install:  ## Install dependencies
	poetry install

fetch-data:  ## Download StatsBomb subset (competitions, matches, events)
	poetry run python scripts/fetch_statsbomb_subset.py --events

# Database
migrate-create:  ## Create a new migration (use MSG="description")
	poetry run alembic revision --autogenerate -m "$(MSG)"

migrate-up:  ## Run database migrations
	poetry run alembic upgrade head

migrate-down:  ## Rollback last migration
	poetry run alembic downgrade -1

# Local Postgres (Docker)
db-up:  ## Start Postgres in Docker
	docker compose up -d db

db-down:  ## Stop Postgres and remove container
	docker compose down

db-logs:  ## Tail Postgres logs
	docker compose logs -f db

db-psql:  ## Open psql shell in the Postgres container
	docker exec -it opponent_metrics_db psql -U $${POSTGRES_USER:-app} -d $${POSTGRES_DB:-opponent_metrics}

# Data Pipeline
ingest-competitions:  ## Ingest competitions
	poetry run python scripts/ingest_competitions.py

ingest-matches:  ## Ingest matches
	poetry run python scripts/ingest_matches.py

ingest-events:  ## Ingest events
	poetry run python scripts/ingest_events.py

ingest-all: ingest-competitions ingest-matches ingest-events  ## Run full ingestion pipeline

# Normalization
normalize-events:  ## Normalize all raw events and populate detail tables
	poetry run python scripts/normalize_events.py --only-missing --batch-size 20000 --fill-missing-detail

# Features
build-features:  ## Build shot features (VERSION=v1)
	poetry run python scripts/build_shot_features.py --version $(or $(VERSION),v1)

build-profiles:  ## Build opponent profiles (VERSION=v1)
	poetry run python scripts/build_opponent_profiles.py --version $(or $(VERSION),v1)

# CxA
build-cxa-baselines:  ## Build cxA baselines (DATABASE_URL=..., MODEL_NAME=cxg, VERSION=, K_ACTIONS=3, DECAY=0.6, LIMIT_SHOTS=)
	poetry run python scripts/build_cxa_baselines.py \
		--database-url $(or $(DATABASE_URL),$(DATABASE_URL)) \
		--model-name $(or $(MODEL_NAME),cxg) \
		$(if $(VERSION),--version $(VERSION),) \
		--k-actions $(or $(K_ACTIONS),3) \
		--decay $(or $(DECAY),0.6) \
		$(if $(LIMIT_SHOTS),--limit-shots $(LIMIT_SHOTS),)

# Training
train-cxg:  ## Train CxG model (FEATURES=v1, VERSION=cxg_v1)
	poetry run python scripts/train_cxg.py --features $(or $(FEATURES),v1) --version $(or $(VERSION),cxg_v1)

neutralize:  ## Generate neutralized predictions (MODEL=cxg_v1, FEATURES=v1)
	poetry run python scripts/neutralize_cxg.py --model $(or $(MODEL),cxg_v1) --features $(or $(FEATURES),v1)

# Evaluation
evaluate:  ## Evaluate model (MODEL=cxg_v1, FEATURES=v1)
	poetry run python scripts/evaluate_cxg.py --model $(or $(MODEL),cxg_v1) --features $(or $(FEATURES),v1)

reports:  ## Export reports (MODEL=cxg_v1, FEATURES=v1)
	poetry run python scripts/export_reports.py --model $(or $(MODEL),cxg_v1) --features $(or $(FEATURES),v1)

# API
api:  ## Start API server
	poetry run uvicorn opponent_adjusted.api.service:app --host 0.0.0.0 --port 8000 --reload

# Development
test:  ## Run tests
	poetry run pytest -v

test-cov:  ## Run tests with coverage
	poetry run pytest --cov=src/opponent_adjusted --cov-report=html --cov-report=term

lint:  ## Run linting
	poetry run ruff check src/ scripts/ tests/

format:  ## Format code
	poetry run black src/ scripts/ tests/

type-check:  ## Run type checking
	poetry run mypy src/

# Cleanup
clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

# Full pipeline
pipeline: ingest-all build-features build-profiles train-cxg neutralize evaluate reports  ## Run complete pipeline
