.PHONY: help install migrate-create migrate-up migrate-down \
	db-up db-down db-logs db-psql \
	ingest-competitions ingest-matches ingest-events \
	normalize-events ingestion-report data-smoke \
	build-features build-profiles \
	run-cxa-pipeline run-cxg-pipeline run-cxg-end-to-end check-cxg-outputs cxg-validate cxg-run cxg-smoke \
	run-cxg-analysis run-cxt-pipeline train-cxt evaluate-cxt \
	fetch-data api test lint format clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup
install:  ## Install dependencies
	poetry install

fetch-data:  ## Download configured StatsBomb subset (CONFIG=..., FORCE=1 optional)
	poetry run python scripts/fetch_statsbomb_subset.py --events --config $(or $(CONFIG),configs/statsbomb_subset.json) $(if $(FORCE),--force,)

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

ingest-events:  ## Ingest events (LIMIT=10 optional)
	poetry run python scripts/ingest_events.py $(if $(LIMIT),--limit $(LIMIT),)

ingest-all: ingest-competitions ingest-matches ingest-events  ## Run full ingestion pipeline

# Normalization
normalize-events:  ## Normalize all raw events and populate detail tables
	poetry run python scripts/normalize_events.py --only-missing --batch-size 20000 --fill-missing-detail

ingestion-report:  ## Write database ingestion status report
	poetry run python scripts/report_ingestion_status.py

data-smoke: fetch-data ingest-all normalize-events ingestion-report  ## Fetch, ingest, normalize and report data status

# Features
build-features:  ## Build shot features (VERSION=v1)
	poetry run python scripts/build_shot_features.py --version $(or $(VERSION),v1)

build-profiles:  ## Build opponent profiles (VERSION=v1)
	poetry run python scripts/build_opponent_profiles.py --version $(or $(VERSION),v1)

# CxA
run-cxa-pipeline:  ## Run CxA pipeline
	poetry run python scripts/run_cxa_pipeline.py

# CxG
run-cxg-pipeline:  ## Run CxG pipeline
	poetry run python scripts/run_cxg_pipeline.py

run-cxg-end-to-end:  ## Train, evaluate, and export CxG modeling outputs
	poetry run python scripts/run_cxg_end_to_end.py

check-cxg-outputs:  ## Validate generated CxG output contract and Git ignore rules
	poetry run python scripts/check_cxg_outputs.py

cxg-validate:  ## Generate CxG validation summary, calibration, and slice reports
	poetry run python scripts/validate_cxg_outputs.py

cxg-run: run-cxg-pipeline run-cxg-end-to-end check-cxg-outputs cxg-validate  ## Regenerate and validate CxG outputs

cxg-smoke: cxg-run  ## Alias for the full local CxG reproducibility smoke

run-cxg-analysis:  ## Run CxG analysis
	poetry run python scripts/run_cxg_analysis.py

# CxT
run-cxt-pipeline:  ## Run CxT pipeline
	poetry run python scripts/run_cxt_pipeline.py

train-cxt:  ## Train CxT model
	poetry run python scripts/train_cxt_model.py

evaluate-cxt:  ## Evaluate CxT model
	poetry run python scripts/evaluate_cxt_final.py

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
pipeline: ingest-all normalize-events build-features build-profiles run-cxg-pipeline run-cxt-pipeline ingestion-report  ## Run complete pipeline
