.PHONY: help install migrate-create migrate-up migrate-down \
	db-up db-down db-logs db-psql \
	ingest-competitions ingest-matches ingest-events \
	normalize-events build-possessions ingestion-report data-smoke \
	build-features build-profiles \
	build-cxa-action-features cxa-action-features-smoke \
	run-cxa-pipeline run-cxa-end-to-end cxa-run cxa-smoke analysis-cxa \
	run-cxg-pipeline run-cxg-end-to-end run-cxg-diagnostic-training cxg-diagnostic-train validate-cxg-diagnostic cxg-diagnostic-validate generate-cxg-diagnostic-results cxg-diagnostic-results analyze-cxg-feature-impact check-cxg-outputs cxg-validate cxg-run cxg-smoke \
	run-cxg-analysis analysis-cxg analysis-v1 run-cxt-pipeline cxt-baseline cxt-run \
	fetch-data api dashboard streamlit-dashboard clean-rebuild reproduce reproduce-v1 test lint format format-check clean

help:  ## Show this help message
	@echo Usage: make [target]
	@echo.
	@echo Available targets:
	@python -c "import re; rows=[]; [rows.append((m.group(1), m.group(2))) for line in open('Makefile', encoding='utf-8') for m in [re.match(r'^([A-Za-z0-9_-]+):.*?## (.*)$$', line)] if m]; print('\n'.join(f'  {name:<28} {desc}' for name, desc in sorted(rows)))"

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

build-possessions:  ## Build possession rows from normalized events
	poetry run python scripts/build_possessions.py

ingestion-report:  ## Write database ingestion status report
	poetry run python scripts/report_ingestion_status.py

data-smoke: migrate-up fetch-data ingest-all normalize-events build-possessions ingestion-report  ## Migrate, fetch, ingest, normalize, build possessions and report data status

# Features
build-features:  ## Build shot features (VERSION=v1)
	poetry run python scripts/build_shot_features.py --version $(or $(VERSION),v1)

build-profiles:  ## Build opponent profiles (VERSION=v1)
	poetry run python scripts/build_opponent_profiles.py --version $(or $(VERSION),v1)

# CxA
build-cxa-action-features:  ## Build CxA action features from normalized events
	poetry run python scripts/build_cxa_action_features.py

cxa-action-features-smoke:  ## Build CxA action features on a small deterministic subset
	poetry run python scripts/build_cxa_action_features.py --smoke --max-matches 20

run-cxa-pipeline:  ## Run CxA pipeline
	poetry run python scripts/run_cxa_pipeline.py

run-cxa-end-to-end:  ## Train, evaluate, and export CxA baseline outputs
	poetry run python scripts/run_cxa_end_to_end.py

cxa-run: build-cxa-action-features run-cxa-end-to-end  ## Regenerate CxA baseline features and model outputs

cxa-smoke: cxa-action-features-smoke  ## Alias for the local CxA feature smoke

analysis-cxa:  ## Run pre-model CxA target and action-feature analysis
	poetry run python scripts/run_cxa_analysis.py

# CxG
run-cxg-pipeline:  ## Run CxG pipeline
	poetry run python scripts/run_cxg_pipeline.py

run-cxg-end-to-end:  ## Train, evaluate, and export CxG modeling outputs
	poetry run python scripts/run_cxg_end_to_end.py

run-cxg-diagnostic-training:  ## Train diagnostic-informed CxG model candidates
	poetry run python scripts/run_cxg_diagnostic_training.py

cxg-diagnostic-train: run-cxg-diagnostic-training  ## Alias for diagnostic-informed CxG training

validate-cxg-diagnostic:  ## Validate diagnostic-informed CxG against baseline
	poetry run python scripts/validate_cxg_diagnostic_model.py

cxg-diagnostic-validate: validate-cxg-diagnostic  ## Alias for diagnostic-informed CxG validation

generate-cxg-diagnostic-results:  ## Generate promoted diagnostic CxG result outputs
	poetry run python scripts/generate_cxg_diagnostic_results.py

cxg-diagnostic-results: generate-cxg-diagnostic-results  ## Alias for diagnostic CxG results

analyze-cxg-feature-impact:  ## Analyze promoted diagnostic CxG feature impact
	poetry run python scripts/analyze_cxg_feature_impact.py

check-cxg-outputs:  ## Validate generated CxG output contract and Git ignore rules
	poetry run python scripts/check_cxg_outputs.py

cxg-validate:  ## Generate CxG validation summary, calibration, and slice reports
	poetry run python scripts/validate_cxg_outputs.py

cxg-run: run-cxg-pipeline run-cxg-end-to-end check-cxg-outputs cxg-validate  ## Regenerate and validate CxG outputs

cxg-smoke: cxg-run  ## Alias for the full local CxG reproducibility smoke

run-cxg-analysis:  ## Run CxG analysis
	poetry run python scripts/run_cxg_analysis.py

analysis-cxg: run-cxg-analysis  ## Alias for the CxG analysis report

analysis-v1: analysis-cxg analysis-cxa  ## Run available pre-model analysis reports

# CxT
run-cxt-pipeline:  ## Run baseline CxT pipeline
	poetry run python scripts/run_cxt_pipeline.py

cxt-baseline: run-cxt-pipeline  ## Regenerate baseline CxT outputs

cxt-run: cxt-baseline  ## Alias for the local baseline CxT run

train-cxt:
	poetry run python scripts/train_cxt_model.py

evaluate-cxt:
	poetry run python scripts/evaluate_cxt_final.py

# API
api:  ## Start API server
	poetry run uvicorn opponent_adjusted.api.service:app --host 0.0.0.0 --port 8000 --reload

# Dashboard
dashboard:  ## Start Streamlit dashboard v1
	poetry run streamlit run app/streamlit_app.py

streamlit-dashboard: dashboard  ## Alias for Streamlit dashboard v1

clean-rebuild: migrate-up fetch-data ingest-all normalize-events build-possessions ingestion-report build-features build-profiles build-cxa-action-features run-cxg-pipeline run-cxt-pipeline  ## Rebuild generated local data from a clean checkout

reproduce: clean-rebuild  ## Alias for clean local reproducibility path

reproduce-v1: migrate-up fetch-data ingest-all normalize-events build-possessions build-features build-profiles build-cxa-action-features run-cxg-pipeline run-cxg-end-to-end run-cxa-pipeline run-cxa-end-to-end run-cxt-pipeline ingestion-report  ## Reproduce v1 generated outputs from a clean checkout

# Development
test:  ## Run tests
	poetry run pytest -v

test-cov:  ## Run tests with coverage
	poetry run pytest --cov=src/opponent_adjusted --cov-report=html --cov-report=term

lint:  ## Run linting
	poetry run ruff check src/ scripts/ tests/

format:  ## Format code
	poetry run black src/ scripts/ tests/

format-check:  ## Check formatting
	poetry run black --check src scripts tests app

type-check:  ## Run type checking
	poetry run mypy src/

# Cleanup
clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

# Full pipeline
pipeline: migrate-up ingest-all normalize-events build-possessions build-features build-profiles build-cxa-action-features run-cxg-pipeline run-cxt-pipeline ingestion-report  ## Run complete pipeline
