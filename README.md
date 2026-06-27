# Opponent-Adjusted Football Metrics

Opponent-adjusted football analytics workflows built on StatsBomb Open Data. The repository now has a tested ingestion foundation, feature-contract checks, a reproducible CxG baseline runner, model artifact export, model-card documentation, API-backed CxG prediction, and CI quality gates.

This is not yet a complete v1 product. CxG is the most mature metric family; CxA, CxT, dashboard integration, and final release packaging remain partial.

## Current Scope

### Implemented

- StatsBomb Open Data ingestion foundation with SQLAlchemy/Alembic database support.
- Fixture-backed ingestion tests using small StatsBomb-like test data.
- Feature contracts for CxG, CxA, and CxT inputs.
- CxG feature-building and baseline end-to-end runner.
- CxG model artifact and metadata export.
- CxG model-card documentation.
- FastAPI-backed CxG prediction from the exported model artifact.
- CxA baseline feature, modelling, attribution, and aggregate reporting path.
- Baseline CxT threat grid, action threat, aggregate, and interpretation reporting path.
- CI quality gates for linting, formatting, typing, and tests.

### Partial

- Dashboard and storytelling work is in planning stage for v1 release.
- CxT+, Contextual CxT, Advanced CxT, and OD-CxT are deferred until after v1.
- Dashboard files exist, but the dashboard is not yet a stable v1 output surface.
- Final v1 packaging, release notes, and full fresh-clone walkthrough are still pending.

## Quick Start

### Requirements

- Python 3.11+
- Poetry
- PostgreSQL 12+ for database-backed ingestion workflows
- Docker Compose if using the bundled local database service

### Install

```bash
git clone https://github.com/varunrout/opponent-adjusted-metrics.git
cd opponent-adjusted-metrics
poetry install
cp .env.example .env
```

### Optional Local Database

```bash
docker compose up -d db
poetry run alembic upgrade head
```

If application code runs inside another container, set the database host to `db` rather than `localhost`.

## Main Workflows

### CxG Feature Pipeline

Build the CxG feature outputs:

```bash
poetry run python scripts/run_cxg_pipeline.py
```

### CxG End-to-End Baseline

Train, evaluate, and export the current CxG baseline artifacts and metadata:

```bash
poetry run python scripts/run_cxg_end_to_end.py
```

This generates the API-loadable CxG artifact at `outputs/modeling/cxg/models/contextual_model.joblib` with required metadata at `outputs/modeling/cxg/models/contextual_model.json`. The `/predict/cxg` endpoint requires those local generated files to exist.

Validate the regenerated output contract and Git ignore rules:

```bash
poetry run python scripts/check_cxg_outputs.py
```

Generate validation reports for metrics, calibration, slices, grouped checks, and baseline comparison where available:

```bash
poetry run python scripts/validate_cxg_outputs.py
```

Or run the full local CxG reproducibility smoke:

```bash
make cxg-smoke
```

Generated files, including validation reports, are written under `feature_store/` and `outputs/`. They are intentionally not tracked by Git. See [docs/OUTPUTS.md](docs/OUTPUTS.md) for regeneration notes.

### CxA Baseline

Build baseline CxA action features and train/evaluate the first event-data baseline:

```bash
poetry run python scripts/run_cxa_pipeline.py
poetry run python scripts/run_cxa_end_to_end.py
```

Or run both steps:

```bash
make cxa-smoke
```

This emits generated action predictions, player/team/sequence aggregates, and attribution reports under `feature_store/cxa/` and `outputs/modeling/cxa/`. The CxA path is baseline attribution, not a final causal assist model or API surface.

### CxT Baseline

Generate the baseline CxT threat grid, action-level threat values, aggregates, and interpretation reports:

```bash
poetry run python scripts/run_cxt_pipeline.py
```

Or use the Make target:

```bash
make cxt-baseline
```

The baseline uses a deterministic 12x8 zone/grid threat table and calculates `cxt_value = end_threat - start_threat`. It emits player/team/sequence aggregates, zone-transition summaries, top-action reports, and an interpretation summary for final-third entries, box entries, pass threat, carry threat, and progressive actions. It is explainable and leakage-safe, not production-grade. CxT+ / Contextual / Advanced / OD-CxT remain future enhancements.

## V1 Product Roadmap

The current v1 direction is to turn the modelling foundations into a reviewable football analytics product. CxG, CxA, and baseline CxT have reproducible generated outputs, aggregate reports, tests, and documented contracts. The next product layer is dashboard and storytelling work: a project overview, player/team analysis, CxG/CxA/CxT pages, an action-level explorer, diagnostics, and example insights.

The v1 target is a portfolio-ready analytics surface backed by generated outputs under `feature_store/` and `outputs/`, without committing those outputs. Advanced CxT enhancements, including CxT+, Contextual CxT, Advanced CxT, OD-CxT, and OD-CxT+, are intentionally deferred until after v1.

### API Service

Start the FastAPI service:

```bash
poetry run uvicorn opponent_adjusted.api.service:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /models/cxg/version`
- `POST /predict/cxg`

## Validation

The repository quality gates are:

```bash
poetry run ruff check src scripts tests
poetry run black --check src scripts tests
poetry run pytest -v -m "not e2e"
poetry run mypy src/opponent_adjusted/api/schemas.py src/opponent_adjusted/features/cxg/context.py src/opponent_adjusted/features/cxg/geometry.py src/opponent_adjusted/features/context.py src/opponent_adjusted/features/geometry.py
```

## Project Layout

```text
opponent-adjusted-metrics/
|-- alembic/                  # Database migrations
|-- configs/                  # Versioned feature contracts and data-subset configs
|-- dashboard/                # Partial dashboard application
|-- docs/                     # Status, roadmap, model cards, and methodology notes
|-- scripts/                  # Runnable ingestion, feature, modelling, and validation commands
|-- src/opponent_adjusted/    # Package code
|-- tests/                    # Unit, fixture-backed, and e2e-style tests
|-- feature_store/            # Generated feature outputs, ignored by Git
`-- outputs/                  # Generated model/report outputs, ignored by Git
```

## Data

The project uses StatsBomb Open Data. Configured subsets and test fixtures are tracked; raw downloaded data and generated feature/model outputs are not.

Tracked examples:

- `configs/statsbomb_subset.json`
- `configs/feature_contracts/*.json`
- `tests/fixtures/statsbomb/**`

Ignored generated data:

- `feature_store/`
- `outputs/`
- `*.parquet`, `*.csv`, `*.joblib`, `*.pkl`, `*.pickle`

## Methodology Notes

The current CxG baseline focuses on reproducibility and contract-backed model export. It should be read as a baseline CxG path, not a final calibrated production model. Future work includes calibration refinements, monitoring, richer slice validation, and stable registry-backed aggregate serving.

CxA and CxT documentation in this repository is useful methodology and exploratory work, but those metric families should not be described as production complete until their pipelines, validation, model cards, and dashboard/API surfaces are finished.

CxA design and guardrails are documented in [docs/modeling/cxa/design_contract.md](docs/modeling/cxa/design_contract.md), with the machine-readable contract in [configs/feature_contracts/cxa_v1.json](configs/feature_contracts/cxa_v1.json). The current CxA path is a baseline event-data model and attribution layer, not a final causal assist model.

CxT design and leakage guardrails are documented in [docs/modeling/cxt/design_contract.md](docs/modeling/cxt/design_contract.md), with the machine-readable contract in [configs/feature_contracts/cxt_v1.json](configs/feature_contracts/cxt_v1.json). The current CxT path is a deterministic baseline zone/grid model, not CxT+ or an opponent-adjusted threat model.

## Documentation

- [Project status](docs/PROJECT_STATUS.md)
- [Roadmap](docs/ROADMAP.md)
- [Generated outputs](docs/OUTPUTS.md)
- [CxG model card](docs/modeling/cxg/model_card.md)
- [Feature contracts](docs/feature_contracts.md)
- [Data dictionary](docs/data_dictionary.md)

Historical reports that no longer represent current project status are kept under `docs/archive/`.

## Contributing

Contributions should keep documentation claims aligned with tested behavior. For code changes, add or update tests, run the validation commands above, and avoid committing generated outputs unless they are explicitly curated small examples.

## License

This project uses StatsBomb Open Data under the [StatsBomb open data license](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

## Citation

```bibtex
@software{opponent_adjusted_metrics,
  title = {Opponent-Adjusted Football Metrics},
  author = {Varun Rout},
  year = {2026},
  url = {https://github.com/varunrout/opponent-adjusted-metrics}
}
```

## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for open football event data.
- The football analytics community for methodological foundations.
