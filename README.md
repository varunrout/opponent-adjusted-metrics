# Opponent-Adjusted Football Metrics

Portfolio/demo release for contextual football analytics built on StatsBomb-style event data.

**v1.0.0 status:** CxG, CxA, baseline CxT, aggregate reports, interpretation reports, and Streamlit dashboard v1 are implemented. Generated outputs are reproducible locally and intentionally ignored by Git.

This is not a production deployment. It is a reviewable v1 analytics product surface that shows how raw events become model outputs, reports, and football insight.

## Core Metrics

- **CxG:** contextual shot quality. It answers: how good was the shot?
- **CxA:** baseline chance-creation action value. It answers: which actions moved possessions toward chances?
- **Baseline CxT:** threat added by moving the ball between pitch zones. It answers: who progresses the ball into more dangerous areas?

Deferred until after v1: CxT+, Contextual CxT, Advanced CxT, OD-CxT, OD-CxT+, production deployment, live data ingestion, and tracking-data workflows.

## 5-10 Minute Reviewer Quickstart

```bash
git clone https://github.com/varunrout/opponent-adjusted-metrics.git
cd opponent-adjusted-metrics
poetry install
poetry run pytest -v -m "not e2e"
make dashboard
```

The dashboard starts even when generated outputs are missing. Missing-output messages and empty tables are expected in a clean checkout.

To populate dashboard data locally:

```bash
make cxg-smoke
make cxa-smoke
make cxt-baseline
```

Reviewer docs:

- [v1.0.0 release notes](docs/releases/v1.0.0.md)
- [v1 reviewer quickstart](docs/releases/v1_reviewer_quickstart.md)
- [dashboard demo walkthrough](docs/dashboard/demo_walkthrough.md)
- [v1 release scope](docs/releases/v1_scope.md)
- [generated outputs](docs/OUTPUTS.md)

## Dashboard

Run:

```bash
make dashboard
```

Direct command:

```bash
poetry run streamlit run app/streamlit_app.py
```

Dashboard sections:

- Overview and v1 status
- Player analysis
- Team analysis
- CxG
- CxA
- CxT
- Action explorer
- Reports / diagnostics
- About methodology

Suggested demo flow: start with Overview, compare Player and Team analysis, explain CxG/CxA/CxT pages, use Action explorer to trace aggregate values back to actions, then show Reports / diagnostics.

## Main Workflows

CxG:

```bash
poetry run python scripts/run_cxg_pipeline.py
poetry run python scripts/run_cxg_end_to_end.py
poetry run python scripts/check_cxg_outputs.py
poetry run python scripts/validate_cxg_outputs.py
```

CxA:

```bash
poetry run python scripts/run_cxa_pipeline.py
poetry run python scripts/run_cxa_end_to_end.py
```

CxT:

```bash
poetry run python scripts/run_cxt_pipeline.py
```

Convenience targets:

```bash
make cxg-smoke
make cxa-smoke
make cxt-baseline
```

## Validation

```bash
poetry run ruff check src scripts tests app
poetry run black --check src scripts tests app
poetry run pytest -v -m "not e2e"
poetry run mypy src/opponent_adjusted/api/schemas.py src/opponent_adjusted/features/cxg/context.py src/opponent_adjusted/features/cxg/geometry.py src/opponent_adjusted/features/context.py src/opponent_adjusted/features/geometry.py
```

## Project Structure

```text
opponent-adjusted-metrics/
|-- app/                      # Streamlit dashboard v1
|-- configs/                  # Feature and dashboard contracts
|-- dashboard/                # Earlier dashboard assets/components
|-- docs/                     # Release, dashboard, modelling, and output docs
|-- scripts/                  # Ingestion, feature, modelling, and validation commands
|-- src/opponent_adjusted/    # Package code
|-- tests/                    # Unit, contract, dashboard, and smoke tests
|-- feature_store/            # Generated features, ignored by Git
`-- outputs/                  # Generated model/report outputs, ignored by Git
```

## Generated Outputs

Generated files under `feature_store/` and `outputs/` are not committed. The repository tracks source code, contracts, tests, and documentation needed to regenerate them.

Ignored generated examples:

- `feature_store/cxg/`
- `feature_store/cxa/`
- `feature_store/cxt/`
- `outputs/modeling/cxg/`
- `outputs/modeling/cxa/`
- `outputs/modeling/cxt/`

## API

Start the FastAPI service:

```bash
poetry run uvicorn opponent_adjusted.api.service:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /models/cxg/version`
- `POST /predict/cxg`

The `/predict/cxg` endpoint requires generated local CxG model artifacts.

## Documentation

- [Changelog](CHANGELOG.md)
- [v1 release checklist](docs/releases/v1_release_checklist.md)
- [Feature contracts](docs/feature_contracts.md)
- [CxG model card](docs/modeling/cxg/model_card.md)
- [CxA design contract](docs/modeling/cxa/design_contract.md)
- [CxT design contract](docs/modeling/cxt/design_contract.md)
- [Dashboard design](docs/dashboard/v1_dashboard_design.md)
- [Project story](docs/storytelling/v1_project_story.md)

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
