# Opponent-Adjusted Football Metrics

Portfolio/demo release for contextual football analytics built on StatsBomb-style event data.

**v1.0.0 status:** CxG, CxA, baseline CxT, aggregate reports, interpretation reports, DB persistence, and Streamlit dashboard v1 are implemented. Generated outputs and the local SQLite database are reproducible locally and intentionally ignored by Git.

This is not a production deployment. It is a reviewable v1 analytics product surface that shows how raw events become feature tables, model outputs, reports, database rows, and football insight.

## Promoted Diagnostic CxG Portfolio

The promoted diagnostic CxG model is the governed shot-quality layer for the portfolio. It scores shot-level CxG from pre-shot, leakage-safe features, then rolls those predictions into player, team, feature-driver, and category insight outputs.

The fair CxG baseline excludes StatsBomb xG as a training feature. Against that fair baseline, the promoted diagnostic model improves log loss, Brier score, and ROC AUC; expected calibration error remains transparently monitored because the baseline is slightly better calibrated in the latest run.

Portfolio entry points:

- [CxG portfolio summary](outputs/portfolio/cxg/cxg_portfolio_summary.md)
- [CxG model scorecard](outputs/portfolio/cxg/cxg_model_scorecard.json)
- [CxG portfolio charts](outputs/portfolio/cxg/charts/)

Regenerate the promoted CxG portfolio locally:

```bash
make build-features
make run-cxg-end-to-end
make run-cxg-diagnostic-training
make validate-cxg-diagnostic
make generate-cxg-diagnostic-results
make analyze-cxg-feature-impact
make build-cxg-portfolio-summary
make dashboard
```

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

To fully regenerate the v1 local database, feature store, model outputs, and reports:

```bash
make reproduce-v1
make ingestion-report
make dashboard
```

To regenerate model families individually after data is already ingested and normalized:

```bash
make cxg-run
make cxa-run
make cxt-baseline
make ingestion-report
```

`make cxa-smoke` only builds a small CxA action-feature smoke dataset. It does not populate full CxA model outputs.

Reviewer docs:

- [v1.0.0 release notes](docs/releases/v1.0.0.md)
- [v1 reviewer quickstart](docs/releases/v1_reviewer_quickstart.md)
- [v1 results summary](docs/modeling/v1_results_summary.md)
- [clean-run reproducibility](docs/reproducibility/clean_run_reproducibility.md)
- [DB schema and lineage](docs/data/db_schema_and_lineage.md)
- [generated outputs](docs/OUTPUTS.md)
- [dashboard demo walkthrough](docs/dashboard/demo_walkthrough.md)

## Dashboard

The active dashboard entry point is `app/streamlit_app.py`. The older
`dashboard/` directory is retained as a legacy/experimental path and is not the
primary CxG portfolio app.

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
- Promoted CxG portfolio
- CxG
- CxA
- CxT
- Action explorer
- Reports / diagnostics
- About methodology

Suggested demo flow: start with Overview, open Promoted CxG portfolio for the governed CxG model story, then use Player, Team, CxG, CxA, CxT, and Action explorer tabs for supporting generated outputs.

## Main Workflows

Full v1 reproduction:

```bash
make reproduce-v1
```

CxG:

```bash
make cxg-run
```

CxA:

```bash
make cxa-run
```

CxT:

```bash
make cxt-baseline
```

Database status:

```bash
make ingestion-report
```

The current v1 sample run writes 15,623 CxG shot predictions, 1,091,388 CxA action predictions, and 1,091,388 CxT action-threat rows to the local SQLite database.

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
|-- docs/                     # Release, dashboard, modelling, data, and output docs
|-- scripts/                  # Ingestion, feature, modelling, and validation commands
|-- src/opponent_adjusted/    # Package code
|-- tests/                    # Unit, contract, dashboard, and smoke tests
|-- feature_store/            # Generated features, ignored by Git
`-- outputs/                  # Generated model/report outputs, ignored by Git
```

## Generated Outputs

Generated files under `feature_store/` and `outputs/` are not committed. The local SQLite database under `data/` is also ignored. The repository tracks source code, contracts, tests, migrations, and documentation needed to regenerate them.

Ignored generated examples:

- `data/opponent_adjusted.db`
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
- [v1 results summary](docs/modeling/v1_results_summary.md)
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
