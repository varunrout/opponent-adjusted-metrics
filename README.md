# Opponent-Adjusted Football Metrics

Portfolio/demo release for contextual football analytics built on StatsBomb-style event data.

**v1.0.0 status:** CxG and CxA portfolio layers are dashboard-ready, CxT analysis is available with modelling/promotion pending, and Streamlit dashboard v1 is implemented. Generated outputs and the local SQLite database are reproducible locally and intentionally ignored by Git.

This is not a production deployment. It is a reviewable v1 analytics product surface that shows how raw events become feature tables, model outputs, reports, database rows, and football insight.

## Current Project Status

| Metric layer | Current status | Portfolio/dashboard status | Notes |
|---|---|---|---|
| CxG | Promoted | Portfolio/dashboard-ready | Fair baseline excludes StatsBomb xG leakage; diagnostic CxG is the governed shot-quality model. |
| CxA | Provisionally promoted | Portfolio/dashboard-ready | Baseline comparator is reference-only/in-sample, so the promotion caveat stays visible. |
| CxT | Analysis completed | Modelling pending | Pre-model analysis exists; Baseline CxT modelling/promotion and portfolio outputs are not complete yet. |
| CxA+ | Pending | Not implemented | Future extension after the current CxA shot-creation probability layer. |
| Advanced CxA | Pending | Not implemented | Future extension; not a current dashboard claim. |

## What This Project Demonstrates

- Governed football metric development from raw events to portfolio outputs.
- Leakage-aware feature contracts and reference-only field separation.
- Baseline vs diagnostic model comparison with explicit validation caveats.
- Model promotion gates that can promote, provisionally promote, block, or defer.
- Portfolio-ready metric reporting with player/team names and static charts.
- Streamlit dashboard presentation over generated, reproducible local artifacts.

## Promoted Diagnostic CxG Portfolio

The promoted diagnostic CxG model is the governed shot-quality layer for the portfolio. It scores shot-level CxG from pre-shot, leakage-safe features, then rolls those predictions into player, team, feature-driver, and category insight outputs.

The fair CxG baseline excludes StatsBomb xG as a training feature. Against that fair baseline, the promoted diagnostic model improves log loss, Brier score, and ROC AUC; expected calibration error remains transparently monitored because the baseline is slightly better calibrated in the latest run.

Portfolio entry points (generated into the gitignored `outputs/` tree, so they are not tracked in this repo; regenerate them with the commands below):

- CxG portfolio summary: `outputs/portfolio/cxg/cxg_portfolio_summary.md`
# Opponent-Adjusted Metrics Productionisation

This repository contains the current GCP productionisation baseline for opponent-adjusted metrics. The implemented scope is StatsBomb Bronze ingestion, 360 coverage auditing, the Silver transformation, `oam_core` publishing, and the Terraform foundation.

Install dependencies and run the retained validation suite:

```bash
poetry install
poetry run pytest -q
```

Operational entry points are in `scripts/` for subset retrieval, 360 auditing, Silver building, and `oam_core` publishing. The historical model-family, dashboard, and database implementation remains available through Git history and the `pre-gcp-productionisation-2026-08-19` tag; no model-family implementation is present in this clean tree.
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
