# Generated Outputs

Generated outputs are intentionally not tracked by Git. They can be recreated from source code, configs, fixtures, and raw StatsBomb data, so the repository keeps only the instructions and versioned contracts needed to reproduce them.

The `outputs/` and `feature_store/` directories are generated, ignored by Git, and should not be committed as routine documentation or model evidence.

Ignored generated locations include:

- `feature_store/cxg/`
- `outputs/modeling/cxg/`
- Other files under `feature_store/` and `outputs/`

Ignored generated file types include:

- `*.parquet`
- `*.csv`
- `*.joblib`
- `*.pkl`
- `*.pickle`

## Regenerate CxG Feature Outputs

From a clean checkout with dependencies installed and source data/database prepared, run:

```bash
poetry run python scripts/run_cxg_pipeline.py
```

Expected generated directory:

```text
feature_store/cxg/
```

This command builds the CxG feature outputs used by the downstream baseline runner. The exact files depend on the configured data subset and pipeline settings.

## Regenerate CxG Modeling Outputs

Then run:

```bash
poetry run python scripts/run_cxg_end_to_end.py
```

Expected generated directory:

```text
outputs/modeling/cxg/
```

This command trains/evaluates the current CxG baseline path and exports generated artifacts such as model files, metadata, metrics, prediction outputs, and aggregate outputs.

Expected generated files:

```text
outputs/modeling/cxg/models/contextual_model.joblib
outputs/modeling/cxg/models/contextual_model.json
outputs/modeling/cxg/reports/metrics.json
outputs/modeling/cxg/predictions/shot_predictions.parquet
outputs/modeling/cxg/aggregates/player_cxg.parquet
outputs/modeling/cxg/aggregates/team_cxg.parquet
outputs/modeling/cxg/reports/model_card.md
```

Validate the local output contract and Git ignore rules:

```bash
poetry run python scripts/check_cxg_outputs.py
```

The same regeneration path is available as a Make target:

```bash
make cxg-smoke
```

## What Should Be Committed

Commit source code, documentation, tests, migrations, and curated configs such as:

- `configs/**/*.json`
- `tests/fixtures/**`
- `docs/modeling/cxg/model_card.md`

Do not commit generated model artifacts, feature stores, bulk reports, or regenerated CSV/parquet outputs unless a future PR explicitly defines a small curated example fixture.
