# Generated Outputs

Generated outputs are intentionally not tracked by Git. They can be recreated from source code, configs, fixtures, and raw StatsBomb data, so the repository keeps only the instructions and versioned contracts needed to reproduce them.

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

Run:

```bash
poetry run python scripts/run_cxg_pipeline.py
```

Expected generated directory:

```text
feature_store/cxg/
```

This command builds the CxG feature outputs used by the downstream baseline runner. The exact files depend on the configured data subset and pipeline settings.

## Regenerate CxG Modeling Outputs

Run:

```bash
poetry run python scripts/run_cxg_end_to_end.py
```

Expected generated directory:

```text
outputs/modeling/cxg/
```

This command trains/evaluates the current CxG baseline path and exports generated artifacts such as model files, metadata, metrics, prediction outputs, and aggregate outputs.

## What Should Be Committed

Commit source code, documentation, tests, migrations, and curated configs such as:

- `configs/**/*.json`
- `tests/fixtures/**`
- `docs/modeling/cxg/model_card.md`

Do not commit generated model artifacts, feature stores, bulk reports, or regenerated CSV/parquet outputs unless a future PR explicitly defines a small curated example fixture.
