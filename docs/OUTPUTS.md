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

The generated `contextual_model.joblib` and `contextual_model.json` pair is the local artifact contract used by `/predict/cxg`. The endpoint returns a controlled 501 response until these generated files exist and the metadata contains the required API inference fields.

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

Generate the CxG validation reports:

```bash
poetry run python scripts/validate_cxg_outputs.py
```

Expected generated validation files:

```text
outputs/modeling/cxg/reports/validation_summary.json
outputs/modeling/cxg/reports/calibration_table.csv
outputs/modeling/cxg/reports/slice_metrics.csv
```

These validation reports summarize main CxG metrics, fold/grouped validation where available, calibration bins, slice metrics, and baseline comparison when a provider or baseline xG column is present. If no baseline column is present, the summary records that the comparison was skipped rather than inventing one.

The same regeneration path is available as a Make target:

```bash
make cxg-smoke
```

## Regenerate CxA Baseline Outputs

CxA has a first event-data baseline path. It is not the final attribution system.

Build action features:

```bash
poetry run python scripts/run_cxa_pipeline.py
```

Train, evaluate, score and export the baseline model:

```bash
poetry run python scripts/run_cxa_end_to_end.py
```

Expected generated CxA files:

```text
feature_store/cxa/action_features.parquet
outputs/modeling/cxa/models/baseline_model.joblib
outputs/modeling/cxa/models/baseline_model.json
outputs/modeling/cxa/reports/metrics.json
outputs/modeling/cxa/reports/attribution_summary.json
outputs/modeling/cxa/predictions/action_predictions.parquet
outputs/modeling/cxa/aggregates/player_cxa.parquet
outputs/modeling/cxa/aggregates/team_cxa.parquet
outputs/modeling/cxa/aggregates/sequence_cxa.parquet
```

Generated CxA outputs remain ignored by Git under the existing `feature_store/` and `outputs/` rules.

## Regenerate CxT Baseline Outputs

CxT has a leakage-safe baseline zone/grid model. It is an explainable baseline, not CxT+, Advanced CxT, OD-CxT, or a production-grade threat model.

Run:

```bash
poetry run python scripts/run_cxt_pipeline.py
```

Expected generated CxT baseline files:

```text
feature_store/cxt/action_features.parquet
outputs/modeling/cxt/threat_grid.parquet
outputs/modeling/cxt/predictions/action_threat.parquet
outputs/modeling/cxt/aggregates/player_cxt.parquet
outputs/modeling/cxt/aggregates/team_cxt.parquet
outputs/modeling/cxt/aggregates/sequence_cxt.parquet
outputs/modeling/cxt/reports/metrics.json
outputs/modeling/cxt/reports/zone_transition_summary.csv
outputs/modeling/cxt/reports/zone_transition_summary.parquet
outputs/modeling/cxt/reports/top_actions.csv
outputs/modeling/cxt/reports/interpretation_summary.json
```

Optional CSV mirrors can be generated with `--write-csv`:

```text
outputs/modeling/cxt/predictions/action_threat.csv
outputs/modeling/cxt/aggregates/player_cxt.csv
outputs/modeling/cxt/aggregates/team_cxt.csv
outputs/modeling/cxt/aggregates/sequence_cxt.csv
```

The baseline calculates `cxt_value = end_threat - start_threat` from a deterministic pitch grid. Player/team/sequence aggregates summarize who and which possessions add threat; zone-transition and top-action reports explain where threat is created. Generated CxT outputs remain ignored by Git under the existing `feature_store/` and `outputs/` rules. CxT+, Advanced CxT, and OD-CxT remain future roadmap items.

## What Should Be Committed

Commit source code, documentation, tests, migrations, and curated configs such as:

- `configs/**/*.json`
- `tests/fixtures/**`
- `docs/modeling/cxg/model_card.md`

Do not commit generated model artifacts, feature stores, bulk reports, or regenerated CSV/parquet outputs unless a future PR explicitly defines a small curated example fixture.

## Dashboard Consumption

The Streamlit dashboard v1 reads generated outputs through `configs/dashboard/v1_dashboard_contract.json`.

Run:

```bash
make dashboard
```

or:

```bash
poetry run streamlit run app/streamlit_app.py
```

The dashboard is a demo/portfolio shell, not a production deployment. It starts even when generated outputs are missing and shows availability status plus empty tables for missing files. Regenerate CxG, CxA, and CxT outputs locally to populate the dashboard.
