# Generated Outputs

Generated outputs are intentionally not tracked by Git. They can be recreated from source code, configs, fixtures, migrations, and raw StatsBomb data. The repository keeps only the instructions and versioned contracts needed to reproduce them.

The generated local surfaces are:

- `data/opponent_adjusted.db` for SQLite database state.
- `feature_store/` for engineered feature parquet files.
- `outputs/` for model artifacts, predictions, aggregates, metrics, validation reports, and dashboard-readable summaries.
- `outputs/analysis/` for pre-model analysis artifacts that inform modelling decisions before
  any predictions or aggregate model outputs are produced.

## Current V1 Sample Run

The current post-PR43 v1 sample run reported these local DB counts in `outputs/reports/ingestion/db_status.json`:

| Table | Rows |
|---|---:|
| competitions | 5 |
| teams | 74 |
| players | 2,038 |
| matches | 610 |
| raw_events | 2,143,146 |
| events | 2,143,146 |
| possessions | 110,432 |
| shots | 15,623 |
| shot_features | 15,623 |
| action_features | 1,091,388 |
| model_registry | 3 |
| shot_predictions | 15,623 |
| action_predictions | 1,091,388 |
| action_threat_predictions | 1,091,388 |
| aggregates_player | 5,503 |
| aggregates_team | 222 |
| aggregates_sequence | 107,626 |
| evaluation_metrics | 78 |

These counts are examples from the current v1 sample run, not hard-coded acceptance thresholds.

## Regenerate Everything

```bash
make reproduce-v1
make ingestion-report
```

Expanded pipeline:

```bash
make migrate-up
make fetch-data
make ingest-all
make normalize-events
make build-possessions
make build-features
make build-profiles
make build-cxa-action-features
make run-cxg-pipeline
make run-cxg-end-to-end
make run-cxa-end-to-end
make run-cxt-pipeline
make ingestion-report
```

Generated files and the SQLite database remain ignored by Git.

## CxG Outputs

### Pre-model CxG Analysis

Command:

```bash
make analysis-cxg
```

This layer sits between feature engineering and model training:

```text
raw / normalized events
-> feature engineering
-> CxG analysis layer
-> CxG modelling decisions / model training
-> post-model prediction reporting
```

It reads `shot_features` joined to `shots`. Optional provider xG columns, such as
`statsbomb_xg`, are treated only as external benchmarks/reference variables. It does
not read `shot_predictions`, `model_registry`, CxG aggregate model outputs,
opponent-adjusted CxG outputs, or prediction leaderboards.

Generated files:

```text
outputs/analysis/cxg/
outputs/analysis/cxg/00_target/target_summary.csv
outputs/analysis/cxg/00_target/target_balance.png
outputs/analysis/cxg/01_feature_distributions/feature_missingness.csv
outputs/analysis/cxg/01_feature_distributions/numeric_distributions.png
outputs/analysis/cxg/01_feature_distributions/categorical_top_levels.csv
outputs/analysis/cxg/02_feature_target_relationships/numeric_target_relationships.csv
outputs/analysis/cxg/02_feature_target_relationships/categorical_target_relationships.csv
outputs/analysis/cxg/02_feature_target_relationships/numeric_target_relationships.png
outputs/analysis/cxg/03_feature_correlations/numeric_correlations.csv
outputs/analysis/cxg/03_feature_correlations/high_correlations.csv
outputs/analysis/cxg/03_feature_correlations/correlation_heatmap.png
outputs/analysis/cxg/04_slice_stability/slice_stability.csv
outputs/analysis/cxg/04_slice_stability/slice_stability.png
outputs/analysis/cxg/05_data_quality/data_quality.csv
outputs/analysis/cxg/05_data_quality/cleaning_recommendations.csv
outputs/analysis/cxg/06_leakage_checks/leakage_checks.csv
outputs/analysis/cxg/report.md
```

Every report section follows: Question -> Calculation -> Visual/Table ->
Interpretation -> Modelling implication -> Limitation.

### CxG Modelling

Command:

```bash
make cxg-run
```

Generated files:

```text
feature_store/cxg/shot_features.parquet
outputs/modeling/cxg/models/contextual_model.joblib
outputs/modeling/cxg/models/contextual_model.json
outputs/modeling/cxg/reports/metrics.json
outputs/modeling/cxg/reports/validation_summary.json
outputs/modeling/cxg/reports/calibration_table.csv
outputs/modeling/cxg/reports/slice_metrics.csv
outputs/modeling/cxg/reports/model_card.md
outputs/modeling/cxg/predictions/shot_predictions.parquet
outputs/modeling/cxg/aggregates/player_cxg.parquet
outputs/modeling/cxg/aggregates/team_cxg.parquet
```

DB persistence:

- `model_registry`: one `cxg` model row for `cxg_contextual_20260628` in the current sample run.
- `shot_predictions`: one row per scored shot.
- `aggregates_player` and `aggregates_team`: CxG player/team aggregate rows.
- `evaluation_metrics`: metrics from `metrics.json` and validation outputs.

## CxA Outputs

Command:

```bash
make cxa-run
```

Generated files:

```text
feature_store/cxa/action_features.parquet
feature_store/cxa/pipeline_metadata.json
outputs/modeling/cxa/models/baseline_model.joblib
outputs/modeling/cxa/models/baseline_model.json
outputs/modeling/cxa/reports/metrics.json
outputs/modeling/cxa/reports/attribution_summary.json
outputs/modeling/cxa/predictions/action_predictions.parquet
outputs/modeling/cxa/aggregates/player_cxa.parquet
outputs/modeling/cxa/aggregates/team_cxa.parquet
outputs/modeling/cxa/aggregates/sequence_cxa.parquet
```

DB persistence:

- `action_features`: engineered CxA action features from `feature_store/cxa/action_features.parquet`.
- `model_registry`: one `cxa` model row for `cxa_baseline_20260628` in the current sample run.
- `action_predictions`: action-level CxA predictions.
- `aggregates_player`, `aggregates_team`, and `aggregates_sequence`: CxA aggregate rows.
- `evaluation_metrics`: CxA model metrics.

`make cxa-smoke` only builds a small action-feature smoke dataset and does not populate the full CxA model outputs.

## CxT Outputs

Command:

```bash
make cxt-baseline
```

Generated files:

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

DB persistence:

- `model_registry`: one `cxt` model row for `cxt-baseline-v1` in the current sample run.
- `action_threat_predictions`: action-level baseline CxT rows from `outputs/modeling/cxt/predictions/action_threat.parquet`.
- `aggregates_player`, `aggregates_team`, and `aggregates_sequence`: CxT aggregate rows.
- `evaluation_metrics`: CxT report metrics.

Baseline CxT calculates `cxt_value = end_threat - start_threat` from a deterministic pitch grid. CxT+, Contextual CxT, Advanced CxT, OD-CxT, and OD-CxT+ remain future roadmap items.

## File To DB Mapping

| Generated file | DB table |
|---|---|
| `feature_store/cxa/action_features.parquet` | `action_features` |
| `outputs/modeling/cxg/models/contextual_model.joblib` and `.json` | `model_registry` |
| `outputs/modeling/cxa/models/baseline_model.joblib` and `.json` | `model_registry` |
| `outputs/modeling/cxt/threat_grid.parquet` | `model_registry` artifact path for baseline CxT |
| `outputs/modeling/cxg/predictions/shot_predictions.parquet` | `shot_predictions` |
| `outputs/modeling/cxa/predictions/action_predictions.parquet` | `action_predictions` |
| `outputs/modeling/cxt/predictions/action_threat.parquet` | `action_threat_predictions` |
| `outputs/modeling/cxg/aggregates/player_cxg.parquet` | `aggregates_player` |
| `outputs/modeling/cxg/aggregates/team_cxg.parquet` | `aggregates_team` |
| `outputs/modeling/cxa/aggregates/player_cxa.parquet` | `aggregates_player` |
| `outputs/modeling/cxa/aggregates/team_cxa.parquet` | `aggregates_team` |
| `outputs/modeling/cxa/aggregates/sequence_cxa.parquet` | `aggregates_sequence` |
| `outputs/modeling/cxt/aggregates/player_cxt.parquet` | `aggregates_player` |
| `outputs/modeling/cxt/aggregates/team_cxt.parquet` | `aggregates_team` |
| `outputs/modeling/cxt/aggregates/sequence_cxt.parquet` | `aggregates_sequence` |
| `outputs/modeling/*/reports/metrics.json` | `evaluation_metrics` |

The persistence layer is idempotent by model family/version. Re-running a model path replaces that model/version's generated DB rows without deleting other metric families.

## Dashboard Consumption

The Streamlit dashboard v1 reads generated outputs through `configs/dashboard/v1_dashboard_contract.json`.

```bash
make dashboard
```

The dashboard is a demo/portfolio shell, not a production deployment. It starts even when generated outputs are missing and shows availability status plus empty tables for missing files. Regenerate CxG, CxA, and CxT outputs locally to populate the analysis pages.

## What Should Be Committed

Commit source code, documentation, tests, migrations, and curated configs such as:

- `configs/**/*.json`
- `tests/fixtures/**`
- `docs/**/*.md`

Do not commit generated model artifacts, local SQLite DBs, feature stores, or regenerated CSV/parquet outputs unless a future PR explicitly defines a small curated example fixture.
