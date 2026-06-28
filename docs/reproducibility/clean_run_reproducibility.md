# Clean Run Reproducibility

Clean local reproducibility means a reviewer can start from a fresh checkout, regenerate the local SQLite database, rebuild engineered features, run the v1 model families, and produce dashboard-readable outputs without committing generated data.

Generated outputs are local artifacts. The repository commits code, migrations, contracts, tests, and docs; it does not commit `data/opponent_adjusted.db`, `feature_store/`, or `outputs/`.

## Prerequisites

- Python dependencies installed with Poetry.
- Network access for the configured StatsBomb Open Data subset when running `make fetch-data`.
- Enough local disk space for the SQLite database, feature parquet files, and model outputs.

```bash
poetry install
```

## Full V1 Command

```bash
make reproduce-v1
```

Then verify status:

```bash
make ingestion-report
```

The report is written to:

```text
outputs/reports/ingestion/db_status.json
```

## Expanded Pipeline

`make reproduce-v1` expands to the v1 pipeline below:

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

The sequence matters:

- Migrations create the DB schema.
- Fetch/ingest/normalize populate raw and normalized event tables.
- Possession and feature builders materialize reusable feature layers.
- CxG, CxA, and CxT runners write both files and DB-backed model outputs.
- The ingestion report checks that core generated tables are populated.

## Current V1 Sample Run Indicators

The current post-PR43 local sample run reported these counts:

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
| aggregates_sequence | 107,626 |
| evaluation_metrics | 78 |

Treat these as an example from the current v1 sample run. They may change if the configured data subset changes.

Expected readiness flags from the current run are all true:

- `has_competitions`
- `has_matches`
- `has_raw_events`
- `has_normalized_events`
- `has_possessions`
- `has_shots`
- `has_shot_features`
- `has_action_features`
- `has_model_registry`
- `has_predictions`
- `has_cxg_predictions`
- `has_cxa_predictions`
- `has_cxt_predictions`
- `has_sequence_aggregates`

The current model registry contains:

| Model | Version | Algorithm |
|---|---|---|
| cxg | `cxg_contextual_20260628` | `contextual_logistic` |
| cxa | `cxa_baseline_20260628` | `baseline_action_classifier` |
| cxt | `cxt-baseline-v1` | `baseline_grid_threat` |

## Context Checks

The current shot-feature context check shows that the possession/context columns are populated:

| Column | Nulls | Zeros | Min | Avg | Max |
|---|---:|---:|---:|---:|---:|
| possession_sequence_length | 0 | 0 | 1 | 27.929 | 265 |
| possession_duration | 0 | 206 | 0.0 | 29.847 | 2,946.0 |
| previous_action_gap | 0 | 8,931 | 0.0 | 0.782 | 12.0 |
| recent_def_actions_count | 0 | 6,611 | 0 | 0.763 | 4 |
| pressure_proxy_score | 0 | 112 | -0.575 | -0.073 | 1.378 |

Zeros are valid for several columns. For example, many shots have no recent defensive action in the lookback window.

## Generated Outputs

After a successful run, files are expected under:

```text
feature_store/cxg/
feature_store/cxa/
feature_store/cxt/
outputs/modeling/cxg/
outputs/modeling/cxa/
outputs/modeling/cxt/
outputs/reports/ingestion/
```

These paths are ignored by Git. Do not commit regenerated parquet, CSV, joblib, SQLite, or bulk output files.
