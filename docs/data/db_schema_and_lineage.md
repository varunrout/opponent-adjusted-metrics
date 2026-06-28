# DB Schema And Lineage

This document describes how v1 data moves from raw StatsBomb-style events into engineered features, model outputs, aggregate reports, and database tables.

## Lineage Overview

```text
raw_events
-> events
-> event detail tables: passes, carries, dribbles, pressures, ball_receipts, clearances, duels, blocks, interceptions
-> possessions
-> shots
-> shot_features
-> action_features
-> model_registry
-> shot_predictions / action_predictions / action_threat_predictions
-> aggregates_player / aggregates_team / aggregates_sequence
-> evaluation_metrics
```

## Table Families

### Raw And Normalized Events

- `raw_events` stores raw StatsBomb-style event JSON and basic event metadata.
- `events` stores normalized event rows with canonical IDs, event type, period, clock, team/player IDs, possession number, location, pressure flag, and outcome.
- Detail tables such as `passes`, `carries`, `dribbles`, `pressures`, and `ball_receipts` store type-specific fields linked back to `events`.

These are event/action tables. They describe what happened on the pitch and are the source for downstream feature engineering.

### Possessions And Shots

- `possessions` is built incrementally from normalized `events` using match and StatsBomb possession number.
- `shots` stores shot-level facts linked to shot events, including team, player, opponent team, shot descriptors, provider xG where available, and outcome.

Possessions are a reusable context layer. Shots are the CxG modelling grain.

### Engineered Feature Tables

- `shot_features` stores CxG input features such as geometry, score state, possession context, recent defensive-action count, and pressure proxy.
- `action_features` stores engineered CxA action features from `feature_store/cxa/action_features.parquet`, including action IDs, match/team/player IDs, sequence IDs, start/end locations, movement/progression fields, and explicit target/evaluation fields.

These tables are model input layers. Target/reference fields are named as targets or evaluation references and should not be confused with leakage-safe input features.

### Model Registry

- `model_registry` records the model family, version, algorithm, artifact path, and calibration/metric metadata.

The current v1 sample run contains three model families:

| Model | Version | Artifact |
|---|---|---|
| cxg | `cxg_contextual_20260628` | `outputs/modeling/cxg/models/contextual_model.joblib` |
| cxa | `cxa_baseline_20260628` | `outputs/modeling/cxa/models/baseline_model.joblib` |
| cxt | `cxt-baseline-v1` | `outputs/modeling/cxt/threat_grid.parquet` |

The registry separates CxG, CxA, and CxT model versions so their prediction and aggregate rows can coexist.

### Prediction Tables

- `shot_predictions` stores CxG shot-level predictions.
- `action_predictions` stores CxA action-level predictions.
- `action_threat_predictions` stores CxT action-level threat deltas.

These are model output tables. They should be persisted idempotently by model/version and should not be used as raw event sources.

### Aggregate And Metric Tables

- `aggregates_player` stores model-specific player aggregates.
- `aggregates_team` stores model-specific team aggregates.
- `aggregates_sequence` stores CxA/CxT sequence-level aggregates.
- `evaluation_metrics` stores model evaluation/report metrics.

These tables are reporting surfaces. They support the dashboard and reviewer inspection.

## Current V1 Sample Counts

The current post-PR43 local sample run reports:

| Table | Rows |
|---|---:|
| raw_events | 2,143,146 |
| events | 2,143,146 |
| possessions | 110,432 |
| shots | 15,623 |
| shot_features | 15,623 |
| action_features | 1,091,388 |
| shot_predictions | 15,623 |
| action_predictions | 1,091,388 |
| action_threat_predictions | 1,091,388 |
| aggregates_player | 5,503 |
| aggregates_team | 222 |
| aggregates_sequence | 107,626 |
| evaluation_metrics | 78 |

These counts are examples from the current v1 sample run and may change with a different data subset.

## Idempotency

Feature and model-output persistence is designed to be idempotent:

- Re-running CxA action feature building replaces CxA action feature rows for the same feature family/version.
- Re-running CxG, CxA, or CxT model persistence replaces that model/version's prediction, aggregate, and metric rows.
- CxA/CxT persistence does not delete CxG rows, and CxG persistence does not delete CxA/CxT rows.

This lets reviewers regenerate a single modelling family without rebuilding the entire database.

## File Lineage

| File | DB surface |
|---|---|
| `feature_store/cxg/shot_features.parquet` | feeds `shot_features`/CxG model input |
| `feature_store/cxa/action_features.parquet` | `action_features` |
| `outputs/modeling/cxg/predictions/shot_predictions.parquet` | `shot_predictions` |
| `outputs/modeling/cxa/predictions/action_predictions.parquet` | `action_predictions` |
| `outputs/modeling/cxt/predictions/action_threat.parquet` | `action_threat_predictions` |
| `outputs/modeling/*/aggregates/player_*.parquet` | `aggregates_player` |
| `outputs/modeling/*/aggregates/team_*.parquet` | `aggregates_team` |
| `outputs/modeling/cxa/aggregates/sequence_cxa.parquet` | `aggregates_sequence` |
| `outputs/modeling/cxt/aggregates/sequence_cxt.parquet` | `aggregates_sequence` |
| `outputs/modeling/*/reports/metrics.json` | `evaluation_metrics` |

Generated files and DB rows represent the same modelling pipeline but are not always one-to-one row mirrors. Database tables apply model/version uniqueness and idempotent persistence rules.
