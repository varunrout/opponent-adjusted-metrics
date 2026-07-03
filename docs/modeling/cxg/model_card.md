# CxG Model Card

## Status

CxG is implemented as a reproducible v1 contextual expected-goals baseline. It trains a sklearn-compatible contextual logistic model, writes local file outputs, and persists model outputs to the SQLite database.

This is a pragmatic event-data baseline, not a claim of production-grade calibration or betting-grade forecasting.

## Objective

CxG estimates the probability that a shot becomes a goal using shot location, shot descriptors, match context, possession context, and opponent/profile proxy features available from event data.

## Target

- Target: `is_goal`
- Current v1 sample rows: 15,623 shots
- Current v1 sample goals: 1,639
- Current v1 goal rate: 0.1049

## Run Command

```bash
make cxg-run
```

Expanded commands:

```bash
poetry run python scripts/run_cxg_pipeline.py
poetry run python scripts/run_cxg_end_to_end.py
poetry run python scripts/check_cxg_outputs.py
poetry run python scripts/validate_cxg_outputs.py
```

## Feature Families

The current CxG pipeline uses event-derived shot features:

- Geometry: shot distance, angle, centrality, distance to goal line.
- Shot descriptors: body part, technique, shot type, first-time flag, blocked flag.
- Match context: period, minute, minute bucket, score state.
- Possession context: possession sequence length, duration, previous action gap.
- Pressure/opponent proxy context: under-pressure flag, recent defensive action count, pressure proxy, opponent defensive profile features where available.

No tracking data is required.

`statsbomb_xg` is retained as a provider reference/benchmark column only. It is excluded from both baseline and diagnostic training feature sets.

## Current V1 Metrics

From `outputs/modeling/cxg/baseline/reports/metrics.json`:

| Metric | Value |
|---|---:|
| rows | 15,623 |
| folds | 5 |
| Brier mean | 0.073251 |
| Log loss mean | 0.261105 |
| ROC AUC mean | 0.810041 |

From `outputs/modeling/cxg/baseline/reports/validation_summary.json`:

| Validation metric | Value |
|---|---:|
| mean predicted CxG | 0.105071 |
| Brier | 0.073248 |
| Log loss | 0.261096 |
| ROC AUC | 0.809201 |
| calibration bins | 10 |
| mean absolute calibration error | 0.042051 |
| grouped validation matches | 606 |

Provider `statsbomb_xg` was available for baseline comparison in the current run:

| Baseline metric | Value |
|---|---:|
| baseline column | `statsbomb_xg` |
| mean provider xG | 0.105949 |
| Brier | 0.072433 |
| Log loss | 0.257550 |
| ROC AUC | 0.819568 |

Provider xG slightly outperformed this baseline CxG on the current sample. That is expected for a simple reproducible model and should be read as a calibration benchmark, not a failure of the pipeline.

## Outputs

Generated files:

```text
feature_store/cxg/shot_features.parquet
outputs/modeling/cxg/baseline/models/contextual_model.joblib
outputs/modeling/cxg/baseline/models/contextual_model.json
outputs/modeling/cxg/baseline/reports/metrics.json
outputs/modeling/cxg/baseline/reports/validation_summary.json
outputs/modeling/cxg/baseline/reports/calibration_table.csv
outputs/modeling/cxg/baseline/reports/slice_metrics.csv
outputs/modeling/cxg/baseline/reports/model_card.md
outputs/modeling/cxg/baseline/predictions/shot_predictions.parquet
outputs/modeling/cxg/baseline/aggregates/player_cxg.parquet
outputs/modeling/cxg/baseline/aggregates/team_cxg.parquet
```

DB persistence:

- `model_registry`: model row for `cxg_contextual_20260628`.
- `shot_predictions`: 15,623 rows in the current v1 sample run.
- `aggregates_player`: CxG player aggregate rows.
- `aggregates_team`: CxG team aggregate rows.
- `evaluation_metrics`: 20 CxG metric rows in the current v1 sample run.

## API Compatibility

The generated metadata sidecar at `outputs/modeling/cxg/baseline/models/contextual_model.json` is the local artifact contract used by `src/opponent_adjusted/api/cxg_inference.py`.

The `/predict/cxg` endpoint requires generated local model artifacts. Without them it returns a controlled unavailable response.

## Limitations

- The model uses event-derived features only; it does not use tracking data.
- Opponent defensive ratings are proxies and should be interpreted as contextual adjustments, not causal defensive quality.
- Provider xG is currently a strong baseline and outperforms this simple CxG model on the current sample.
- Player/team aggregates can be unstable for small shot samples.
- The runner is designed for reproducibility and portfolio clarity, not production deployment.
