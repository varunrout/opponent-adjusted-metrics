# CxG Model Card

## Status

CxG now has a reproducible end-to-end runner for local feature tables and fixture/synthetic test data. The implementation is a pragmatic sklearn logistic-regression baseline, not a claim of production-grade football forecasting.

## One-command run

```bash
poetry run python scripts/run_cxg_end_to_end.py
```

The command auto-discovers CxG feature outputs in `feature_store/cxg/`, preferring `shot_features.parquet` and falling back to other existing CxG datasets. To use a specific dataset:

```bash
poetry run python scripts/run_cxg_end_to_end.py --input feature_store/cxg/shot_features.parquet
```

All generated modeling outputs are written under `outputs/modeling/cxg/` by default.

## Inputs

The runner expects a shot-level table with:

- `is_goal` target labels.
- `match_id` for grouped evaluation.
- Any supported CxG contextual features already produced by the CxG feature pipeline.

When optional context columns are absent, the runner fills conservative neutral defaults so fixture tests and smaller datasets remain reproducible.

## Outputs

The end-to-end run emits:

- `outputs/modeling/cxg/models/contextual_model.joblib` — sklearn-compatible model artifact loadable with joblib.
- `outputs/modeling/cxg/models/contextual_model.json` — API-compatible metadata with model version, artifact path, created timestamp, features, metrics and output paths.
- `outputs/modeling/cxg/predictions/shot_predictions.parquet` — shot-level raw, neutral and opponent-adjusted CxG scores.
- `outputs/modeling/cxg/aggregates/player_cxg.parquet` — player aggregates.
- `outputs/modeling/cxg/aggregates/team_cxg.parquet` — team aggregates.
- `outputs/modeling/cxg/reports/metrics.json` — cross-validated metrics.
- `outputs/modeling/cxg/reports/model_card.md` — run-specific model card/report.

## Evaluation

Evaluation uses deterministic cross-validation with match grouping when possible. The exported metrics currently include mean Brier score, mean log loss, mean ROC AUC and fold-level rows.

## Neutral and opponent-adjusted scoring

The runner exports:

- `cxg_raw`: model score in observed context.
- `cxg_neutral`: model score after neutralizing score state, minute bucket and opponent defensive profile proxies.
- `cxg_opp_adjusted_diff`: raw minus neutral CxG.
- `cxg_opp_adjusted_ratio`: raw divided by neutral CxG.

## API compatibility

The metadata sidecar is intentionally written next to the joblib artifact at `outputs/modeling/cxg/models/contextual_model.json`, matching the discovery path used by `src/opponent_adjusted/api/cxg_inference.py`.

## Limitations

- The model uses event-derived features only; it does not use tracking data.
- Opponent defensive ratings are proxies and should be interpreted as contextual adjustments, not causal defensive quality.
- Small or biased samples can produce unstable player/team aggregates.
- The runner is designed for reproducibility and portfolio clarity, not production deployment or betting use.
