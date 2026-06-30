# Diagnostic CxG Results

Diagnostic CxG results are the final promoted output layer for `diagnostic_v1`.
The command consumes the trained model from issue #55 and the validation
recommendation from issue #56, then writes shot, player, and team result files
for portfolio or dashboard use.

## Command

```bash
make generate-cxg-diagnostic-results
```

By default, promoted outputs are written only when
`outputs/validation/cxg/diagnostic_v1/promotion_recommendation.json` recommends
`promote` or `provisional_promote`. If validation returns `needs_revision` or
`do_not_promote`, the command writes a blocked `model_promotion_summary.json`
and `cxg_results_report.md` without pretending the model is promoted.

For exploratory analysis only:

```bash
poetry run python scripts/generate_cxg_diagnostic_results.py --allow-non-promoted
```

## How Predictions Are Generated

The script loads:

- `selected_model.joblib`
- `selected_model_metadata.json`
- `feature_contract.json`
- diagnostic governance artifacts
- validation promotion artifacts
- the CxG shot feature table

Feature selection is driven by the selected model metadata recorded during
training. The results layer does not re-resolve the contract or train a new
model.

## Promotion Gate

Promotion is controlled by validation:

- `promote` -> `promotion_status = promoted`
- `provisional_promote` -> `promotion_status = provisionally_promoted`
- `needs_revision` or `do_not_promote` -> blocked by default

Feature governance artifacts must be present, and selected features must not use
leakage/reference columns or synthetic default features that training excluded.

## Player and Team Summaries

Shot predictions are aggregated to player and team summaries:

- `shots`
- `goals`
- `total_cxg`
- `mean_cxg_per_shot`
- `goals_minus_cxg`
- ranks by total and mean CxG

When baseline shot predictions can be joined safely, the summaries include
baseline total CxG and diagnostic-vs-baseline deltas.

## Baseline Comparison

Baseline predictions are joined by `shot_id` where possible, then by `event_id`,
then by a conservative multi-column fallback. Missing or unjoinable baseline
outputs do not fail result generation; they are recorded in the promotion summary
and quality checks.

## Dashboard Use

The result files under `outputs/results/cxg/diagnostic_v1/` are the stable
portfolio-facing surface. Dashboard code should prefer these promoted results
over training cross-validation predictions or validation comparison tables.
