# Diagnostic CxG Validation

Diagnostic CxG validation compares two generated modelling surfaces:

- `outputs/modeling/cxg/baseline/`
- `outputs/modeling/cxg/diagnostic_v1/`

Run:

```bash
make validate-cxg-diagnostic
```

The validation command does not retrain models. It reads the baseline
predictions and the selected diagnostic cross-validated predictions, then writes
validation outputs to:

```text
outputs/validation/cxg/diagnostic_v1/
```

## Comparison Method

The diagnostic prediction file contains rows for every candidate model. The
validator reads `models/selected_model_metadata.json` and filters
`predictions/cross_validated_predictions.parquet` to the selected candidate only.
The baseline is compared against that selected diagnostic model, not against a
combined candidate table.

## Why Calibration Matters

ROC AUC measures rank ordering: whether better chances tend to receive higher
scores than worse chances. CxG also needs calibrated probabilities, because a
0.20 chance should behave like a goal roughly 20 percent of the time across
large samples. The validator therefore reports log loss, Brier score, expected
calibration error, and calibration bins alongside ROC AUC.

## Slice Stability

The validator checks available football slices such as body part, shot type,
pressure state, score state, and set-piece context. Optional slice columns are
allowed to be missing; missing columns are recorded in
`validation_summary.json`. Sparse slices are labelled so they are interpreted
directionally rather than treated as firm promotion evidence.

## Promotion Logic

The promotion recommendation uses probability quality and governance, not ROC
AUC alone. A model can be promoted only when it matches or improves log loss and
Brier score, has acceptable calibration, is stable across folds, has no
forbidden leakage/reference features, and has valid probability bounds. Mixed
but acceptable results can receive `provisional_promote`; unstable, poorly
calibrated, or leakage-affected results are marked for revision or rejection.

## Handoff To #57

This validation layer decides whether `diagnostic_v1` is ready for promoted
prediction generation. Issue #57 should only handle final prediction/result
promotion and player/team reporting if this validation recommends `promote` or
`provisional_promote`.
