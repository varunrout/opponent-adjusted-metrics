# Diagnostic-Informed CxG Training

The diagnostic-informed CxG training layer sits after pre-model CxG analysis
and before full validation or prediction promotion:

```text
raw / normalized data
-> feature engineering
-> pre-model diagnostic analysis
-> diagnostic-informed modelling decisions
-> training
-> validation
-> promoted predictions/results
```

Run:

```bash
poetry run python scripts/run_cxg_diagnostic_training.py
```

or:

```bash
make run-cxg-diagnostic-training
```

This does not replace the baseline CxG path (`run-cxg-end-to-end`, `cxg-run`,
and `outputs/modeling/cxg/`). It writes a separate diagnostic training surface:

```text
outputs/modeling/cxg/diagnostic_v1/
```

The training script uses `configs/feature_contracts/cxg_diagnostic_v1.json` to
resolve eligible numeric, binary, and categorical features from the shot-feature
input. Reference-only columns such as provider xG and leakage/post-model columns
are excluded from model matrices.

Candidate models:

- `geometry_logistic`: simple location/geometry logistic baseline.
- `diagnostic_logistic`: regularised logistic regression over the diagnostic feature contract.
- `gradient_boosting`: deterministic nonlinear benchmark.
- `extra_trees`: deterministic tree ensemble robustness benchmark.

Outputs:

- `feature_contract.json`: resolved contract copy with provisional selected model.
- `model_candidates.json`: candidate definitions and resolved feature groups.
- `model_comparison.csv`: aggregate training comparison metrics.
- `fold_metrics.csv`: fold-level Brier, log loss, ROC AUC where valid, and support metrics.
- `selected_model_metadata.json`: selected candidate metadata and feature decisions.
- `selected_model.joblib`: final selected model fit on all training rows.
- `cross_validated_predictions.parquet`: fold predictions for every candidate.
- `training_report.md`: training explanation and validation handoff notes.

Issue #56 should perform full validation, calibration, and slice checks on this
selected model. Issue #57 should handle final promoted predictions/results.
