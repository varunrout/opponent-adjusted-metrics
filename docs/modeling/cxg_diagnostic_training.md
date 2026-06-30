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

This does not replace the baseline CxG path (`run-cxg-end-to-end`, `cxg-run`).
CxG modelling outputs are versioned under:

```text
outputs/modeling/cxg/baseline/
outputs/modeling/cxg/diagnostic_v1/
```

`baseline` is the existing CxG training layer. `diagnostic_v1` is the
diagnostic-informed CxG training layer that uses decisions from pre-model CxG
analysis.

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

- `contracts/feature_contract.json`: resolved contract copy with provisional selected model.
- `diagnostics/excluded_columns.csv`: leakage and reference-only columns present in input.
- `diagnostics/resolved_features.json`: resolved feature lists and unavailable optional columns.
- `diagnostics/feature_group_summary.csv`: feature availability by contract feature group.
- `models/model_candidates.json`: candidate definitions and resolved feature groups.
- `models/selected_model_metadata.json`: selected candidate metadata and feature decisions.
- `models/selected_model.joblib`: final selected model fit on all training rows.
- `predictions/cross_validated_predictions.parquet`: fold predictions for every candidate.
- `reports/model_comparison.csv`: aggregate training comparison metrics.
- `reports/fold_metrics.csv`: fold-level Brier, log loss, ROC AUC where valid, and support metrics.
- `reports/training_summary.json`: compact machine-readable training summary.
- `reports/training_report.md`: training explanation and validation handoff notes.

Issue #56 should perform full validation, calibration, and slice checks on this
selected model. Issue #57 should handle final promoted predictions/results.
