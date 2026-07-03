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

After validation, `diagnostic_v1` needed a modelling-strength revision:
validation found that the selected `diagnostic_logistic` model was cleaner and
better calibrated than baseline, but worse on log loss, Brier score, and ROC
AUC. The training layer therefore keeps the governance contract intact while
expanding the candidate set and making selection more directly probability-score
driven.

The training script uses `configs/feature_contracts/cxg_diagnostic_v1.json` to
resolve eligible numeric, binary, and categorical features from the shot-feature
input. Reference-only columns such as provider xG and leakage/post-model columns
are excluded from model matrices.

Candidate models:

- `geometry_logistic`: simple location/geometry logistic baseline.
- `diagnostic_logistic`: regularised logistic regression over the diagnostic feature contract.
- `diagnostic_baseline_parity_logistic`: governed all-feature logistic candidate configured for baseline-parity comparison.
- `calibrated_diagnostic_logistic_sigmoid`: diagnostic logistic candidate with sigmoid calibration inside each training fold.
- `gradient_boosting`: deterministic nonlinear benchmark.
- `calibrated_gradient_boosting_sigmoid`: gradient boosting with sigmoid calibration inside each training fold.
- `extra_trees`: deterministic tree ensemble robustness benchmark.
- `calibrated_extra_trees_sigmoid`: extra trees with sigmoid calibration inside each training fold.

Candidate selection is primary-score first: lowest mean log loss wins, Brier is
the first tie-breaker, fold-level calibration proxy is considered after those
primary probability metrics, and ROC AUC remains secondary context. Validation
still decides whether the selected model can be promoted.

## Source-Derived CxG Context Features

Follow-up validation showed the revised diagnostic candidates were governed and
better calibrated, but still worse than baseline on log loss and Brier. The
remaining bottleneck was feature signal rather than leakage, fold instability,
or the promotion gate.

The CxG feature pipeline now exports additional pre-shot context directly from
StatsBomb raw/normalized events so diagnostic training can resolve them as
source-available features:

- `play_pattern`, `set_piece_category`, and `set_piece_phase` from shot event
  context.
- `score_diff_at_shot`, `is_leading`, `is_trailing`, `is_drawing`,
  `score_state`, and `simple_state` from cumulative score before the shot.
- `possession_sequence_length`, `possession_duration`,
  `previous_action_gap`, `time_gap_seconds`, and `possession_match` from
  same-possession events up to the shot.
- `pressure_state`, `pressure_proxy_score`, `recent_def_actions_count`, and
  `def_label` from shot pressure flags and preceding opponent defensive events.

These fields are all derived before the shot outcome is applied. Provider xG,
shot outcome, blocked status, and model output columns remain reference-only or
excluded from diagnostic model matrices. Validation still decides whether any
diagnostic model is promotable.

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
- `reports/candidate_calibration_summary.csv`: fold-level goal rate versus mean predicted probability by candidate.
- `reports/candidate_probability_summary.csv`: candidate probability range, mean, spread, and null counts.
- `reports/training_summary.json`: compact machine-readable training summary.
- `reports/training_report.md`: training explanation and validation handoff notes.

Issue #56 should perform full validation, calibration, and slice checks on this
selected model. Issue #57 should handle final promoted predictions/results.
