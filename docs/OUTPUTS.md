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

Modelling outputs use the standard layout:

```text
outputs/modeling/{metric}/{model_version}/{artifact_type}/...
```

For CxG, `baseline` is the existing contextual CxG training layer and
`diagnostic_v1` is the diagnostic-informed training layer added for issue #55.

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

### Diagnostic-Informed CxG Training

Command:

```bash
make run-cxg-diagnostic-training
```

This layer trains candidate CxG models from a diagnostic-informed feature
contract while keeping the baseline CxG modelling path intact. It uses the same
shot-feature input discovery as the baseline runner and writes separate training
artifacts for validation issue #56.

The upstream CxG feature store now includes source-derived, pre-shot context
features such as play pattern, set-piece category/phase, score state before the
shot, possession timing up to the shot, and pressure/defensive-trigger proxies.
These fields give diagnostic training richer signal without relying on provider
xG, post-shot outcomes, prediction outputs, or synthetic modelling defaults.

Generated files:

```text
outputs/modeling/cxg/diagnostic_v1/contracts/feature_contract.json
outputs/modeling/cxg/diagnostic_v1/diagnostics/excluded_columns.csv
outputs/modeling/cxg/diagnostic_v1/diagnostics/resolved_features.json
outputs/modeling/cxg/diagnostic_v1/diagnostics/feature_group_summary.csv
outputs/modeling/cxg/diagnostic_v1/models/selected_model.joblib
outputs/modeling/cxg/diagnostic_v1/models/selected_model_metadata.json
outputs/modeling/cxg/diagnostic_v1/models/model_candidates.json
outputs/modeling/cxg/diagnostic_v1/predictions/cross_validated_predictions.parquet
outputs/modeling/cxg/diagnostic_v1/reports/training_report.md
outputs/modeling/cxg/diagnostic_v1/reports/model_comparison.csv
outputs/modeling/cxg/diagnostic_v1/reports/fold_metrics.csv
outputs/modeling/cxg/diagnostic_v1/reports/candidate_calibration_summary.csv
outputs/modeling/cxg/diagnostic_v1/reports/candidate_probability_summary.csv
outputs/modeling/cxg/diagnostic_v1/reports/training_summary.json
```

See `docs/modeling/cxg_diagnostic_training.md` for interpretation guidance.

### Diagnostic CxG Validation

Command:

```bash
make validate-cxg-diagnostic
```

This layer compares the selected diagnostic-informed CxG model against the
baseline CxG model. It evaluates probability quality, calibration, fold
stability, slice-level calibration, and feature governance before any promoted
prediction/reporting work in issue #57.

Generated files:

```text
outputs/validation/cxg/diagnostic_v1/validation_summary.json
outputs/validation/cxg/diagnostic_v1/model_comparison_validation.csv
outputs/validation/cxg/diagnostic_v1/fold_stability.csv
outputs/validation/cxg/diagnostic_v1/calibration_bins.csv
outputs/validation/cxg/diagnostic_v1/slice_calibration.csv
outputs/validation/cxg/diagnostic_v1/promotion_recommendation.json
outputs/validation/cxg/diagnostic_v1/validation_report.md
outputs/validation/cxg/diagnostic_v1/plots/calibration_curve.png
outputs/validation/cxg/diagnostic_v1/plots/predicted_vs_actual_by_slice.png
```

See `docs/modeling/cxg_diagnostic_validation.md` for validation and promotion
logic.

### Diagnostic CxG Results

Command:

```bash
make generate-cxg-diagnostic-results
```

This layer consumes the selected diagnostic model and the validation promotion
recommendation. It does not retrain or revalidate the model. Promoted result
files are generated only when validation recommends `promote` or
`provisional_promote`; rejected models write a blocked promotion summary unless
the command is run explicitly with `--allow-non-promoted` for exploratory
outputs.

Generated files:

```text
outputs/results/cxg/diagnostic_v1/shot_predictions.parquet
outputs/results/cxg/diagnostic_v1/player_cxg_summary.csv
outputs/results/cxg/diagnostic_v1/team_cxg_summary.csv
outputs/results/cxg/diagnostic_v1/model_promotion_summary.json
outputs/results/cxg/diagnostic_v1/prediction_quality_checks.csv
outputs/results/cxg/diagnostic_v1/cxg_results_report.md
outputs/results/cxg/diagnostic_v1/player_cxg_summary.parquet
outputs/results/cxg/diagnostic_v1/team_cxg_summary.parquet
outputs/results/cxg/diagnostic_v1/top_players_by_cxg.csv
outputs/results/cxg/diagnostic_v1/team_cxg_rankings.csv
outputs/results/cxg/diagnostic_v1/baseline_vs_diagnostic_summary.csv
```

See `docs/modeling/cxg_diagnostic_results.md` for promotion and result-output
guidance.

### Promoted Diagnostic CxG Feature Impact

Command:

```bash
make analyze-cxg-feature-impact
```

This post-promotion analysis explains why the promoted diagnostic CxG model
wins without retraining it or changing validation, promotion, governance, or
result-generation behavior. It loads the selected diagnostic model, the selected
feature metadata, the CxG feature store, and promoted shot predictions, then
computes model-agnostic feature impact artifacts.

Generated files:

```text
outputs/modeling/cxg/diagnostic_v1/feature_impact/permutation_importance.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/group_perturbation_summary.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_body_part.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_technique.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_shot_type.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_play_pattern.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_set_piece_category.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_set_piece_phase.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_pressure_state.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_score_state.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_def_label.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/category_lift_minute_bucket_label.csv
outputs/modeling/cxg/diagnostic_v1/feature_impact/feature_impact_summary.json
outputs/modeling/cxg/diagnostic_v1/feature_impact/feature_impact_report.md
```

Missing optional category columns are skipped and recorded in
`feature_impact_summary.json`. `statsbomb_xg` remains reference-only and is not
included as a model-impact feature.

### Static CxG Portfolio Summary

Command:

```bash
make build-cxg-portfolio-summary
```

This reporting layer converts promoted diagnostic CxG results and feature-impact
artifacts into static portfolio outputs for GitHub or reviewer-facing Markdown.
It does not retrain models, change validation/promotion behavior, or replace the
active Streamlit dashboard. The `make dashboard` app now reads these files for
the promoted CxG portfolio overview and still degrades gracefully when they are
missing.

Generated files:

```text
outputs/portfolio/cxg/cxg_portfolio_summary.md
outputs/portfolio/cxg/cxg_model_scorecard.json
outputs/portfolio/cxg/cxg_team_rankings.csv
outputs/portfolio/cxg/cxg_player_rankings.csv
outputs/portfolio/cxg/cxg_feature_driver_summary.csv
outputs/portfolio/cxg/cxg_category_insights.csv
outputs/portfolio/cxg/charts/model_metric_comparison.png
outputs/portfolio/cxg/charts/feature_group_impact.png
outputs/portfolio/cxg/charts/top_feature_importance.png
outputs/portfolio/cxg/charts/team_cxg_ranking.png
outputs/portfolio/cxg/charts/player_cxg_ranking.png
outputs/portfolio/cxg/charts/goals_minus_cxg_teams.png
outputs/portfolio/cxg/charts/category_lift_body_part.png
outputs/portfolio/cxg/charts/category_lift_shot_type.png
outputs/portfolio/cxg/charts/category_lift_set_piece_category.png
```

### Baseline CxG Modelling

Command:

```bash
make cxg-run
```

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
outputs/modeling/cxa/baseline/models/baseline_model.joblib
outputs/modeling/cxa/baseline/models/baseline_model.json
outputs/modeling/cxa/baseline/reports/metrics.json
outputs/modeling/cxa/baseline/reports/attribution_summary.json
outputs/modeling/cxa/baseline/predictions/action_predictions.parquet
outputs/modeling/cxa/baseline/aggregates/player_cxa.parquet
outputs/modeling/cxa/baseline/aggregates/team_cxa.parquet
outputs/modeling/cxa/baseline/aggregates/sequence_cxa.parquet
```

CxA modelling outputs follow the same versioned layout as CxG:
`outputs/modeling/cxa/baseline/` contains the existing baseline layer and
`outputs/modeling/cxa/diagnostic_v1/` contains diagnostic contract/training
artifacts. Readers should prefer `baseline/` and fall back to the older loose
`outputs/modeling/cxa/...` paths only for legacy compatibility.

DB persistence:

- `action_features`: engineered CxA action features from `feature_store/cxa/action_features.parquet`.
- `model_registry`: one `cxa` model row for `cxa_baseline_20260628` in the current sample run.
- `action_predictions`: action-level CxA predictions.
- `aggregates_player`, `aggregates_team`, and `aggregates_sequence`: CxA aggregate rows.
- `evaluation_metrics`: CxA model metrics.

`make cxa-smoke` only builds a small action-feature smoke dataset and does not populate the full CxA model outputs.

### CxA Current-State Audit Layer (Pre-Diagnostic Promotion)

Command:

```bash
make audit-cxa-current-state
```

Generated files:

```text
outputs/audits/cxa/cxa_current_state_audit.md
outputs/audits/cxa/cxa_current_state_audit.json
outputs/audits/cxa/cxa_output_inventory.csv
outputs/audits/cxa/cxa_id_quality.csv
outputs/audits/cxa/cxa_feature_inventory.csv
outputs/audits/cxa/cxa_target_audit.csv
outputs/audits/cxa/cxa_prediction_audit.csv
outputs/audits/cxa/cxa_aggregate_audit.csv
outputs/audits/cxa/cxa_risk_register.csv
```

This audit layer reports current CxA output readiness and leakage/lineage risks before diagnostic CxA modeling and promotion.

### CxA Diagnostic Feature Contract

Command:

```bash
make prepare-cxa-diagnostic-contract
```

This contract-prep layer separates diagnostic CxA model input candidates from
targets, attribution/reference values, model outputs, identifiers, and leakage
risks. It does not train a model or change existing baseline CxA outputs.

Generated files:

```text
outputs/modeling/cxa/diagnostic_v1/contracts/feature_contract.json
outputs/modeling/cxa/diagnostic_v1/diagnostics/resolved_features.json
outputs/modeling/cxa/diagnostic_v1/diagnostics/excluded_columns.csv
outputs/modeling/cxa/diagnostic_v1/diagnostics/feature_group_summary.csv
outputs/modeling/cxa/diagnostic_v1/reports/feature_contract_report.md
```

`shot_created` is the primary binary diagnostic target. `created_shot_cxg` and
`cxa_value` are attribution/reference outputs, not model input features.
Identifier columns remain available for audit, joins, and aggregation but are
excluded from feature candidates.

### CxA Diagnostic Training

Command:

```bash
make run-cxa-diagnostic-training
```

This diagnostic training layer fits candidate CxA classifiers from
`outputs/modeling/cxa/diagnostic_v1/contracts/feature_contract.json` using
`shot_created` as the primary target. It enforces the contract leakage guard and
does not replace baseline CxA outputs or promote a model.

Generated files:

```text
outputs/modeling/cxa/diagnostic_v1/models/model_candidates.json
outputs/modeling/cxa/diagnostic_v1/models/selected_model.joblib
outputs/modeling/cxa/diagnostic_v1/models/selected_model_metadata.json
outputs/modeling/cxa/diagnostic_v1/predictions/cross_validated_predictions.parquet
outputs/modeling/cxa/diagnostic_v1/reports/model_comparison.csv
outputs/modeling/cxa/diagnostic_v1/reports/training_report.md
outputs/modeling/cxa/diagnostic_v1/reports/training_summary.json
```

Validation and promotion are separate follow-up steps. CxA+ and Advanced CxA
state-value enhancements remain out of scope for this diagnostic candidate
training layer.

### CxA Diagnostic Validation

Command:

```bash
make validate-cxa-diagnostic
```

This validation layer compares the selected diagnostic CxA model against the
current fair baseline CxA predictions. It evaluates probability quality,
calibration, average precision, top-k retrieval, slice stability, join quality,
and prediction quality checks before writing a promotion recommendation. It does
not generate promoted CxA result outputs.

Generated files:

```text
outputs/validation/cxa/diagnostic_v1/validation_summary.json
outputs/validation/cxa/diagnostic_v1/promotion_recommendation.json
outputs/validation/cxa/diagnostic_v1/validation_report.md
outputs/validation/cxa/diagnostic_v1/baseline_vs_diagnostic_metrics.csv
outputs/validation/cxa/diagnostic_v1/calibration_summary.csv
outputs/validation/cxa/diagnostic_v1/threshold_summary.csv
outputs/validation/cxa/diagnostic_v1/slice_summary.csv
outputs/validation/cxa/diagnostic_v1/validation_quality_checks.csv
```

If validation recommends promotion or provisional promotion, a later results PR
should generate governed promoted CxA outputs. CxA+ and Advanced CxA remain out
of scope for this validation layer.

### Diagnostic CxA Results

Command:

```bash
make generate-cxa-diagnostic-results
```

This results layer scores the full CxA action feature table with the selected
diagnostic model and writes governed action/player/team/sequence outputs. It
does not retrain or revalidate the model. If validation returns
`provisional_promote`, outputs are labelled `provisionally_promoted` because the
current baseline comparison is reference-only/in-sample rather than a strict
OOF/holdout comparator.

Generated files:

```text
outputs/results/cxa/diagnostic_v1/action_predictions.parquet
outputs/results/cxa/diagnostic_v1/player_cxa_summary.csv
outputs/results/cxa/diagnostic_v1/player_cxa_summary.parquet
outputs/results/cxa/diagnostic_v1/team_cxa_summary.csv
outputs/results/cxa/diagnostic_v1/team_cxa_summary.parquet
outputs/results/cxa/diagnostic_v1/sequence_cxa_summary.csv
outputs/results/cxa/diagnostic_v1/sequence_cxa_summary.parquet
outputs/results/cxa/diagnostic_v1/top_players_by_cxa.csv
outputs/results/cxa/diagnostic_v1/team_cxa_rankings.csv
outputs/results/cxa/diagnostic_v1/baseline_vs_diagnostic_summary.csv
outputs/results/cxa/diagnostic_v1/model_promotion_summary.json
outputs/results/cxa/diagnostic_v1/prediction_quality_checks.csv
outputs/results/cxa/diagnostic_v1/cxa_results_report.md
```

`diagnostic_cxa` is the model-estimated probability that an action creates a
shot. `created_shot_cxg` and `created_shot_id` may appear only as clearly named
reference columns and are not model features. CxA+ and Advanced CxA value
attribution remain later work.

### Promoted Diagnostic CxA Feature Impact

Command:

```bash
make analyze-cxa-feature-impact
```

This reporting layer explains the provisionally promoted diagnostic CxA model
without retraining it, changing validation, or changing governed result
generation. It reads the selected diagnostic model, governed feature contract,
action feature store, and promoted diagnostic action predictions, then computes
lightweight model-agnostic feature impact artifacts.

Generated files:

```text
outputs/modeling/cxa/diagnostic_v1/feature_impact/feature_impact_summary.csv
outputs/modeling/cxa/diagnostic_v1/feature_impact/feature_group_impact.csv
outputs/modeling/cxa/diagnostic_v1/feature_impact/top_feature_examples.csv
outputs/modeling/cxa/diagnostic_v1/feature_impact/feature_impact_report.md
outputs/modeling/cxa/diagnostic_v1/feature_impact/feature_impact_summary.json
```

`created_shot_cxg`, `cxa_value`, identifiers, prediction outputs,
requires-review columns, and excluded-unknown columns remain outside the model
feature set. This layer explains the current diagnostic CxA probability model
only; CxA+ and Advanced CxA remain later work.

## CxT Outputs

### Pre-model Ball Progression / CxT Analysis

Command:

```bash
make analysis-cxt
```

This diagnostic layer reads pre-model progression/action features before threat
value construction. It does not read `action_threat_predictions`, `model_registry`,
post-model CxT aggregates, leaderboards, player/team threat reports, or dashboard
storytelling outputs.

Generated files:

```text
outputs/analysis/cxt/
outputs/analysis/cxt/00_action_coverage/tables/action_type_coverage.csv
outputs/analysis/cxt/00_action_coverage/plots/action_type_coverage.png
outputs/analysis/cxt/00_action_coverage/tables/id_coverage.csv
outputs/analysis/cxt/00_action_coverage/tables/location_coverage.csv
outputs/analysis/cxt/01_spatial_coverage/tables/start_zone_coverage.csv
outputs/analysis/cxt/01_spatial_coverage/tables/end_zone_coverage.csv
outputs/analysis/cxt/01_spatial_coverage/tables/transition_coverage.csv
outputs/analysis/cxt/01_spatial_coverage/plots/start_zone_coverage.png
outputs/analysis/cxt/01_spatial_coverage/plots/end_zone_coverage.png
outputs/analysis/cxt/01_spatial_coverage/plots/transition_coverage.png
outputs/analysis/cxt/02_feature_distributions/tables/numeric_feature_profiles.csv
outputs/analysis/cxt/02_feature_distributions/tables/categorical_feature_profiles.csv
outputs/analysis/cxt/03_feature_target_relationships/tables/missing_target_proxy.csv
outputs/analysis/cxt/03_feature_target_relationships/tables/action_type_progression_summary.csv
outputs/analysis/cxt/03_feature_target_relationships/plots/action_type_progression_summary.png
outputs/analysis/cxt/03_feature_target_relationships/tables/zone_progression_summary.csv
outputs/analysis/cxt/03_feature_target_relationships/plots/zone_progression_summary.png
outputs/analysis/cxt/03_feature_target_relationships/tables/final_third_box_entry_summary.csv
outputs/analysis/cxt/03_feature_target_relationships/plots/final_third_box_entry_summary.png
outputs/analysis/cxt/04_feature_correlations/tables/numeric_correlations.csv
outputs/analysis/cxt/04_feature_correlations/tables/high_correlations.csv
outputs/analysis/cxt/04_feature_correlations/tables/targeted_redundancy_checks.csv
outputs/analysis/cxt/04_feature_correlations/plots/correlation_heatmap.png
outputs/analysis/cxt/05_transition_stability/tables/transition_stability.csv
outputs/analysis/cxt/05_transition_stability/plots/transition_stability.png
outputs/analysis/cxt/05_transition_stability/tables/sparse_transitions.csv
outputs/analysis/cxt/05_transition_stability/tables/zone_resolution_recommendations.csv
outputs/analysis/cxt/06_slice_stability/tables/slice_stability.csv
outputs/analysis/cxt/06_slice_stability/plots/slice_stability.png
outputs/analysis/cxt/07_data_quality/tables/feature_quality.csv
outputs/analysis/cxt/07_data_quality/tables/football_value_checks.csv
outputs/analysis/cxt/07_data_quality/tables/cleaning_recommendations.csv
outputs/analysis/cxt/08_leakage_checks/tables/leakage_checks.csv
outputs/analysis/cxt/08_leakage_checks/tables/feature_training_eligibility.csv
outputs/analysis/cxt/report.md
```

See `docs/analysis/cxt_pre_model_analysis.md` for interpretation guidance.

### CxT Modelling

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
| `outputs/modeling/cxg/baseline/models/contextual_model.joblib` and `.json` | `model_registry` |
| `outputs/modeling/cxa/baseline/models/baseline_model.joblib` and `.json` | `model_registry` |
| `outputs/modeling/cxt/threat_grid.parquet` | `model_registry` artifact path for baseline CxT |
| `outputs/modeling/cxg/baseline/predictions/shot_predictions.parquet` | `shot_predictions` |
| `outputs/modeling/cxa/baseline/predictions/action_predictions.parquet` | `action_predictions` |
| `outputs/modeling/cxt/predictions/action_threat.parquet` | `action_threat_predictions` |
| `outputs/modeling/cxg/baseline/aggregates/player_cxg.parquet` | `aggregates_player` |
| `outputs/modeling/cxg/baseline/aggregates/team_cxg.parquet` | `aggregates_team` |
| `outputs/modeling/cxa/baseline/aggregates/player_cxa.parquet` | `aggregates_player` |
| `outputs/modeling/cxa/baseline/aggregates/team_cxa.parquet` | `aggregates_team` |
| `outputs/modeling/cxa/baseline/aggregates/sequence_cxa.parquet` | `aggregates_sequence` |
| `outputs/modeling/cxt/aggregates/player_cxt.parquet` | `aggregates_player` |
| `outputs/modeling/cxt/aggregates/team_cxt.parquet` | `aggregates_team` |
| `outputs/modeling/cxt/aggregates/sequence_cxt.parquet` | `aggregates_sequence` |
| `outputs/modeling/*/reports/metrics.json` | `evaluation_metrics` |

The persistence layer is idempotent by model family/version. Re-running a model path replaces that model/version's generated DB rows without deleting other metric families.

## Dashboard Consumption

The Streamlit dashboard v1 reads generated outputs through `configs/dashboard/v1_dashboard_contract.json`.
The active entry point is `app/streamlit_app.py`; `dashboard/` is retained as a
legacy/experimental path. The promoted CxG portfolio tab consumes the static
portfolio pack under `outputs/portfolio/cxg/`.

```bash
make dashboard
```

The dashboard is a demo/portfolio shell, not a production deployment. It starts even when generated outputs are missing and shows availability status plus empty tables for missing files. Regenerate CxG, CxA, CxT, and the CxG portfolio summary outputs locally to populate the analysis pages.

## What Should Be Committed

Commit source code, documentation, tests, migrations, and curated configs such as:

- `configs/**/*.json`
- `tests/fixtures/**`
- `docs/**/*.md`

Do not commit generated model artifacts, local SQLite DBs, feature stores, or regenerated CSV/parquet outputs unless a future PR explicitly defines a small curated example fixture.
