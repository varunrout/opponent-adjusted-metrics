# CxG / CxG+ Split Policy And Parallel Modelling Plan

Date: 2026-08-21
Run context: cxg-analysis-20260820T201934Z

> **STATUS NOTE (added 2026-08-21, later same day):** the bivariate/multivariate/model-spec/freeze work this plan describes past Step 4 was built, then deliberately reverted back to univariate-only (`cleanup_to_univariate_state.py`) — see Drive `00_PROJECT_STATE`, "CXG REVERTED TO UNIVARIATE RESULT STATE." Current live analysis state is univariate-only plus the split infrastructure (Steps 1-4 tables), extended since to cover the new `opponent_adjusted` family (ODI, defensive profile clusters) at the same univariate depth. This document's split-table design (Steps 1-3) and the general split-discipline principle (train-only feature confirmation) remain in force and are still being followed for new work. The specific sequential plan below (Steps 5-11: bivariate, baselines, parallel candidate training, freeze) does not reflect current state and should be treated as a future intent, not a status report, until explicitly resumed.

## Decision

The full-dataset analysis completed so far is valid as exploratory analysis and feature understanding, but it is not sufficient as the final feature-promotion protocol for modelling. Before production model selection, CxG and CxG+ must use canonical train/validation/test splits by match_id.

## Why

- Full-dataset EDA, null reports, summary statistics and chart QA are acceptable because they describe the corpus.
- Feature promotion, interaction confirmation and model metrics must be split-aware to avoid overfitting feature choices to the full dataset.
- The existing GroupKFold baseline reduced leakage during offline evaluation, but it is not the same as a locked train/validation/test protocol.
- Test data must remain untouched until final unbiased model reporting.

## Model Tracks

### CxG

Scope: event-wide model over all shots.

Population:
- All shots in `oam_features.cxg_training_matrix_v1`.
- Current known row count: 15,737 shots.

Initial baseline:
- `shot_x_sb`
- `shot_y_sb`

Candidate event-context layer:
- Train-only feature confirmation will decide whether event_context shortlist features enter the event-wide model.

### CxG+

Scope: 360-enhanced model over shots with StatsBomb 360 frames.

Population:
- `has_360_frame = TRUE` rows in `oam_features.cxg_training_matrix_v1`.
- Current known row count: 3,960 shots.

Initial baseline:
- `shot_x_sb`
- `shot_y_sb`

Candidate 360 core:
- `visible_goal_angle_proxy`
- `goal_mouth_defender_count`
- `visible_goal_angle_delta`
- `estimated_goalface_occlusion`
- `defensive_line_depth`
- `gk_distance_to_shooter`
- `defenders_between_ball_and_goal`

Candidate expansion:
- CxG+ shortlist/review features from line_shape_360, goalkeeper_360 and defensive_360.
- Event_context may be tested as calibration/context layer, not assumed as core ranking signal.

## Required Split Design

Canonical split table:
- BigQuery table: `oam_analysis.cxg_match_splits_v1`
- Key: `match_id`
- Fields: `split`, `split_seed`, `has_360_match`, event shot count, event goal count, 360 shot count, 360 goal count.

Recommended split proportions:
- Train: 70%
- Validation: 15%
- Test: 15%

Split constraints:
- Split by `match_id`, never by shot/event row.
- Preserve goal-rate balance as much as possible.
- Preserve 360 coverage balance as much as possible.
- Ensure validation and test contain enough 360 shots/goals for CxG+ evaluation.
- Keep test sealed after creation.

## Sequential Plan

1. Build canonical match-level split table.
2. Validate split balance for both CxG and CxG+.
3. Create split-aware modelling surfaces:
   - `oam_analysis.cxg_event_model_matrix_v1`
   - `oam_analysis.cxg_plus_360_model_matrix_v1`
4. Re-run train-only feature confirmation:
   - univariate signal on train only
   - bivariate interaction on train only
   - redundancy/correlation on train only
5. Validate selected features on validation split:
   - direction stability
   - support stability
   - pair-interaction stability
   - uplift over XY baseline
6. Establish clean baselines:
   - CxG event-wide XY baseline
   - CxG+ 360-cohort XY baseline
7. Train candidate models in parallel:
   - CxG event-wide candidate(s)
   - CxG+ 360 candidate(s)
8. Compare validation metrics and calibration.
9. Freeze final feature set and hyperparameters.
10. Run test once for final model report.
11. Only after test report: move toward model artifact, scoring table and dashboard integration.

## Current Status

The previous selected features remain provisional. They are strong exploratory candidates, but final promotion requires train-only selection and validation/test confirmation.
