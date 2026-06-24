# CxT Model Evaluation Report

Generated: 2026-03-06 12:28:06

## Executive Summary

The CxT (Contextual xT) model predicts expected threat value for ball progression actions
(passes, carries, dribbles), adjusted for opponent defensive quality and game context.

### Key Results

| Metric | Value |
|--------|-------|
| Completion AUC | 0.8891 |
| Completion Brier | 0.149276 |
| xT Gain R² | 0.620 |
| xT Gain MAE | 0.0047 |
| CxT-Actual Correlation | 0.423 |
| Calibration ECE | 0.2148 |

## 1. Model Architecture

CxT uses a two-stage model:

1. **Completion Model** (Logistic Regression): Predicts P(action completes)
2. **xT Gain Model** (Ridge Regression): Predicts E[xT_delta | completion]

**Final CxT** = P(completion) × E[xT_gain | completion]

### Features Used

- **Numeric**: start_xt, minute_normalized, opponent ratings
- **Binary**: under_pressure, is_progressive, zone flags, action types
- **Categorical**: action_type, start_third, macro_zone_start

### Leakage Control

`xt_delta` is excluded from completion-model inputs. It is used only as the regression target for the value-gain model on completed actions. This control is documented in `docs/modeling/cxt/leakage_controls.md`.

## 2. Discrimination Performance

### 2.1 Completion Model

| Metric | Value |
|--------|-------|
| AUC | 0.8891 |
| Brier Score | 0.149276 |
| Log Loss | 0.425250 |
| Sample Size | 436,050 |
| Positive Rate | 90.6% |

**Note**: The completion model should be reviewed using the saved model configuration to confirm that all post-action fields are excluded. The current guardrail explicitly removes `xt_delta` from completion features.

### 2.2 xT Gain Model

| Metric | Value |
|--------|-------|
| R² | 0.620 |
| MAE | 0.0047 |
| RMSE | 0.0109 |
| Correlation | 0.787 |
| Sample Size | 395,270 |

### 2.3 Combined CxT

| Metric | Value |
|--------|-------|
| CxT-Actual Correlation | 0.423 |
| Mean CxT | 0.0012 |
| Std CxT | 0.0083 |
| Mean Actual | 0.0027 |

## 3. Calibration

Expected Calibration Error (ECE): **0.2148**

See calibration plot in outputs/analysis/cxt/evaluation/

## 4. Feature Importance

### 4.1 Completion Model - Top Features

| Feature | Importance |
|---------|------------|
| action_type | 0.1040 |
| is_carry | 0.0590 |
| is_progressive | 0.0483 |
| is_pass | 0.0451 |
| opponent_zone_block_rate | 0.0322 |
| start_third | 0.0285 |
| start_is_central | 0.0182 |
| is_into_penalty_area | 0.0180 |
| is_into_final_third | 0.0137 |
| is_first_half | 0.0095 |

### 4.2 xT Gain Model - Top Features

| Feature | Importance |
|---------|------------|
| is_into_penalty_area | 0.6557 |
| start_xt | 0.4591 |
| moved_to_att_third | 0.4542 |
| is_progressive | 0.4042 |
| is_into_final_third | 0.3011 |
| start_third | 0.0183 |
| zone_changed | 0.0107 |
| moved_wide_to_central | 0.0059 |
| start_is_central | 0.0024 |
| opponent_zone_block_rate | 0.0011 |

## 5. Aggregations

### 5.1 Top Teams by Total CxT

| Team ID | Actions | Total CxT | Mean CxT | vs Expected |
|---------|---------|-----------|----------|-------------|
| 16.0 | 24,167.0 | 35.19 | 0.0015 | +35.30 |
| 21.0 | 29,092.0 | 33.74 | 0.0012 | +37.71 |
| 23.0 | 31,464.0 | 29.68 | 0.0009 | +45.80 |
| 29.0 | 20,273.0 | 28.05 | 0.0014 | +23.41 |
| 22.0 | 22,641.0 | 26.85 | 0.0012 | +29.12 |
| 15.0 | 20,859.0 | 26.49 | 0.0013 | +26.38 |
| 7.0 | 11,502.0 | 22.43 | 0.0020 | +18.55 |
| 3.0 | 17,054.0 | 21.55 | 0.0013 | +21.83 |
| 26.0 | 18,620.0 | 20.53 | 0.0011 | +39.47 |
| 11.0 | 15,113.0 | 18.98 | 0.0013 | +22.91 |

### 5.2 Top Players by Total CxT

| Player ID | Actions | Total CxT | Mean CxT | Completion % |
|-----------|---------|-----------|----------|--------------|
| 236.0 | 1,673.0 | 9.96 | 0.0060 | 91.1% |
| 230.0 | 950.0 | 5.05 | 0.0053 | 87.7% |
| 83.0 | 993.0 | 4.67 | 0.0047 | 90.9% |
| 361.0 | 1,212.0 | 4.34 | 0.0036 | 83.0% |
| 589.0 | 499.0 | 4.01 | 0.0080 | 91.8% |
| 343.0 | 981.0 | 3.77 | 0.0038 | 90.5% |
| 517.0 | 1,064.0 | 3.72 | 0.0035 | 92.5% |
| 516.0 | 1,697.0 | 3.55 | 0.0021 | 87.5% |
| 376.0 | 919.0 | 3.41 | 0.0037 | 93.0% |
| 41.0 | 1,262.0 | 3.34 | 0.0027 | 92.1% |
| 459.0 | 716.0 | 3.19 | 0.0045 | 94.1% |
| 357.0 | 2,930.0 | 3.15 | 0.0011 | 91.5% |
| 67.0 | 1,196.0 | 3.04 | 0.0025 | 91.6% |
| 210.0 | 1,502.0 | 2.92 | 0.0019 | 93.1% |
| 385.0 | 1,211.0 | 2.91 | 0.0024 | 90.0% |

## 6. Conclusions

### Strengths

- xT Gain R² of 0.62 shows meaningful predictive power.
- Opponent context features improve predictions.
- Model handles different action types appropriately.
- The completion feature set excludes `xt_delta` as a leakage control.

### Limitations

- Limited to StatsBomb Open Data coverage.
- No available match score data for richer game-state features.
- CxT should still receive final fixture-backed tests that assert leakage-sensitive columns are absent from saved model configuration.

### Recommendations

1. Add tests that assert `xt_delta`, post-action outcome fields, and target columns are absent from completion features.
2. Consider separate models for each action type.
3. Add more granular opponent zone defensive metrics.
4. Regenerate this report after final CxT completion and model-card work.

## Appendix: Model Artifacts

- Model: `outputs/modeling/cxt/latest/`
- Features: `feature_store/cxt/progressions_featured.parquet`
- Slice analysis: `outputs/analysis/cxt/slice_evaluation/`
- This report: `data/reports/cxt/evaluation_report.md`
