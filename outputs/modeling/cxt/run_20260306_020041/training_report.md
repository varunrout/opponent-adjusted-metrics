# CxT Model Training Report

Generated: 2026-03-06 02:00:45

## Model Overview

The CxT (Contextual Expected Threat) model predicts the expected threat value
of ball progressions (passes, carries, dribbles) adjusted for opponent strength
and game context.

**Model Architecture:**
- Completion Model: Logistic Regression (predicts P(success))
- xT Gain Model: Ridge Regression (predicts E[xT_delta | success])
- CxT = P(success) × E[xT_delta | success]

## Cross-Validation Metrics

### Completion Model

| Metric | Mean | Std |
|--------|------|-----|
| AUC | 1.000 | 0.000 |
| Brier | 0.0000 | 0.0000 |
| Log Loss | 0.0010 | 0.0001 |

### xT Gain Model

| Metric | Mean | Std |
|--------|------|-----|
| R² | 0.620 | 0.002 |
| MAE | 0.0047 | 0.0000 |
| RMSE | 0.0109 | 0.0001 |

## Final Evaluation

| Metric | Value |
|--------|-------|
| Completion AUC | 1.000 |
| Completion Brier | 0.0000 |
| xT Gain R² | 0.620 |
| xT Gain MAE | 0.0047 |
| CxT-Actual Correlation | 0.787 |

## Feature Summary

- Numeric features: 7
- Binary features: 20
- Categorical features: 3

### Numeric Features

- start_xt
- xt_delta
- minute_normalized
- opponent_global_rating
- opponent_zone_rating
- opponent_global_block_rate
- opponent_zone_block_rate

### Binary Features

- under_pressure
- is_late_game
- is_first_half
- is_second_half
- is_extra_time
- is_very_late
- is_early_game
- start_is_central
- is_progressive
- is_into_final_third
- ... and 10 more

### Categorical Features

- action_type
- start_third
- macro_zone_start

## Interpretation

- **CxT > 0**: Action expected to add threat (positive progression)
- **CxT < 0**: Action expected to reduce threat (negative progression)
- Higher CxT indicates more dangerous progressions considering context
