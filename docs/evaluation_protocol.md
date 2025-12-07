# Evaluation Protocol

Comprehensive evaluation methodology for opponent-adjusted metrics models.

## Overview

This protocol ensures rigorous, reproducible evaluation of model performance across multiple dimensions:
- **Calibration**: How well predicted probabilities match observed frequencies
- **Discrimination**: Ability to separate positive and negative cases
- **Slice Performance**: Consistency across subgroups
- **Neutralization Validity**: Proper isolation of inherent vs. contextual effects

---

## Metrics

### Primary Metrics

#### 1. Brier Score
Measures mean squared error of probabilistic predictions.

```
Brier = (1/N) Σ (p_i - y_i)²
```

- **Range**: [0, 1] (lower is better)
- **Interpretation**: Average squared deviation from true outcome
- **Acceptance**: ≤ baseline_xg - 0.002 (improvement required)

#### 2. Log Loss (Binary Cross-Entropy)
Measures probabilistic prediction quality with logarithmic penalty for confident wrong predictions.

```
LogLoss = -(1/N) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

- **Range**: [0, ∞] (lower is better)
- **Acceptance**: < baseline_xg log loss

#### 3. ROC AUC (Area Under Receiver Operating Characteristic)
Measures discrimination ability independent of threshold.

```
AUC = P(p_positive > p_negative)
```

- **Range**: [0.5, 1.0] (higher is better, 0.5 = random)
- **Acceptance**: > 0.70 (good discrimination)

#### 4. Expected Calibration Error (ECE)
Measures deviation between predicted probabilities and observed frequencies across bins.

```
ECE = Σ (n_b / N) |acc_b - conf_b|
```

Where:
- n_b: number of predictions in bin b
- acc_b: accuracy in bin b
- conf_b: average confidence in bin b

- **Range**: [0, 1] (lower is better)
- **Acceptance**: ≤ 0.06 overall, ≤ 0.05 for high-pressure slice
- **Bins**: 10 equal-frequency bins

---

### Secondary Metrics

#### 5. Precision-Recall AUC
For imbalanced classes (goals are rare ~10%).

#### 6. Reliability Diagram
Visual calibration assessment:
- X-axis: Predicted probability (binned)
- Y-axis: Observed frequency
- Perfect calibration: diagonal line

#### 7. Feature Importance
Top 20 features by:
- Gain (LightGBM built-in)
- SHAP values (sample-based, optional)

---

## Evaluation Slices

Evaluate model performance on meaningful subgroups to ensure robustness.

### Required Slices

| Slice Name | Filter | Rationale |
|------------|--------|-----------|
| **Overall** | All shots | Baseline performance |
| **Under Pressure** | `under_pressure = True` | High-pressure scenarios |
| **High Pressure Score** | `pressure_proxy_score > Q3` | Top quartile pressure |
| **Low Pressure Score** | `pressure_proxy_score < Q1` | Bottom quartile pressure |
| **Strong Defense** | `opponent_def_rating_global > Q3` | Top quartile opponent quality |
| **Weak Defense** | `opponent_def_rating_global < Q1` | Bottom quartile opponent quality |
| **Leading** | `score_diff_at_shot > 0` | Team ahead |
| **Trailing** | `score_diff_at_shot < 0` | Team behind |
| **Drawing** | `score_diff_at_shot = 0` | Tied game |
| **First Half** | `period = 1` | Early game |
| **Second Half** | `period = 2` | Late game |
| **Close Range** | `shot_distance <= 12` | Within 12 meters |
| **Long Range** | `shot_distance > 20` | Beyond 20 meters |

### Acceptance Criteria per Slice

- **Calibration (ECE)**: 
  - Overall: ≤ 0.06
  - Under pressure: ≤ 0.05
  - Other slices: ≤ 0.08

- **AUC**: 
  - All slices: > 0.65

- **Sample Size**: 
  - Report only if slice contains ≥ 100 shots

---

## Validation Strategy

### Time-Based Split
Avoid temporal leakage by splitting on match date:

```
Train: Matches before 2022-06-01
Validation: Matches 2022-06-01 to 2022-12-01
Test: Matches after 2022-12-01
```

### Alternative: Tournament-Based Split
For cross-competition generalization:

```
Train: FIFA WC 2018 + UEFA Euro 2020
Validation: FIFA WC 2022
Test: UEFA Euro 2024
```

### Cross-Validation
5-fold cross-validation on training set for hyperparameter tuning.

---

## Neutralization Validation

Assess quality of opponent-adjusted metrics.

### 1. Mean Adjustment Check
```
E[opponent_adjusted_diff] ≈ 0 ± 0.005
```

**Rationale**: Neutralization should center around zero when averaged across all opponents.

### 2. Opponent Strength Correlation
```
corr(opponent_def_rating_global, opponent_adjusted_diff) > 0
```

**Rationale**: Stronger opponents should reduce raw CxG relative to neutral.

### 3. Pressure Correlation
```
corr(pressure_proxy_score, opponent_adjusted_diff) > 0
```

**Rationale**: Higher pressure should reduce raw CxG relative to neutral.

### 4. Calibration Improvement
Under-pressure slice calibration should improve vs. non-contextual baseline.

---

## Reporting Requirements

### 1. Summary Table
```
| Metric       | Overall | Under Pressure | Strong Defense | Trailing |
|--------------|---------|----------------|----------------|----------|
| Brier        | 0.085   | 0.092          | 0.089          | 0.087    |
| Log Loss     | 0.312   | 0.335          | 0.321          | 0.318    |
| AUC          | 0.743   | 0.721          | 0.735          | 0.739    |
| ECE          | 0.042   | 0.048          | 0.045          | 0.043    |
| N            | 5432    | 1823           | 1358           | 2145     |
```

### 2. Calibration Curves
Generate reliability plots for:
- Overall
- Under pressure
- Strong defense
- By zone (A-F)

Save to: `reports/calibration/{model_version}_*.png`

### 3. Feature Importance
Top 20 features with gain values.

Save to: `reports/feature_importance/{model_version}_importance.csv`

### 4. Slice Metrics CSV
Full metrics for all slices.

Save to: `reports/slices/{model_version}_slice_metrics.csv`

### 5. Aggregates
Player and team aggregates (top 50 by summed_cxg).

Save to: `reports/aggregates/{model_version}_player_aggregates.csv`

---

## Acceptance Gates

Model can proceed to production if:

1. ✅ Brier score improvement ≥ 0.002 vs. baseline xG
2. ✅ Overall ECE ≤ 0.06
3. ✅ Under-pressure ECE ≤ 0.05
4. ✅ Mean(opponent_adjusted_diff) within ±0.005
5. ✅ AUC > 0.70 overall
6. ✅ No slice ECE > 0.08 (for slices with N ≥ 100)

If any gate fails:
- **Calibration issue**: Apply isotonic or Platt scaling
- **Poor slice performance**: Add slice-specific features or stratified sampling
- **Low AUC**: Add more discriminative features or increase model capacity
- **Neutralization issue**: Review reference context definition

---

## Comparison to Baseline

### Baseline Model
Simple logistic regression using only:
- StatsBomb xG
- Shot distance
- Shot angle

### Required Improvements
- Brier: -0.002 or better
- ECE: -0.01 or better
- AUC: +0.02 or better

---

## Monitoring in Production

### Online Metrics (if deployed)
- **Calibration drift**: Monthly ECE calculation
- **Feature drift**: Distribution shift detection
- **Performance degradation**: AUC tracking

### Triggers for Retraining
- ECE > 0.08 for two consecutive months
- AUC drops below 0.68
- New competition data available

---

## Statistical Significance Testing

Use bootstrap resampling (1000 iterations) to compute 95% confidence intervals for:
- Brier score difference vs. baseline
- AUC difference vs. baseline

Report CI alongside point estimates.

---

## Reproducibility

All evaluation must be:
1. **Deterministic**: Fixed random seeds
2. **Versioned**: Link to feature version and model version
3. **Auditable**: Store predictions in database with timestamps
4. **Documented**: Evaluation config saved with model artifact

---

## Example Evaluation Command

The project will ultimately expose a dedicated evaluation entry point (e.g., a module under `opponent_adjusted.evaluation`). Until that module is implemented, the command below should be treated as illustrative rather than copy‑paste runnable:

```bash
poetry run python -m opponent_adjusted.evaluation.cxg_evaluate \
  --model cxg_v1 \
  --features v1 \
  --output-dir reports/evaluation/cxg_v1 \
  --compare-baseline \
  --bootstrap-ci
```

Expected outputs for a conforming evaluator:
- `reports/evaluation/cxg_v1/summary.json`
- `reports/evaluation/cxg_v1/calibration_plots/`
- `reports/evaluation/cxg_v1/slice_metrics.csv`
- `reports/evaluation/cxg_v1/feature_importance.csv`

---

## References

1. Niculescu-Mizil & Caruana (2005). "Predicting good probabilities with supervised learning."
2. Guo et al. (2017). "On Calibration of Modern Neural Networks."
3. StatsBomb. "xG Evaluation Best Practices."
