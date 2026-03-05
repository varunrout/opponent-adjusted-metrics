# cXA Feature Store EDA Report

**Generated**: 2026-02-02 21:24:46

---

## Executive Summary

This report provides comprehensive Exploratory Data Analysis (EDA) for the cXA (Credit Expected Assist) feature store.

---

## Phase 0: Data Overview

### Dataset Summary

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| shots | 5,606 | 32 | All shots in the dataset |
| passes | 240,105 | 47 | All passes with features |
| possessions | 45,405 | 39 | Possession-level data |
| sequences | 4,123 | 87 | Pass sequences leading to shots (wide format) |
| action_sequences | 4,910 | 79 | Action sequences for cXA-xG (wide format) |

### Key Findings
- **87.6%** of shots have preceding action sequences (4,910/5,606)
- **73.6%** of shots have preceding pass sequences (4,123/5,606)
- Shooter ID/name columns have 100% null - needs data pipeline fix

---

## Phase 1: Passes EDA

### Key Statistics
- **Total Passes**: 240,105
- **Assist Rate**: 0.15% (extreme class imbalance)
- **Complete Passes**: 82.3%

### Feature Insights

| Feature | Mean | Description |
|---------|------|-------------|
| end_x | 64.20 | Average pass end location |
| end_y | 39.89 | Centered around midfield |
| pass_length | 20.87 | Average pass distance |
| xt_delta | -0.00 | xT gain (slightly negative on average) |

### Boolean Feature Rates
- **is_cross**: 2.3%
- **is_through_ball**: 0.35%
- **is_into_box**: 6.6%
- **is_progressive**: 26.9%
- **is_final_third**: 30.6%

### Class Imbalance (Critical)
- Assists = 0.15% of all passes
- Recommendation: Use stratified sampling, SMOTE, or class weights

---

## Phase 2: Shots EDA

### Key Statistics
- **Total Shots**: 5,606
- **Total Goals**: 507
- **Conversion Rate**: 9.0%
- **Total xG**: ~500

### xG Calibration
- xG appears well calibrated (Goals ≈ sum(xG))
- Higher xG shots have higher conversion rates

---

## Phase 3: Pass Sequences EDA

### Sequence Length Analysis
- **Total Sequences**: 4,123
- **Mean passes per sequence**: 2.34
- **Sequences with 1 pass**: 1,058 (25.7%)
- **Sequences with 2+ passes**: 3,065 (74.3%)

### Goal vs Non-Goal
- Goal sequences: mean = 2.37 passes
- Non-goal sequences: mean = 2.34 passes
- Difference not statistically significant (p=0.58)

---

## Phase 4: Action Sequences EDA

### Key Statistics
- **Total Shot Windows**: 4,910
- **Mean actions per window**: 3.33
- **Total Goals**: 439
- **Conversion Rate**: 8.9%
- **Total xG**: 452.72

### Action Type Distribution by Position

| Position | Pass % | Carry % | n_total |
|----------|--------|---------|---------|
| 1 (last) | 47.7% | 52.1% | 4,910 |
| 2 | 61.0% | 38.8% | 3,571 |
| 3 | 47.1% | 52.7% | 3,069 |
| 4 | 57.8% | 42.2% | 2,589 |
| 5 | 51.1% | 48.9% | 2,227 |

### Key Insight
- Position 1 (last action before shot) has MORE carries than passes (52% vs 48%)
- This validates our cXA-xG model including carries

### Goal vs Non-Goal
- Goal shots: mean actions = 3.45, mean xG = 0.24
- Non-goal shots: mean actions = 3.32, mean xG = 0.08
- Higher xG shots have more preceding actions

---

## Key Insights & Recommendations

### 1. Class Imbalance (CRITICAL)
- Assists = 0.15% of passes
- **Action**: Use stratified CV, SMOTE, or adjusted class weights

### 2. Feature Importance
- Location features (end_x, end_y) highly predictive
- Boolean features (is_cross, is_through_ball, is_into_box) meaningful

### 3. Carries Matter
- 52% of position-1 actions are carries (not passes)
- cXA-xG model correctly includes carries in credit allocation

### 4. Data Quality
- shooter_id/shooter_name columns are 100% null
- Some sequences have fewer shots than expected (data pipeline issue)

---

## Output Files

All EDA outputs saved in :

-  - Data alignment and schema analysis
-  - Pass feature distributions and correlations
-  - Shot analysis and xG calibration
-  - Sequence pattern analysis
-  - Action type and carry analysis
