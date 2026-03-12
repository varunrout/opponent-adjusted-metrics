# CxT Feature Store EDA Report

**Generated**: 2026-03-06 12:27:33

---

## Executive Summary

This report provides comprehensive Exploratory Data Analysis (EDA) for the CxT (Contextual Expected Threat) feature store.

---

## Phase 0: Data Overview

### Dataset Summary

- **Total Actions**: 436,050
- **Total Columns**: 35
- **Unique Matches**: 230
- **Unique Teams**: 54
- **Unique Players**: 1,612

### Action Type Distribution

| Action Type | Count | Percentage |
|-------------|-------|------------|
| pass | 240,105 | 55.1% |
| carry | 192,225 | 44.1% |
| dribble | 3,720 | 0.9% |

### xT Statistics

| Metric | Mean | Std |
|--------|------|-----|
| start_xt | 0.020722 | 0.0196 |
| end_xt | 0.026972 | 0.0356 |
| xt_delta | 0.000233 | 0.0207 |

---

## Phase 1: Progressions EDA

### Key Findings

- **Pressure Rate**: 22.6%
- **Pressure xT Effect**: -0.000400

### Insight

Actions under pressure show LOWER xT delta, confirming pressure's impact on ball progression.

---

## Phase 2: Outcomes EDA

### Key Statistics

- **Pass Completion Rate**: 83.0%
- **Final Third Entries**: 32,131 (7.37%)
- **Penalty Area Entries**: 21,586 (4.95%)

### Carry Analysis

- **Total Carries**: 192,225
- **Mean xT Delta**: 0.002154
- **Progressive Rate**: 6.9%

---

## Phase 3: Opponent Context EDA

### Key Findings

- **Pressure Tier xT Signal**: 0.000400 (Low - High)
  - ✓ Expected: Higher xT vs low-pressure opponents
- **Home xT Advantage**: 0.000100

---

## Key Insights & Recommendations

### 1. Action Type Mix

- Passes dominate but carries contribute ~44% of actions
- Carries should be included in CxT modeling (validated)

### 2. Pressure Effects

- Pressure reduces ball progression quality
- Include pressure as a contextual feature

### 3. Opponent Context

- Playing against high-pressure teams reduces xT accumulation
- Opponent defensive quality should be a key feature

---

## Output Files

All EDA outputs saved in `outputs/analysis/cxt/eda/`:

- `csv/` - Tabular summaries
- `plots/` - Visualizations
- `eda_report.md` - This report
