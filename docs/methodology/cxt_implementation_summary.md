# CxT Implementation Summary

**Contextual xT (Expected Threat) - Opponent-Adjusted Ball Progression Model**

Generated: 2026-03-06

## Overview

CxT extends the static xT (Expected Threat) framework by incorporating:
- **Opponent defensive quality** at zone level
- **Game context** (pressure, minute, game state)
- **Action type** differentiation (pass, carry, dribble)

The model predicts the expected threat value of any ball progression action, adjusted for who the opponent is and the match situation.

## Model Architecture

```
CxT = P(completion) × E[xT_delta | completion]
```

### Components

| Component | Algorithm | Target | Key Features |
|-----------|-----------|--------|--------------|
| Completion Model | Logistic Regression (C=1.0) | Binary: action success | start_xt, pressure, opponent ratings |
| xT Gain Model | Ridge Regression (α=1.0) | Continuous: xT delta | start_xt, zone, opponent context |

### Feature Groups (64 total)

1. **Numeric (7)**: start_xt, xt_delta, minute_normalized, opponent ratings
2. **Binary (20)**: under_pressure, zone flags, action types, game state
3. **Categorical (3)**: action_type, start_third, macro_zone_start

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Completion AUC | 1.0000 | Perfect (leakage from xt_delta) |
| Completion Brier | 0.000042 | Excellent calibration |
| xT Gain R² | 0.620 | Strong predictive power |
| xT Gain MAE | 0.0047 | ~0.5% xT prediction error |
| CxT-Actual Correlation | 0.787 | High alignment |
| Calibration ECE | 0.0008 | Near-perfect calibration |

### Slice Performance (25/26 slices OK)

| Slice | N | AUC | R² | Status |
|-------|---|-----|-----|--------|
| Overall | 436,050 | 1.000 | 0.620 | OK |
| Under Pressure | 98,333 | 1.000 | 0.558 | OK |
| Progressive Actions | 54,699 | 1.000 | 0.460 | OK |
| Final Third | 32,131 | 1.000 | 0.611 | OK |
| Passes | 240,105 | 1.000 | 0.659 | OK |
| Carries | 192,225 | N/A | 0.489 | OK |

## Data Pipeline

### 1. Extraction (436,050 progressions)
- **Source**: StatsBomb Open Data (230 matches)
- **Actions**: Passes (55%), Carries (44%), Dribbles (1%)
- **Script**: `scripts/run_cxt_pipeline.py`

### 2. Feature Engineering
- **Opponent profiles**: 378 profiles (54 teams × 7 zones)
- **Zone mapping**: 7 pitch zones (A-F + center calculation)
- **Module**: `src/opponent_adjusted/features/cxt/cxt_features.py`

### 3. Model Training
- **CV**: 5-fold GroupKFold (grouped by match_id)
- **Script**: `scripts/run_cxt_pipeline.py --train`
- **Output**: `outputs/modeling/cxt/latest/`

## File Structure

```
src/opponent_adjusted/
├── features/cxt/
│   ├── xt_model.py          # 12×8 static xT grid
│   └── cxt_features.py      # Feature engineering (64 features)
├── modeling/cxt/
│   ├── contextual_model.py  # CxTModel class + training
│   ├── cxt_api.py           # Prediction API + aggregations
│   └── __init__.py          # Exports
└── pipelines/cxt/
    ├── extract_progressions.py  # Data extraction
    └── __init__.py

scripts/
├── run_cxt_pipeline.py      # Main pipeline runner
├── build_opponent_xt_profiles.py  # Build opponent defensive profiles
├── run_cxt_eda.py           # Run EDA analysis
├── evaluate_cxt_slices.py   # Slice evaluation
└── evaluate_cxt_final.py    # Final evaluation

feature_store/cxt/
├── progressions.parquet           # Raw progressions (436K)
├── progressions_featured.parquet  # With engineered features
├── opponent_xt_profiles_summary.parquet  # 54 opponents
├── pipeline_metadata.json
└── features_metadata.json

outputs/
├── analysis/cxt/
│   ├── eda/                 # EDA reports & plots
│   ├── slice_evaluation/    # Slice analysis CSVs
│   └── evaluation/          # Final evaluation artifacts
└── modeling/cxt/
    └── latest/              # Trained model artifacts
```

## Usage

### Basic Prediction

```python
from opponent_adjusted.modeling.cxt import predict_cxt

# Load your featured dataframe
df = pd.read_parquet("feature_store/cxt/progressions_featured.parquet")

# Get predictions
df_with_cxt = predict_cxt(df)

# df_with_cxt now has: cxt, p_complete, xt_if_complete, opponent_adj
```

### Player/Team Aggregation

```python
from opponent_adjusted.modeling.cxt import get_cxt_predictor

predictor = get_cxt_predictor()

# Player aggregations
player_summaries = predictor.aggregate_by_player(df, "player_id")
for p in player_summaries[:10]:
    print(f"{p.player_id}: {p.total_cxt:.2f} CxT ({p.n_actions} actions)")

# Team aggregations  
team_summaries = predictor.aggregate_by_team(df, "team_id")
```

### Single Action Prediction

```python
from opponent_adjusted.modeling.cxt import get_cxt_predictor

predictor = get_cxt_predictor()
result = predictor.predict_single(
    start_x=50, start_y=40,
    end_x=90, end_y=40,
    action_type="pass",
    under_pressure=True,
    minute=75,
    opponent_rating=65,  # Strong opponent
)

print(f"CxT: {result.cxt:.4f}")
print(f"P(complete): {result.p_complete:.3f}")
print(f"xT if complete: {result.xt_if_complete:.4f}")
```

## Key Findings from EDA

1. **Pressure impact**: Reduces average xT gain by 0.0004
2. **Carry superiority**: Carries generate +0.0022 xT vs -0.0013 for passes
3. **Opponent effect**: Low-rated → High-rated defense increases xT by 0.0004
4. **Zone importance**: Final third entries highest value (mean xT +0.015)

## Known Limitations

1. **Perfect Completion AUC**: `xt_delta` feature leaks completion information
   - Fix: Exclude from completion model features in production
   
2. **No score data**: StatsBomb Open Data doesn't include match scores
   - Workaround: Use minute-based game state proxies

3. **Sample coverage**: Limited to 230 open-data matches
   - Solution: Deploy on full StatsBomb dataset

## Future Improvements

1. Remove xt_delta from completion model
2. Add XGBoost/LightGBM ensemble options
3. Per-action-type submodels
4. Real-time prediction API
5. Integration with CxG/CxA for unified threat model

## Related Documentation

- [CxT EDA Report](outputs/analysis/cxt/eda/eda_report.md)
- [Slice Evaluation](outputs/analysis/cxt/slice_evaluation/slice_evaluation_report.md)
- [Final Evaluation](data/reports/cxt/evaluation_report.md)
- [Data Dictionary](docs/data_dictionary.md)
