# xT Position-Based Analysis - Technical Implementation Report

## Objective

Refine the expected threat (xT) analysis from Karun Singh's 120×80 grid model by analyzing threat generation across **on-pitch positions** (where shots are actually taken), rather than aggregating all shots together.

This addresses the user requirement:
> "Based on that position data and formation data we need to see what kind of positions tactical vs on pitch the xT is generated from."

## Implementation Approach

### Phase 1: Position Inference (COMPLETED)

**Module**: `src/opponent_adjusted/analysis/xt_position_refined.py`

Instead of mapping baseline player IDs to StatsBomb IDs (which requires database access), we use **location-based position inference**:

```python
def infer_broad_position(x: float) -> str:
    """Classify position by pass_end_x coordinate."""
    if x < 40:
        return "Defender"
    elif x < 80:
        return "Midfielder"
    else:
        return "Forward"
```

**Rationale**: 
- Pass end location indicates where the shot-creating action occurred
- On-pitch position (where you are) matters more than assigned position for shot creation
- Avoids database dependency while capturing actual threat generation locations

### Position Categories

| Category | X Range | Description | Logic |
|----------|---------|-------------|-------|
| **Defender** | 0-40 | Deep defensive, center-back area | Rare shot creation |
| **Midfielder** | 40-80 | Midfield box-to-box area | Transition and supporting shots |
| **Forward** | 80-120 | Attacking third, inside box | Primary shot creation |

### Phase 2: Detailed Position Grid (COMPLETED)

**9-Zone Analysis**: Further subdivides each position by vertical location

```python
def infer_detailed_position(x: float, y: float) -> str:
    """Create 9-zone grid: 3 horizontal × 3 vertical."""
    # Horizontal: Defensive (0-40), Midfield (40-80), Attacking (80-120)
    # Vertical: Wing (0-27), Central (27-53), Wing (53-80)
    
    # Results in: Defending (Wing/Central/Wing), Midfield (Wing/Central/Wing), etc.
```

**Output**: 5 zones with sufficient sample size:
- Attacking (Central): 3,059 shots, 0.1117 avg xT
- Attacking (Wing): 1,129 shots, 0.0432 avg xT
- Midfield (Wing): 86 shots, 0.0457 avg xT
- Midfield (Central): 85 shots, 0.0429 avg xT
- Defensive (Central): 4 shots, 0.0203 avg xT

### Phase 3: Threat Classification (COMPLETED)

Uses existing xTModel with ±1σ thresholds:

```python
analysis["threat_level"] = analysis["avg_xt"].apply(
    xt_model.get_threat_level  # Returns Low/Medium/High
)
```

**Results**:
- All positions classified as **Low to Medium threat**
- No position reaches "High threat" threshold (>0.5603 xT)
- This indicates shots naturally occur in moderate-threat zones
- Grid theoretical maximum (0.9 xT) >> actual shot locations (0.09 xT)

### Phase 4: Analysis & Visualization (COMPLETED)

**Four Analysis Functions**:

1. **analyze_xt_by_position()**
   - Groups by Defender/Midfielder/Forward
   - Computes: count, sum, mean, std, min, max of xT
   - Output: cxa_xt_by_position.csv

2. **analyze_xt_by_detailed_position()**
   - Groups by 9-zone detailed positions
   - Same statistics as above
   - Output: cxa_xt_by_detailed_position.csv

3. **analyze_pitch_heatmap()**
   - Creates 3×3 grid pivot table
   - Rows: Wing/Central/Wing (vertical)
   - Columns: Defensive/Midfield/Attacking (horizontal)
   - Output: cxa_xt_pitch_heatmap_data.csv

4. **enrichment pipeline**
   - Loads baseline with 4,363 shots
   - Adds on_pitch_position, detailed_position columns
   - Saves: cxa_baseline_enriched.csv (788KB)

**Three Visualizations**:

1. **cxa_position_comparison.png** (93KB)
   - Side-by-side bar charts: Avg xT vs Total xT
   - Color-coded by threat level
   - Shows volume vs efficiency trade-off

2. **cxa_detailed_position_comparison.png** (116KB)
   - 9-zone detailed position breakdown
   - Sorted by avg xT
   - Reveals intra-position variation

3. **cxa_pitch_heatmap.png** (146KB)
   - Spatial heatmap across pitch
   - 3×3 grid cells with xT values
   - Red intensity = threat level

## Key Results

### Threat Generation Distribution

```
Position       | Shots |  Total xT | Avg xT | Multiple vs Forward
─────────────────────────────────────────────────────────────────
Defender       |     4 |    0.0812 | 0.0203 |     0.22× (Forward)
Midfielder     |   171 |    7.5813 | 0.0443 |     0.47× (Forward)
Forward        | 4,188 |  390.4690 | 0.0932 |     1.00× (baseline)
─────────────────────────────────────────────────────────────────
TOTAL          | 4,363 |  398.1315 | 0.0913 |
```

### Position Distribution

| Position | Percentage | Interpretation |
|----------|-----------|---|
| Forward | 96.0% | Overwhelming shot volume from attacking third |
| Midfielder | 3.9% | Midfield supporting shots, box-to-box plays |
| Defender | 0.1% | Rare defensive set-piece or counter-shot attempts |

### Detailed Breakdown

```
Position                  | Shots | Avg xT | Threat
─────────────────────────────────────────────────
Attacking (Central)       | 3,059 | 0.1117 | Medium  ← Highest quality
Attacking (Wing)          | 1,129 | 0.0432 | Medium
Midfield (Wing)           |    86 | 0.0457 | Medium
Midfield (Central)        |    85 | 0.0429 | Medium
Defensive (Central)       |     4 | 0.0203 | Low     ← Lowest quality
```

### Within-Position Variation

- **Attacking (Central) leads**: 0.1117 avg xT
  - 3,059 shots (70% of total)
  - Central attacking positions are highest quality
  - Straight-ahead shots to goal more dangerous

- **Attacking (Wing) secondary**: 0.0432 avg xT
  - 1,129 shots (26% of total)
  - Angled shots from wings lower xT than central
  - Wide position affects shot angle, geometry

- **Midfield roles**: 0.0429-0.0457 avg xT
  - Combined 171 shots (4% of total)
  - Supporting roles, half-chances
  - Central vs wing variation minimal (0.0429 vs 0.0457)

- **Defender rare**: 0.0203 avg xT
  - Only 4 shots (0.1% of total)
  - Long-range defensive attempts
  - Very low threat, likely set-pieces

## Data Pipeline

### Input
```
cxa_baselines_pass_level.csv (4,363 rows)
├─ match_id, player_id, team_id
├─ pass_end_x, pass_end_y
├─ xa_plus (xA value)
└─ other pass attributes (type, height, body_part, etc.)
```

### Processing
```
1. Load baseline CSV
2. Infer positions:
   - on_pitch_position (broad: Def/Mid/Fwd)
   - detailed_position (9-zone grid)
3. Compute xT statistics by position
4. Classify threats using xTModel
5. Generate visualizations
6. Export results
```

### Outputs
```
outputs/analysis/cxa/
├─ cxa_xt_by_position.csv                    (Position summary)
├─ cxa_xt_by_detailed_position.csv          (9-zone summary)
├─ cxa_xt_pitch_heatmap_data.csv            (3×3 grid data)
├─ cxa_baseline_enriched.csv                (Full baseline + positions)
├─ cxa_position_comparison.png              (Bar charts)
├─ cxa_detailed_position_comparison.png     (9-zone bars)
└─ cxa_pitch_heatmap.png                    (Heatmap visualization)
```

## Integration with Previous Work

### xTModel Foundation
- Uses existing `xTModel` class from `src/opponent_adjusted/features/xt_model.py`
- Leverages trained 120×80 grid with Karun Singh model
- Applies ±1σ threat thresholds (Low/Medium/High)
- Maintains consistency with prior analysis

### Baseline Data
- Ingests `cxa_baselines_pass_level.csv` (4,363 shots)
- Preserves all existing columns
- Adds 3 new columns: on_pitch_position, detailed_position, threat attributes
- Ready for downstream analysis

### Future Enhancements

#### Tactical Position Mapping (PENDING)
```python
# Requires database access to:
# 1. Map baseline player_id → StatsBomb player_id
# 2. Query starting XI from event JSON
# 3. Assign tactical position (GK, CB, RB, CM, ST, etc.)
# 4. Compare: tactical_position vs on_pitch_position
# 5. Compute tactical_flexibility metric

# Implementation: player_position_mapping.py (created but needs DB)
```

#### Formation Impact Analysis (PENDING)
```python
# Requires match_id mapping between baseline and StatsBomb event files
# 1. Extract formation from event JSON (e.g., "4-3-3", "5-2-3")
# 2. Join formation to baseline passes
# 3. Analyze: xT by (position + formation) combinations
# 4. Identify formation-specific threat patterns

# Implementation: event JSON parsing ready, needs match_id mapping
```

#### Temporal Analysis (FUTURE)
```python
# Analyze position-based xT over:
# - Match periods (H1 vs H2)
# - Time windows (0-30 min, 30-60 min, 60-90 min)
# - Formation changes (static formation vs dynamic)
# - Score states (leading, tied, trailing)
```

## Code Architecture

### Module: xt_position_refined.py (355 lines)

**Key Classes**:

1. **RefinedPositionAnalyzer**
   - Methods for position inference and analysis
   - Separate functions for broad and detailed positions
   - Decoupled from xTModel (accepts as parameter)
   - Extensible for custom position schemes

2. **Helper Functions**
   - `infer_broad_position(x)`: 3-category classification
   - `infer_detailed_position(x, y)`: 9-zone grid
   - `analyze_xt_by_position()`: Aggregation with statistics
   - `analyze_pitch_heatmap()`: Pivot table for visualization
   - `plot_*()`: Matplotlib-based visualization functions

3. **Main Pipeline**
   - `run_refined_analysis()`: Orchestrates full workflow
   - Loads baseline → Enriches → Analyzes → Visualizes → Exports

### Dependencies

```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns            # Enhanced heatmaps

from opponent_adjusted.features.xt_model import xTModel  # Threat classification
```

### Usage

```python
from pathlib import Path
from opponent_adjusted.analysis.xt_position_refined import run_refined_analysis

run_refined_analysis(
    baseline_path=Path("outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv"),
    output_dir=Path("outputs/analysis/cxa")
)
```

## Testing & Validation

### Data Integrity Checks
- ✅ Baseline loads correctly (4,363 rows)
- ✅ Position inference assigns all rows
- ✅ xT values preserved in enriched output
- ✅ Statistics compute without NaN issues
- ✅ Visualizations generate without errors

### Statistical Validation
- ✅ Position totals: 4 + 171 + 4,188 = 4,363 ✓
- ✅ xT sums: 0.0812 + 7.5813 + 390.469 = 398.13 ✓
- ✅ Percentages: 0.09% + 3.92% + 96.0% = 100% ✓
- ✅ Threat classification consistent with model

### Visual Inspection
- ✅ Bar charts readable with proper labels
- ✅ Heatmap colors correspond to xT values
- ✅ No missing data in outputs
- ✅ File sizes reasonable (93KB-788KB range)

## Limitations & Assumptions

### Current Limitations

1. **No Player Identity**
   - Can't correlate shots with actual players
   - Can't determine if "Forward" is striker vs fullback pushed forward
   - Workaround: Add player_id mapping

2. **No Tactical Context**
   - Don't know assigned positions (tactical vs on-pitch)
   - Can't measure tactical flexibility or role fluidity
   - Workaround: Add formation/lineup data extraction

3. **No Formation Context**
   - Don't know if 4-3-3 vs 5-2-3 affects position-based xT
   - Can't analyze formation-specific tactics
   - Workaround: Match baseline match_id to StatsBomb events

4. **Location Inference Only**
   - Position determined solely by pass_end_x coordinate
   - Y-coordinate (wing vs central) only used in detailed analysis
   - More nuanced position classification possible with:
     - Player role from lineup
     - Pass type and direction
     - Possession context

### Assumptions Made

1. **Shot location = position**
   - Assumes player location at pass end reflects their position in play
   - Valid for pass-based xA (more predictive than player name)

2. **X-coordinate thresholds are meaningful**
   - 0-40: Defensive third (valid)
   - 40-80: Midfield (valid)
   - 80-120: Attacking third (valid)
   - Could be refined with player-specific positioning

3. **Threat is static by location**
   - Uses xTModel grid values without context
   - Actual xT may vary by:
     - Game state (leading vs trailing)
     - Opposition defense state
     - Match phase (H1 vs H2)

## Recommendations

### Short-term (Implementable Now)
1. ✅ Use position-enriched baseline for downstream analysis
2. ✅ Export for external tools (Tableau, Power BI)
3. ✅ Compare position patterns across teams/matches
4. Add positional metadata in visualizations

### Medium-term (Requires Database)
1. Implement player_id mapping (database + event JSON)
2. Extract tactical positions from starting XI
3. Compare: tactical_position vs on_pitch_position
4. Compute tactical_flexibility metric

### Long-term (Full Integration)
1. Match baseline match_id to StatsBomb IDs
2. Extract formation from events
3. Analyze formation-specific position patterns
4. Temporal analysis (match progression, score state)
5. Machine learning: predict shot location from position

## Conclusion

This implementation provides a **position-based view of xT generation** across the dataset, revealing:

1. **Clear threat gradient**: Forward (0.093) > Midfielder (0.044) > Defender (0.020)
2. **Position specialization**: Central attacking positions highest threat (0.1117)
3. **Volume concentration**: 96% of shots from attacking third
4. **Foundation for enhancement**: Baseline enriched and ready for tactical/formation analysis

The position-refined analysis adds a spatial dimension to xT, complementing the existing grid-based and temporal analyses. It serves as a bridge toward more sophisticated tactical and role-based threat quantification.
