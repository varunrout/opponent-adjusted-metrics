# cxA Analysis: Visualization Guide

## Overview

The cxA analysis framework includes comprehensive visualization capabilities across all 4 phases. Charts complement the statistical outputs with visual insights for presentations and reports.

## Quick Start

### Generate All Charts

```bash
python scripts/run_cxa_analysis.py --charts
```

### Generate Charts for Specific Phases

```bash
# Only Phase 1 and 2
python scripts/run_cxa_analysis.py --phases 1,2 --charts

# All phases with charts
python scripts/run_cxa_analysis.py --phases 1,2,3,4 --charts
```

## Chart Output Structure

```
outputs/analysis/cxa/
├── phase1_descriptive/
│   └── charts/
│       ├── pass_volume_by_cluster.png
│       ├── completion_by_zone.png
│       ├── delivery_type_breakdown.png
│       ├── possession_by_team_cluster.png
│       ├── sequence_length_distribution.png
│       └── cluster_profiles_radar.png
├── phase2_comparison/
│   └── charts/
│       ├── xa_vs_xaplus_scatter.png
│       ├── comparison_by_cluster.png
│       ├── leaderboard_comparison.png
│       ├── attribution_by_position.png
│       └── comparison_by_splits.png
├── phase3_scenarios/
│   └── charts/
│       ├── passes_per_quality_chance.png
│       ├── direct_vs_patient.png
│       ├── team_style_efficiency.png
│       ├── receive_zones.png
│       ├── delivery_efficiency.png
│       └── goal_patterns.png
└── phase4_justification/
    └── charts/
        ├── context_effects.png
        ├── opponent_effects.png
        ├── completion_stratification.png
        ├── key_pass_stratification.png
        ├── shot_hazard_curves.png
        ├── conditional_shot_quality.png
        └── variance_decomposition.png
```

## Phase 1: Baseline Descriptive Charts

### 1. Pass Volume by Cluster
**File:** `pass_volume_by_cluster.png`
- **Purpose:** Compare total passes and key pass rates across player behavioral clusters
- **Insights:** Identifies which player types contribute most to chance creation
- **Use Case:** Understanding squad composition and role specialization

### 2. Completion by Zone
**File:** `completion_by_zone.png`
- **Purpose:** Bar chart of pass completion rates across pitch zones
- **Insights:** Shows difficulty gradient as passes move forward
- **Use Case:** Benchmarking zone-specific completion standards

### 3. Delivery Type Breakdown
**File:** `delivery_type_breakdown.png`
- **Purpose:** 4-panel comparison (volume, completion, key pass rate, distance) by delivery type
- **Insights:** Risk-reward profiles of ground passes, high passes, crosses
- **Use Case:** Tactical decisions on delivery method selection

### 4. Possession by Team Cluster
**File:** `possession_by_team_cluster.png`
- **Purpose:** 4-panel team style characteristics (passes/poss, duration, shot rate, xG/poss)
- **Insights:** Distinguishes Direct Central from Possession Build-up styles
- **Use Case:** Opponent scouting and tactical preparation

### 5. Sequence Length Distribution
**File:** `sequence_length_distribution.png`
- **Purpose:** Frequency and quality of 1/2/3-pass assist sequences
- **Insights:** Are longer sequences higher quality?
- **Use Case:** Justifying sequence-based xA+ attribution

### 6. Cluster Profiles Radar
**File:** `cluster_profiles_radar.png`
- **Purpose:** Radar chart of 5 key metrics for each player cluster
- **Insights:** Visual fingerprint of playing styles
- **Use Case:** Player scouting and role identification

## Phase 2: xA vs xA+ Comparison Charts

### 1. xA vs xA+ Scatter
**File:** `xa_vs_xaplus_scatter.png`
- **Purpose:** Scatter plot comparing traditional vs sequence-based xA
- **Insights:** Deviation from 45° line shows redistribution effect
- **Use Case:** Demonstrating xA+ methodology impact

### 2. Comparison by Cluster
**File:** `comparison_by_cluster.png`
- **Purpose:** Side-by-side bar charts: xA vs xA+ totals and percentage changes
- **Insights:** Which player types gain/lose value under xA+
- **Use Case:** Understanding position-specific attribution bias

### 3. Leaderboard Comparison
**File:** `leaderboard_comparison.png`
- **Purpose:** Top 20 players by xA+ with rank changes
- **Insights:** Kevin De Bruyne, Kimmich, Griezmann lead; who moved up/down?
- **Use Case:** Player performance narratives and recruitment

### 4. Attribution by Position
**File:** `attribution_by_position.png`
- **Purpose:** Stacked bar showing xA+ distribution across pass positions (1st/2nd/3rd)
- **Insights:** How credit flows through sequences
- **Use Case:** Valuing secondary assists

### 5. Comparison by Splits
**File:** `comparison_by_splits.png`
- **Purpose:** 4-panel comparison across position, pass type, zone, game phase
- **Insights:** Context-dependent differences between metrics
- **Use Case:** Understanding metric behavior in different scenarios

## Phase 3: Footballing Scenarios Charts

### 1. Passes per Quality Chance
**File:** `passes_per_quality_chance.png`
- **Purpose:** Quality chance rate and mean xG by possession length
- **Insights:** Average 7.6 passes needed; diminishing returns after ~10
- **Use Case:** Tactical planning for possession targets

### 2. Direct vs Patient
**File:** `direct_vs_patient.png`
- **Purpose:** 4-panel efficiency comparison (count, xG, goal rate, xG/poss)
- **Insights:** Moderate possessions (5-7 passes) most efficient
- **Use Case:** Style selection based on game state

### 3. Team Style Efficiency
**File:** `team_style_efficiency.png`
- **Purpose:** Conversion metrics (shots/100, xG/100, goal rate) by team cluster
- **Insights:** Direct Central best converters (1.25 xG/100 poss)
- **Use Case:** Recruitment and tactical model selection

### 4. Receive Zones
**File:** `receive_zones.png`
- **Purpose:** Shot quality and frequency by final pass receive zone
- **Insights:** Box Central highest quality (0.147 xG)
- **Use Case:** Training attacking patterns to target high-value zones

### 5. Delivery Efficiency
**File:** `delivery_efficiency.png`
- **Purpose:** 4-panel comparison (completion, xG, goal rate, risk-reward)
- **Insights:** Through balls create best chances (0.213 xG when successful)
- **Use Case:** Final pass selection guidance

### 6. Goal Patterns
**File:** `goal_patterns.png`
- **Purpose:** Goal vs non-goal sequence characteristics (passes, xT, progressive, directness)
- **Insights:** Goals come from more progressive, direct sequences
- **Use Case:** Defining "dangerous" sequence patterns

## Phase 4: Submodel Justification Charts

### 1. Context Effects
**File:** `context_effects.png`
- **Purpose:** Game state variance and pressure impact on completion
- **Insights:** Late game variance 1.39x higher; pressure reduces completion 11.3%
- **Use Case:** Justifying context-aware submodels

### 2. Opponent Effects
**File:** `opponent_effects.png`
- **Purpose:** Completion and final third entry success vs different opponent styles
- **Insights:** 3.3% completion variation by defensive style
- **Use Case:** Justifying opponent-adjusted metrics

### 3. Completion Stratification
**File:** `completion_stratification.png`
- **Purpose:** Heatmap of completion rates by distance and zone
- **Insights:** Variance = 0.0407 (37%-96% range) justifies submodel
- **Use Case:** Modeling pass completion probability

### 4. Key Pass Stratification
**File:** `key_pass_stratification.png`
- **Purpose:** Key pass rates by zone and sequence context
- **Insights:** 52.7% variation justifies dedicated submodel
- **Use Case:** Modeling key pass probability separately

### 5. Shot Hazard Curves
**File:** `shot_hazard_curves.png`
- **Purpose:** Shot probability within k actions by zone
- **Insights:** 9.7% variation across zones
- **Use Case:** Justifying shot hazard submodel

### 6. Conditional Shot Quality
**File:** `conditional_shot_quality.png`
- **Purpose:** Shot xG and goal rate by final pass delivery type
- **Insights:** 0.152 xG variation by delivery
- **Use Case:** Justifying conditional shot quality submodel

### 7. Variance Decomposition
**File:** `variance_decomposition.png`
- **Purpose:** xA variance by context factors (game phase, zone, pressure)
- **Insights:** Quantifies neutralization requirements
- **Use Case:** Designing fair player comparisons

## Programmatic Usage

### Python API

```python
from pathlib import Path
from opponent_adjusted.analysis.cxa_analysis import (
    BaselineDescriptiveAnalyzer,
    Phase1Visualizer
)

# Load data
data_dir = Path("outputs/analysis/cxa/data")
analyzer = BaselineDescriptiveAnalyzer(data_dir)

# Generate charts
visualizer = Phase1Visualizer(analyzer)
output_dir = Path("outputs/analysis/cxa/phase1_descriptive/charts")
visualizer.generate_all_charts(output_dir)

# Or individual charts
visualizer.plot_pass_volume_by_cluster(output_dir / "custom_name.png")
```

### Jupyter Notebook

```python
import matplotlib.pyplot as plt
from opponent_adjusted.analysis.cxa_analysis import Phase2Visualizer, XAComparisonAnalyzer

# Create analyzer
analyzer = XAComparisonAnalyzer(Path("outputs/analysis/cxa/data"))
visualizer = Phase2Visualizer(analyzer)

# Generate chart in-notebook
fig, ax = plt.subplots(figsize=(10, 10))
# ... custom plotting code using visualizer.analyzer data
plt.show()
```

## Chart Customization

All visualizers use seaborn/matplotlib. To customize:

1. **Copy chart method** from `phaseX_charts.py`
2. **Modify style**: `sns.set_style("darkgrid")`, `plt.rcParams['figure.figsize'] = (16, 10)`
3. **Change colors**: `palette='viridis'`, `color='#e74c3c'`
4. **Adjust labels**: `ax.set_title()`, `ax.set_xlabel()`
5. **Add annotations**: `ax.text()`, `ax.annotate()`

## Dependencies

Charts require additional packages:
```bash
pip install matplotlib seaborn
```

If packages not installed, analysis runs without charts (no errors).

## Best Practices

1. **Always generate charts** when preparing presentations or reports
2. **Use Phase 1-2 charts** for exploratory understanding
3. **Use Phase 3 charts** for tactical questions and stakeholder communication
4. **Use Phase 4 charts** for technical audiences and modeling justification
5. **High DPI exports** (300 dpi) suitable for publications
6. **Combine with markdown reports** for comprehensive documentation

## Troubleshooting

### Charts not generating?
```bash
# Check if matplotlib/seaborn installed
pip list | grep -E "matplotlib|seaborn"

# Install if missing
pip install matplotlib seaborn
```

### Memory issues with large datasets?
- Phase2 scatter plot samples to 5000 points if dataset larger
- Other charts aggregate before plotting
- Reduce `top_n` parameters in leaderboard plots

### Custom color schemes?
```python
# Modify at top of phaseX_charts.py
import seaborn as sns
sns.set_palette("husl")  # or custom palette
```

## Integration with Reports

Charts complement the markdown and CSV outputs:

- **scenario_answers.md** + **scenario charts** = complete tactical story
- **submodel_justification.md** + **justification charts** = modeling documentation
- **player_leaderboard.csv** + **leaderboard chart** = performance report

## Next Steps

After generating charts:
1. Review all charts for unexpected patterns
2. Include relevant charts in stakeholder presentations
3. Use Phase 4 charts to guide submodel architecture
4. Proceed to modeling phase with visual evidence

---

**Chart Count:** 24 total visualizations across 4 phases  
**Resolution:** 300 DPI (publication quality)  
**Format:** PNG (easily embeddable)
