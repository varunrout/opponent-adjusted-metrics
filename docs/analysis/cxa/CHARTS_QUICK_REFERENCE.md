# cxA Charts: Quick Reference

## Command Line Usage

```bash
# Generate all charts
python scripts/run_cxa_analysis.py --charts

# Specific phases only
python scripts/run_cxa_analysis.py --phases 1,2 --charts
python scripts/run_cxa_analysis.py --phases 4 --charts

# Analysis without charts
python scripts/run_cxa_analysis.py
```

## Chart Locations

```
outputs/analysis/cxa/
├── phase1_descriptive/charts/        (6 charts)
├── phase2_comparison/charts/         (5 charts)
├── phase3_scenarios/charts/          (6 charts)
└── phase4_justification/charts/      (7 charts)
```

## Best Charts by Use Case

### For Executive Presentations
1. `phase3/passes_per_quality_chance.png` - Possession strategy insights
2. `phase2/leaderboard_comparison.png` - Player performance rankings
3. `phase3/team_style_efficiency.png` - Team style comparison

### For Scouting Reports
1. `phase1/cluster_profiles_radar.png` - Player type identification
2. `phase2/leaderboard_comparison.png` - Top creators ranking
3. `phase1/pass_volume_by_cluster.png` - Role characteristics

### For Tactical Analysis
1. `phase3/receive_zones.png` - Target zones for final passes
2. `phase3/delivery_efficiency.png` - Delivery type selection
3. `phase3/direct_vs_patient.png` - Possession style efficiency

### For Technical Documentation
1. `phase4/completion_stratification.png` - Submodel justification
2. `phase4/variance_decomposition.png` - Context effects
3. `phase2/xa_vs_xaplus_scatter.png` - Metric comparison

### For Academic Papers
1. All `phase4/` charts - Methodological rigor
2. `phase2/comparison_by_splits.png` - Metric behavior analysis
3. `phase1/sequence_length_distribution.png` - Data characteristics

## Key Insights by Chart

| Chart | Key Insight | Numeric |
|-------|-------------|---------|
| passes_per_quality_chance | Average passes needed | 7.6 |
| leaderboard_comparison | Top creator | De Bruyne: 5.82 xA+ |
| team_style_efficiency | Best converters | Direct Central |
| receive_zones | Best receive zone | Box Central: 0.147 xG |
| delivery_efficiency | Best delivery | Through Ball: 0.213 xG |
| completion_stratification | Completion variance | 0.0407 |
| key_pass_stratification | Key pass variation | 52.7% |
| context_effects | Late game variance | 1.39x higher |
| pressure_effect | Pressure impact | -11.3% completion |
| opponent_effects | Style variation | 3.3% completion |

## Python API

```python
from pathlib import Path
from opponent_adjusted.analysis.cxa_analysis import *

# Phase 1
analyzer1 = BaselineDescriptiveAnalyzer(Path("outputs/analysis/cxa/data"))
viz1 = Phase1Visualizer(analyzer1)
viz1.generate_all_charts(Path("output_dir"))

# Phase 2
analyzer2 = XAComparisonAnalyzer(Path("outputs/analysis/cxa/data"))
viz2 = Phase2Visualizer(analyzer2)
viz2.plot_leaderboard_comparison(Path("custom_leaderboard.png"))

# Phase 3
analyzer3 = FootballingScenarioAnalyzer(Path("outputs/analysis/cxa/data"))
viz3 = Phase3Visualizer(analyzer3)
viz3.plot_passes_per_quality_chance(Path("passes.png"))

# Phase 4
analyzer4 = ContextOpponentJustifier(Path("outputs/analysis/cxa/data"))
viz4 = Phase4Visualizer(analyzer4)
viz4.plot_completion_stratification(Path("completion.png"))
```

## Common Tasks

### Export for Presentation
```bash
# Charts are already 300 DPI PNG - ready to use
# Copy to presentation folder:
cp outputs/analysis/cxa/*/charts/*.png presentation/
```

### View in Jupyter
```python
from IPython.display import Image, display

# Display chart
display(Image("outputs/analysis/cxa/phase1_descriptive/charts/cluster_profiles_radar.png"))
```

### Combine with Data
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("outputs/analysis/cxa/phase2_comparison/player_leaderboard.csv")

# Use in custom chart
fig, ax = plt.subplots(figsize=(10, 6))
df.head(10).plot(x='passer_name', y='total_xa_plus', kind='barh', ax=ax)
plt.savefig("custom_chart.png", dpi=300, bbox_inches='tight')
```

## Customization Tips

### Change Colors
```python
# In phaseX_charts.py
sns.set_palette("viridis")  # or "husl", "Set2", etc.
```

### Adjust Size
```python
# In phaseX_charts.py
plt.rcParams['figure.figsize'] = (14, 10)
```

### Add Annotations
```python
ax.text(x, y, 'annotation', fontsize=12, color='red')
ax.annotate('label', xy=(x, y), xytext=(x2, y2), arrowprops=dict(arrowstyle='->'))
```

## Dependencies

```bash
# Required
pip install matplotlib seaborn

# Optional (for advanced customization)
pip install plotly  # Interactive charts
pip install pillow  # Image manipulation
```

## File Sizes

- Each chart: ~500 KB - 2 MB (300 DPI)
- Total: ~15-20 MB for all 24 charts
- Suitable for: presentations, publications, web

## Performance

- **Generation time:** 2-3 minutes for all 24 charts
- **Memory usage:** ~500 MB peak
- **Concurrent execution:** Not recommended (sequential is safer)

## See Also

- [CHARTS_GUIDE.md](CHARTS_GUIDE.md) - Comprehensive usage guide
- [CHARTS_SUMMARY.md](CHARTS_SUMMARY.md) - Execution summary
- [QUICK_START.md](QUICK_START.md) - Analysis quick start
- [CXA_ANALYSIS_PLAN.md](CXA_ANALYSIS_PLAN.md) - Methodology
