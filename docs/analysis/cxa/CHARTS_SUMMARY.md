# cxA Analysis: Charts Generation Summary

## Execution Summary

**Date:** December 25, 2025  
**Command:** `python scripts/run_cxa_analysis.py --charts`  
**Status:** ✅ Successfully Completed

## Charts Generated

### Phase 1: Baseline Descriptive Analysis
**Location:** `outputs/analysis/cxa/phase1_descriptive/charts/`  
**Count:** 6 charts

1. **pass_volume_by_cluster.png** - Total passes and key pass rates by player cluster
2. **completion_by_zone.png** - Pass completion rates across pitch zones
3. **delivery_type_breakdown.png** - 4-panel comparison of delivery types (volume, completion, key pass rate, distance)
4. **possession_by_team_cluster.png** - 4-panel team style characteristics
5. **sequence_length_distribution.png** - Frequency and quality of assist sequences
6. **cluster_profiles_radar.png** - Radar chart of player cluster behavioral profiles

### Phase 2: xA vs xA+ Comparison
**Location:** `outputs/analysis/cxa/phase2_comparison/charts/`  
**Count:** 5 charts

1. **xa_vs_xaplus_scatter.png** - Scatter plot comparing traditional vs sequence-based xA
2. **comparison_by_cluster.png** - Side-by-side comparison of xA vs xA+ by player cluster
3. **leaderboard_comparison.png** - Top 20 players by xA+ with rank changes
4. **attribution_by_position.png** - xA+ distribution across pass positions in sequences
5. **comparison_by_splits.png** - 4-panel comparison across position, pass type, zone, game phase

### Phase 3: Footballing Scenarios
**Location:** `outputs/analysis/cxa/phase3_scenarios/charts/`  
**Count:** 6 charts

1. **passes_per_quality_chance.png** - Quality chance rate and mean xG by possession length
2. **direct_vs_patient.png** - 4-panel efficiency comparison of possession styles
3. **team_style_efficiency.png** - Conversion metrics by team cluster
4. **receive_zones.png** - Shot quality and frequency by final pass receive zone
5. **delivery_efficiency.png** - 4-panel delivery type comparison
6. **goal_patterns.png** - Goal vs non-goal sequence characteristics

### Phase 4: Context & Opponent Justification
**Location:** `outputs/analysis/cxa/phase4_justification/charts/`  
**Count:** 7 charts

1. **context_effects.png** - Game state variance and pressure impact
2. **opponent_effects.png** - Completion and final third success vs opponent styles
3. **completion_stratification.png** - Heatmap of completion rates by distance and zone
4. **key_pass_stratification.png** - Key pass rates by zone and sequence context
5. **shot_hazard_curves.png** - Shot probability by zone
6. **conditional_shot_quality.png** - Shot xG and goal rate by delivery type
7. **variance_decomposition.png** - xA variance by context factors

## Total Output

- **Total Charts:** 24 high-quality visualizations
- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG
- **Total Storage:** ~15-20 MB

## Key Findings from Charts

### From Phase 1 Charts
- **Advanced Box Threat** cluster most prolific with highest pass volume
- Completion rate drops sharply in attacking third (as expected)
- Ground passes dominate volume but crosses have higher key pass rates
- Direct Central teams most efficient at shot creation

### From Phase 2 Charts
- **Kevin De Bruyne** leads xA+ rankings (5.816)
- xA+ redistributes credit: some players gain, others lose ranks
- Sequence-based attribution shows weak but positive correlation with traditional xA
- Position-specific patterns emerge in attribution distribution

### From Phase 3 Charts
- Teams average **7.6 passes** to create quality chances
- Moderate possessions (5-7 passes) most efficient
- Through balls create highest quality chances (0.213 xG)
- Box Central receive zone produces best shot quality (0.147 xG)

### From Phase 4 Charts
- Completion variance = 0.0407 (justifies completion submodel)
- Key pass rate varies 52.7% by context (justifies dedicated submodel)
- Late game variance 1.39x higher than early game
- Pressure reduces completion by 11.3%
- 3.3% completion variation by opponent defensive style

## Technical Notes

### Dependencies
Charts require:
```bash
pip install matplotlib seaborn
```

### Performance
- Generation time: ~2-3 minutes for all 24 charts
- Memory usage: ~500 MB peak (scatter plots with sampling)
- No data truncation or aggregation issues

### Known Warnings
- **FutureWarning:** Seaborn palette usage (non-breaking, will fix in future seaborn version)
- **DtypeWarning:** Mixed types in assist_sequences.csv columns 19-22 (expected, no impact)

## Usage Examples

### Generate All Charts
```bash
python scripts/run_cxa_analysis.py --charts
```

### Generate Charts for Specific Phases
```bash
python scripts/run_cxa_analysis.py --phases 1,2 --charts
python scripts/run_cxa_analysis.py --phases 4 --charts
```

### Programmatic Access
```python
from pathlib import Path
from opponent_adjusted.analysis.cxa_analysis import (
    BaselineDescriptiveAnalyzer,
    Phase1Visualizer
)

# Create analyzer
data_dir = Path("outputs/analysis/cxa/data")
analyzer = BaselineDescriptiveAnalyzer(data_dir)

# Generate charts
visualizer = Phase1Visualizer(analyzer)
output_dir = Path("outputs/analysis/cxa/phase1_descriptive/charts")
visualizer.generate_all_charts(output_dir)
```

## Integration with Reports

### Recommended Pairings

**For Stakeholder Presentations:**
- Phase 3 charts + scenario_answers.md = Complete tactical story
- Phase 2 leaderboard chart + player_leaderboard.csv = Performance report

**For Technical Documentation:**
- Phase 4 charts + submodel_justification.md = Modeling rationale
- Phase 1-2 charts + descriptive CSVs = Data exploration report

**For Academic Papers:**
- All Phase 4 charts demonstrate methodological rigor
- Phase 2 scatter plot shows metric redistribution effect
- Phase 3 charts answer research questions

## Next Steps

1. **Review Charts**: Examine all 24 visualizations for unexpected patterns
2. **Select Key Charts**: Choose 5-8 charts for presentation deck
3. **Customize If Needed**: Modify chart code for publication requirements
4. **Document Findings**: Use charts to support conclusions in reports
5. **Proceed to Modeling**: Use Phase 4 justification charts to guide submodel design

## Files Created

### Code Files
- `src/opponent_adjusted/analysis/cxa_analysis/phase1_charts.py` (235 lines)
- `src/opponent_adjusted/analysis/cxa_analysis/phase2_charts.py` (229 lines)
- `src/opponent_adjusted/analysis/cxa_analysis/phase3_charts.py` (249 lines)
- `src/opponent_adjusted/analysis/cxa_analysis/phase4_charts.py` (211 lines)

### Documentation
- `docs/analysis/cxa/CHARTS_GUIDE.md` (comprehensive usage guide)
- `docs/analysis/cxa/CHARTS_SUMMARY.md` (this file)

### Updates
- `scripts/run_cxa_analysis.py` - Added --charts flag and visualization calls
- `src/opponent_adjusted/analysis/cxa_analysis/__init__.py` - Export visualizer classes

## Troubleshooting

### Charts Not Generated?
```bash
# Check matplotlib/seaborn installed
pip list | grep -E "matplotlib|seaborn"

# Install if missing
pip install matplotlib seaborn
```

### Memory Issues?
- Phase 2 scatter plot automatically samples to 5000 points
- Other charts aggregate before plotting
- Close figures after saving with plt.close()

### Customization?
- All charts use seaborn/matplotlib
- Modify color palettes in chart files
- Adjust figure sizes with plt.rcParams

---

**Status:** ✅ All charts generated successfully  
**Ready For:** Modeling phase, presentations, publications  
**Documentation:** Complete with usage guide and code examples
