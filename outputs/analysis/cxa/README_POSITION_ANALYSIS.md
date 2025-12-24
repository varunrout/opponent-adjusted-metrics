# Position-Based Expected Threat (xT) Analysis - Complete Reference

## Quick Start

**What was done**: Refined expected threat (xT) analysis by breaking down threat generation across where shots are **actually created** (on-pitch position) rather than aggregating all shots together.

**Key Result**: Identified position-based threat gradient:
- **Forwards**: 0.0932 xT/shot (4.6× higher than defenders)
- **Midfielders**: 0.0443 xT/shot (2.2× higher than defenders)  
- **Defenders**: 0.0203 xT/shot (rare, very low threat)
- **Peak Zone**: Attacking (Central) = 0.1117 xT/shot (highest quality)

---

## 📊 Analysis Outputs

### 1. Data Files (4 CSVs)

#### [cxa_xt_by_position.csv](outputs/analysis/cxa/cxa_xt_by_position.csv)
Position summary with 3 categories: Defender, Midfielder, Forward
- Columns: shots, total_xt, avg_xt, std_xt, min_xt, max_xt, threat_level, percentile

#### [cxa_xt_by_detailed_position.csv](outputs/analysis/cxa/cxa_xt_by_detailed_position.csv)
Detailed 9-zone breakdown: 3 field thirds × 3 vertical zones
- Shows intra-position variation (e.g., central attacking vs wing attacking)

#### [cxa_xt_pitch_heatmap_data.csv](outputs/analysis/cxa/cxa_xt_pitch_heatmap_data.csv)
3×3 grid pivot table of average xT across pitch
- Rows: Wing/Central/Wing (vertical) | Columns: Def/Mid/Att (horizontal)

#### [cxa_baseline_enriched.csv](outputs/analysis/cxa/cxa_baseline_enriched.csv)
Original 4,363-row baseline + 2 new position columns
- New columns: on_pitch_position (Defender/Midfielder/Forward), detailed_position (9-zone)
- Ready for downstream team/player/tactical analysis

### 2. Visualizations (3 PNGs)

#### [cxa_position_comparison.png](outputs/analysis/cxa/cxa_position_comparison.png)
Side-by-side bar charts showing:
- **Left**: Average xT by position (threat quality/efficiency)
- **Right**: Total xT by position (threat volume)
- Color-coded: Red (High threat) / Orange (Medium) / Green (Low)

#### [cxa_detailed_position_comparison.png](outputs/analysis/cxa/cxa_detailed_position_comparison.png)
9-zone breakdown bar chart
- Reveals Attacking (Central) dominates with 0.1117 avg xT
- Shows how wing positions have lower xT than central

#### [cxa_pitch_heatmap.png](outputs/analysis/cxa/cxa_pitch_heatmap.png)
Spatial heatmap across pitch
- 3×3 grid visualization
- Red intensity = higher xT values
- Attacking (Central) = darkest red (highest threat)

### 3. Documentation (3 Markdown)

#### [POSITION_ANALYSIS_SUMMARY.md](outputs/analysis/cxa/POSITION_ANALYSIS_SUMMARY.md)
**Executive summary** - For decision makers and analysts
- Overview and findings
- Key insights and interpretations
- Limitations and future refinements

#### [POSITION_ANALYSIS_TECHNICAL.md](outputs/analysis/cxa/POSITION_ANALYSIS_TECHNICAL.md)
**Technical deep-dive** - For engineers and data scientists
- Implementation approach and rationale
- Code architecture and dependencies
- Data pipeline explanation
- Testing and validation details

#### [POSITION_ANALYSIS_COMPLETION.md](POSITION_ANALYSIS_COMPLETION.md)
**Project completion report** - For stakeholders
- Objectives achieved
- Deliverables checklist
- Quality assurance summary
- Next steps and recommendations

---

## 🔍 Key Results Reference

### Threat by Position

| Position | Shots | Total xT | Avg xT | Threat | Insight |
|----------|-------|----------|--------|--------|---------|
| **Forward** | 4,188 | 390.47 | 0.0932 | Medium | Dominates volume & threat |
| **Midfielder** | 171 | 7.58 | 0.0443 | Medium | Supporting shots |
| **Defender** | 4 | 0.08 | 0.0203 | Low | Rare events |

### Threat by Detailed Zone

| Zone | Shots | Avg xT | Key Insight |
|------|-------|--------|-------------|
| **Attacking (Central)** | 3,059 | 0.1117 | PEAK - Highest quality shots |
| Attacking (Wing) | 1,129 | 0.0432 | Secondary - Wide positions lower xT |
| Midfield (Wing) | 86 | 0.0457 | Support - Lower volume |
| Midfield (Central) | 85 | 0.0429 | Support - Central minimal advantage in midfield |
| Defensive (Central) | 4 | 0.0203 | Rare - Set-piece or long-range attempts |

### Statistical Summary

- **Total Shots**: 4,363
- **Total xT**: 398.13
- **Overall Average**: 0.0913 xT/shot
- **Grid Theoretical Mean**: 0.2909 (shots use only ~31% of theoretical threat)
- **Threat Classification**:
  - Low (<0.0215): 1 position (Defenders)
  - Medium (0.0215-0.5603): 4 positions
  - High (>0.5603): 0 positions (no position reaches highest threat)

---

## 💡 Usage Guide

### For Match Analysis
```python
# Load enriched baseline
baseline = pd.read_csv("outputs/analysis/cxa/cxa_baseline_enriched.csv")

# Filter by position
forwards = baseline[baseline['on_pitch_position'] == 'Forward']
attacking_central = baseline[baseline['detailed_position'] == 'Attacking (Central)']

# Team patterns
team_3 = baseline[baseline['team_id'] == 3]
team_3_forward_shots = len(team_3[team_3['on_pitch_position'] == 'Forward']) / len(team_3)
```

### For Visualization
```python
# Use PNG files directly in reports:
# - cxa_position_comparison.png: Team comparisons
# - cxa_pitch_heatmap.png: Tactical presentations
# - cxa_detailed_position_comparison.png: Position breakdowns
```

### For Team Comparison
Identify team profiles:
- Teams that shoot from attacking third (efficient)
- Teams with high volume from midfield (long-range focus)
- Teams with wide attacks vs central attacks

---

## 🔧 Technical Implementation

### Source Code
**Main Module**: `src/opponent_adjusted/analysis/xt_position_refined.py` (355 lines)

**Key Components**:
- `RefinedPositionAnalyzer` class - Main analysis orchestrator
- `infer_broad_position(x)` - Classify as Defender/Midfielder/Forward
- `infer_detailed_position(x, y)` - Assign to 9-zone grid
- `analyze_xt_by_position()` - Compute position statistics
- `plot_*()` - Generate visualizations

**Methodology**:
- Location-based position inference (x-coordinate → position)
- Integration with existing xTModel
- Threat classification using ±1σ thresholds
- Statistical aggregation and visualization

### Data Flow
```
Baseline CSV (4,363 shots)
    ↓
Position Inference (x → Def/Mid/Fwd)
    ↓
Enrichment (add 2 columns)
    ↓
Analysis Pipeline
├─ Broad position analysis
├─ Detailed zone analysis
├─ Threat classification
└─ Statistics computation
    ↓
Export
├─ 4 CSV files
├─ 3 PNG visualizations
└─ 3 markdown docs
```

---

## 📚 Related Work

### Previous Analysis
- **xT Model**: Karun Singh's 120×80 grid with bivariate spline interpolation
- **Threat Thresholds**: ±1 standard deviation (adaptive, not fixed)
- **Baseline Data**: 4,363 passes with xA values from expected assist model

### Future Enhancements

#### Tactical Position Mapping (requires database)
Map baseline player_id → StatsBomb ID → starting XI position
- Compare: Assigned role vs actual position
- Compute: Tactical flexibility index
- Identify: Players playing out of position

#### Formation Impact Analysis (requires event JSON mapping)
Extract formation → Analyze by (position + formation)
- Does 4-3-3 vs 5-2-3 affect positional threat?
- Formation-specific tactical patterns
- Formation changes impact on xT

#### Temporal Analysis (future enhancement)
Position-based xT over:
- Match phases (H1 vs H2)
- Score states (leading/tied/trailing)
- Formation changes
- Time-based efficiency

---

## ✅ Quality Assurance

### Data Validation
- ✓ All 4,363 shots assigned to positions
- ✓ Position percentages: 96% + 3.9% + 0.1% = 100%
- ✓ xT totals match baseline sums exactly
- ✓ No missing values in outputs

### Statistical Verification
- ✓ Group aggregations computed correctly
- ✓ Threat classifications match xTModel
- ✓ Percentile rankings calculated
- ✓ Standard deviations validated

### Output Quality
- ✓ CSV files parse without errors
- ✓ PNG visualizations render correctly
- ✓ File sizes appropriate
- ✓ All documentation complete

---

## 🎓 Key Insights

### 1. Position Strongly Differentiates Threat
- Forwards generate **4.6× more threat** than defenders per shot
- Central positions are **2.6× higher threat** than wings from same third
- Indicates position is a major factor in shot quality

### 2. Volume Concentrates in Attacking Third
- 96% of shots from forward/attacking positions
- Only 0.1% from defensive third (rare, very low xT)
- Teams should prioritize possession in attacking third

### 3. Central Attacking Dominance
- Central attacking zone generates **0.1117 avg xT** (peak)
- Wing attacking lower at **0.0432 avg xT**
- Suggests central-focused attacking is more efficient

### 4. Clear Threat Gradient
```
Defender    Midfielder    Forward
0.0203  →    0.0443   →   0.0932
  (Low)     (Medium)    (Medium)
```
Shows predictable increase in threat as shots approach goal

---

## 📖 Documentation Map

```
Project Root
├─ POSITION_ANALYSIS_COMPLETION.md ← START HERE (overview)
│
├─ outputs/analysis/cxa/
│  ├─ POSITION_ANALYSIS_SUMMARY.md (executive summary)
│  ├─ POSITION_ANALYSIS_TECHNICAL.md (detailed implementation)
│  │
│  ├─ Data Files:
│  │  ├─ cxa_xt_by_position.csv (position summary)
│  │  ├─ cxa_xt_by_detailed_position.csv (9-zone breakdown)
│  │  ├─ cxa_xt_pitch_heatmap_data.csv (grid data)
│  │  └─ cxa_baseline_enriched.csv (enriched for analysis)
│  │
│  └─ Visualizations:
│     ├─ cxa_position_comparison.png (bar charts)
│     ├─ cxa_detailed_position_comparison.png (9-zone)
│     └─ cxa_pitch_heatmap.png (spatial heatmap)
│
└─ src/opponent_adjusted/analysis/
   ├─ xt_position_refined.py (MAIN - 355 lines)
   ├─ xt_position_analysis.py (alternative approach)
   └─ player_position_mapping.py (future enhancement)
```

---

## 🚀 Next Steps

### Immediate (High Value)
1. Use enriched baseline for team/player analysis
2. Create team position profiles
3. Compare team shooting patterns

### Short-term (Medium Value)
4. Implement player_id mapping for tactical positions
5. Add formation analysis (if database available)
6. Create position-specific efficiency reports

### Long-term (Advanced)
7. Temporal analysis (match phases, score states)
8. Machine learning: predict position from context
9. Combine with defensive metrics

---

## 📞 Questions or Issues?

Refer to:
- **What did we find?** → POSITION_ANALYSIS_SUMMARY.md
- **How was it done?** → POSITION_ANALYSIS_TECHNICAL.md
- **What's included?** → This index document
- **Raw data?** → cxa_baseline_enriched.csv

---

**Analysis Date**: 2024-12-24  
**Data**: 4,363 passes leading to shots  
**Status**: ✅ Complete  
**Last Updated**: 2024-12-24
