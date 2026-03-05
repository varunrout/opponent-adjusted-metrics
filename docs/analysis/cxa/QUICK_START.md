# CxA Analysis Quick Start Guide

Welcome to the cxA analysis phase! You've successfully acquired your data, and now it's time to analyze it before building predictive models.

## 📁 What You Have

Your data is ready in:
```
outputs/analysis/cxa/data/
├── passes.csv                 # All passes with features, clusters, xT, xA+
├── assist_sequences.csv       # Pre-shot sequences (1-3 passes before shots)
├── possessions.csv            # Possession-level aggregates
├── player_clusters.csv        # Player behavioral clusters (6 types)
├── team_clusters.csv          # Team style clusters (4 types)
├── shots.csv                  # Shot outcomes
└── lineups.csv                # Match context
```

## 🎯 Analysis Framework

The analysis follows a **progressive 4-phase approach**:

### Phase 1: Baseline Descriptive Analysis
**Goal:** Understand the data fundamentals

- Pass distributions by position, cluster, zone
- Possession patterns by team style
- Assist sequence characteristics
- Player and team cluster profiles

**Key Questions:**
- Who passes the most? From where?
- How do possessions differ by team style?
- What are typical assist sequences?

### Phase 2: xA vs xA+ Comparison
**Goal:** Compare traditional xA (key pass only) with xA+ (sequence-attributed)

**Comparisons across:**
- Player clusters
- Team clusters
- Position groups
- Pass types (cross, through ball, ground)
- Zones (defensive, middle, attacking)
- Pressure states
- Game phases

**Key Questions:**
- Who gains/loses value under xA+ vs xA?
- Does xA+ correlate better with goals?
- How is credit distributed in sequences?

### Phase 3: Footballing Scenarios
**Goal:** Answer real-world tactical questions

**9 Scenario Questions:**
1. How many passes to produce a quality chance?
2. Are direct possessions more efficient than patient build-ups?
3. What's the conversion efficiency of different team styles?
4. Where do the best chances come from? (receive zones)
5. What delivery type creates highest quality chances?
6. How does pressure affect chance creation?
7. What patterns precede goals vs blocked shots?
8. How important is the penultimate pass (second assist)?
9. Which player combinations produce dangerous sequences?

### Phase 4: Context & Opponent Justification
**Goal:** Build evidence for submodel architecture

**Justifications for:**
- ✅ Pass Completion Submodel (pressure, distance, zone effects)
- ✅ Key Pass Submodel (zone, sequence position effects)
- ✅ Shot-Within-K Hazard Submodel (zone-based shot probability)
- ✅ Conditional Shot Quality Submodel (delivery type effects on xG)
- ✅ Context Effects (game state, time, pressure)
- ✅ Opponent Effects (defensive style impacts)

## 🚀 How to Run

### Option 1: Run All Phases (Recommended for first time)
```bash
python scripts/run_cxa_analysis.py
```

### Option 2: Run Specific Phases
```bash
# Run only Phase 1
python scripts/run_cxa_analysis.py --phases 1

# Run Phases 1 and 2
python scripts/run_cxa_analysis.py --phases 1,2

# Run Phase 3 for scenario questions
python scripts/run_cxa_analysis.py --phases 3
```

### Option 3: Use Python API
```python
from pathlib import Path
from opponent_adjusted.analysis.cxa_analysis import (
    BaselineDescriptiveAnalyzer,
    XAComparisonAnalyzer,
    FootballingScenarioAnalyzer,
    ContextOpponentJustifier,
)

data_dir = Path("outputs/analysis/cxa/data")

# Phase 1
analyzer1 = BaselineDescriptiveAnalyzer(data_dir)
report1 = analyzer1.generate_summary_report()
# Access specific analyses
pass_vol = analyzer1.pass_volume_by_position()
cluster_profiles = analyzer1.player_cluster_profiles()

# Phase 2
analyzer2 = XAComparisonAnalyzer(data_dir)
by_cluster = analyzer2.compare_by_player_cluster()
leaderboard = analyzer2.player_leaderboard_comparison(top_n=30)

# Phase 3
analyzer3 = FootballingScenarioAnalyzer(data_dir)
passes_per_chance = analyzer3.passes_per_quality_chance()
delivery_efficiency = analyzer3.delivery_type_efficiency()

# Phase 4
analyzer4 = ContextOpponentJustifier(data_dir)
completion_just = analyzer4.justify_completion_submodel()
opponent_effect = analyzer4.opponent_quality_effect()
```

## 📊 Output Structure

After running, you'll have:

```
outputs/analysis/cxa/
├── data/                          # Your input data
├── phase1_descriptive/
│   ├── pass_volume_by_position.csv
│   ├── pass_volume_by_cluster.csv
│   ├── completion_by_zone.csv
│   ├── delivery_type_breakdown.csv
│   ├── possession_by_team_cluster.csv
│   ├── sequence_lengths.csv
│   ├── player_cluster_profiles.csv
│   ├── team_cluster_profiles.csv
│   └── ...
├── phase2_comparison/
│   ├── by_player_cluster.csv
│   ├── by_position_group.csv
│   ├── player_leaderboard.csv
│   ├── attribution_by_position.csv
│   ├── goal_correlations.csv
│   └── ...
├── phase3_scenarios/
│   ├── passes_per_quality_chance_data.csv
│   ├── direct_vs_patient_data.csv
│   ├── delivery_type_efficiency_data.csv
│   ├── scenario_answers.md          # Human-readable summary
│   └── ...
└── phase4_justification/
    ├── completion_submodel_evidence.csv
    ├── key_pass_submodel_evidence.csv
    ├── opponent_quality_effect_evidence.csv
    ├── submodel_justification.md     # Full justification report
    ├── neutralization_reference.csv
    └── variance_by_context.csv
```

## 🔍 What to Look For

### In Phase 1 (Descriptive)
- Cluster characteristics: Are they distinct and interpretable?
- Position patterns: Do they match football intuition?
- Sequence distributions: Most common patterns?

### In Phase 2 (Comparison)
- Player rank changes: Who benefits from xA+ attribution?
- Correlation improvements: Does xA+ predict goals better?
- Second assist contribution: Is it substantial (20-40%)?

### In Phase 3 (Scenarios)
- Optimal possession length: What creates best chances?
- Delivery trade-offs: Risk vs reward
- Zone effects: Where do dangerous chances come from?

### In Phase 4 (Justification)
- Variance in completion rates: Large enough to justify submodel?
- Context effects: Significant impact on xA?
- Opponent effects: Do defensive styles matter?

## 📝 Key Files to Review

1. **CXA_ANALYSIS_PLAN.md** - Full analysis methodology
2. **scenario_answers.md** - Tactical insights from data
3. **submodel_justification.md** - Evidence for modeling decisions

## 🎓 What This Analysis Achieves

By the end of all 4 phases, you will have:

✅ **Descriptive understanding** of passing, possession, and sequence patterns  
✅ **Comparative insights** on xA vs xA+ attribution methods  
✅ **Tactical answers** to 9 real-world football questions  
✅ **Empirical justification** for your submodel architecture  
✅ **Baseline references** for neutralization  
✅ **Evidence-based modeling decisions** ready for implementation

## 🔜 Next Steps After Analysis

Once analysis is complete:

1. **Review Findings** - Read the markdown summaries
2. **Validate Insights** - Do they match football intuition?
3. **Refine Features** - Based on what matters most
4. **Design Submodels** - Architecture now justified by data
5. **Proceed to Modeling** - Build predictive models with confidence

## 💡 Tips

- **Start with Phase 1** to get comfortable with the data
- **Use Phase 2** to understand attribution differences
- **Phase 3 is great for presentations** - real football questions
- **Phase 4 justifies your technical choices** to reviewers/stakeholders
- **Run phases independently** if you want to iterate on specific analyses

---

**Happy Analyzing! 🎉**

You're now equipped to conduct comprehensive cxA analysis that bridges data science with football tactics.
