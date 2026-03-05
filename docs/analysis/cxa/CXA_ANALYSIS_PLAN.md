# cxA Analysis Plan: From Descriptive to Submodel Justification

**Date:** 2025-12-25  
**Status:** Active Analysis Phase  
**Scope:** Descriptive → Comparative → Scenario → Submodel Justification

---

## Overview

This analysis plan follows a progressive approach to understanding chance creation patterns before building predictive models. The key principle is: **understand the data first, then model it**.

We have acquired the following datasets:
- `passes.csv` - All passes with features, xT, clusters, sequence info
- `assist_sequences.csv` - Pre-shot sequences (last k passes before shots)
- `possessions.csv` - Possession-level aggregates
- `player_clusters.csv` - Player clustering by passing behavior
- `team_clusters.csv` - Team clustering by playing style
- `shots.csv` - Shot outcomes linked to sequences
- `lineups.csv` - Match lineup context

---

## Phase 1: Baseline Descriptive Analysis

**Goal:** Establish foundational understanding of passes, possessions, and assist sequences with cluster-based splits.

### 1.1 Pass Distribution Analysis
- Total pass volume by position group and cluster
- Completion rates by zone (defensive/middle/attacking)
- Progressive pass distribution across player clusters
- Delivery type breakdown (ground/low/high, crosses, through balls)

### 1.2 Possession Analysis
- Possession duration distributions by team cluster
- Passes per possession by playing style
- Possession outcomes (shot/no-shot) by team cluster
- xT accumulation patterns within possessions

### 1.3 Assist Sequence Analysis
- Sequence length distribution (1-pass, 2-pass, 3-pass sequences)
- Pass chain patterns: who feeds whom?
- Zone progression within sequences (start → intermediate → final)
- Delivery types in final pass vs build-up passes

### 1.4 Cluster Profiles
- **Player Clusters:**
  - "Advanced Box Threat" - final third specialists
  - "Central Chance Creator" - creative hub players
  - "Central Circulator" - ball recyclers
  - "Advanced Box Threat 3" - secondary attackers
  - "Deep Chance Creator" - deep-lying playmakers
  - "Deep Progressive" - progressive defenders/GKs
  
- **Team Clusters:**
  - "Balanced" - neutral approach
  - "Direct Central" - central, fast build-up
  - "Possession Build-up" - patient possession
  - "Direct Wide" - wide-focused attacks

**Output:** Summary statistics tables + distribution plots

---

## Phase 2: xA vs xA+ Comparison Analysis

**Goal:** Compare baseline xA (key pass = shot xG) with xA+ (sequence-attributed value) across different splits.

### 2.1 Metric Definitions Recap
- **xA (Baseline):** Credit = xG of resulting shot, assigned only to final pass
- **xA+ (Sequence-Based):** Credit distributed across last k passes using:
  - Recency decay weights
  - xT contribution weights
  - Progress contribution

### 2.2 Comparison Splits
| Split Dimension | Groups |
|-----------------|--------|
| Player Cluster | 6 clusters |
| Team Cluster | 4 clusters |
| Position Group | Forward/Mid/Defender/GK |
| Pass Type | Ground/Low/High/Cross/Through |
| Zone | Defensive/Middle/Attacking |
| Pressure | Under pressure vs Open |
| Game Phase | Early/Mid/Late |

### 2.3 Key Questions
- Where does xA+ differ most from xA?
- Which player types gain/lose credit under xA+?
- How does sequence attribution change the assist leader boards?
- Is xA+ more correlated with actual goals than xA?

### 2.4 Visualizations
- Scatter: xA vs xA+ by player with cluster coloring
- Bar: Mean xA vs xA+ by position group
- Heatmap: Zone-to-Zone attribution patterns
- Line: Game phase effect on xA attribution

---

## Phase 3: Footballing Scenario Analysis

**Goal:** Answer real-world tactical questions about chance creation.

### 3.1 Possession Efficiency Questions
1. **"How many passes (irrespective of carries) does a team need to produce a quality chance?"**
   - Distribution of passes-per-possession for shots with xG > 0.1
   - Team cluster comparison
   - Optimal possession length for chance quality

2. **"Are shorter, direct possessions more efficient than long build-ups?"**
   - xG per possession by duration bucket
   - Directness score vs shot quality

3. **"What is the conversion efficiency of different team styles?"**
   - Shots per 100 possessions by team cluster
   - xG per 100 possessions by team cluster

### 3.2 Delivery and Receive Questions
4. **"Where do the best chances come from?"**
   - Shot xG by receive zone (final pass end location)
   - Box entries vs half-space entries vs central entries

5. **"What type of pass creates the highest quality chances?"**
   - Mean xG by delivery type (cross/through/cutback proxy/ground)
   - Completion risk vs reward tradeoff

6. **"How does pressure affect chance creation?"**
   - xA under pressure vs open play
   - Completion rates under pressure by position

### 3.3 Sequence Pattern Questions
7. **"What patterns precede goals vs blocked shots?"**
   - Sequence archetypes for goal-scoring chances
   - Progressive vs recycled possession goals

8. **"How important is the penultimate pass (second assist)?"**
   - xA+ attribution: final vs 2nd vs 3rd pass
   - Patterns where 2nd assist > final assist

9. **"Which player combinations produce the most dangerous sequences?"**
   - Passer-recipient pairs by sequence xA
   - Chemistry effects (same players repeated)

### 3.4 Visualizations
- Violin: Passes-per-chance by team cluster
- Sankey: Zone flow for goal-scoring sequences
- Pitch heatmap: High-value receive locations
- Network: Top passer-recipient combinations

---

## Phase 4: Contextual & Opponent Justification for Submodels

**Goal:** Demonstrate that context and opponent strength affect chance creation, justifying the need for submodels.

### 4.1 Context Effects Analysis

#### 4.1.1 Game State
- **Score differential effect:**
  - xA when winning vs drawing vs losing
  - Risk-taking (through balls, crosses) by score state
  - Hypothesis: Losing teams take more risks → more xA variance

- **Time effect:**
  - xA distribution by minute bucket (0-15, 15-30, ..., 75-90)
  - Urgency factor: are late-game key passes riskier?

#### 4.1.2 Pressure Effect
- Completion probability under pressure vs not
- xA preservation under pressure by player cluster
- Hypothesis: Elite creators maintain xA under pressure

### 4.2 Opponent Effects Analysis

#### 4.2.1 Opponent Quality Proxy
Using team cluster as a proxy for opponent quality:
- xA against "Balanced" vs "Direct Central" vs "Possession Build-up" defenses
- Completion rates against different opponent styles
- Hypothesis: Some creators thrive vs specific defensive styles

#### 4.2.2 Defensive Zone Pressure
- Key pass success rate by opponent's pressing intensity
- Final third entry success by opponent cluster
- xT gain against high vs low block defenses

### 4.3 Submodel Justification Analysis

#### 4.3.1 Pass Completion Submodel Justification
Show that completion probability varies significantly by:
- Pass distance and direction
- Pressure state
- Pass type (through ball harder than ground)
- Zone (final third completions harder)

**Evidence:** Completion rate tables stratified by these factors

#### 4.3.2 Key Pass Submodel Justification
Show that key pass probability varies by:
- Zone and receiver position
- Sequence position (passes late in possession more likely key)
- Delivery type

**Evidence:** Key pass rate tables stratified by factors

#### 4.3.3 Shot-Within-K Submodel Justification
Show that shot probability within k actions varies by:
- Current possession zone
- xT accumulated so far
- Team pressing state

**Evidence:** Hazard curves by zone and context

#### 4.3.4 Shot Quality Conditional Submodel Justification
Show that shot xG given shot varies by:
- Delivery type (crosses → headers → different xG)
- Receive location
- Game state

**Evidence:** Conditional xG distributions by delivery type

### 4.4 Neutralization Baseline Analysis
- Compute mean context values for neutralization reference:
  - Average pressure rate
  - Modal game state (0-0)
  - Reference minute (55th minute)
- Show variance in xA that would be removed by neutralization

---

## Output Structure

```
outputs/analysis/cxa/
├── data/                          # Already exists with raw features
├── phase1_descriptive/
│   ├── pass_distributions.png
│   ├── possession_analysis.png
│   ├── sequence_patterns.png
│   ├── cluster_profiles.md
│   └── summary_stats.csv
├── phase2_comparison/
│   ├── xa_vs_xaplus_scatter.png
│   ├── attribution_by_cluster.png
│   ├── leaderboard_comparison.csv
│   └── correlation_analysis.md
├── phase3_scenarios/
│   ├── passes_per_chance.png
│   ├── delivery_efficiency.png
│   ├── receive_zone_heatmap.png
│   ├── sequence_patterns.png
│   └── scenario_answers.md
└── phase4_justification/
    ├── context_effects.png
    ├── opponent_effects.png
    ├── completion_stratification.png
    ├── key_pass_stratification.png
    ├── shot_hazard_curves.png
    └── submodel_justification.md
```

---

## Implementation Order

1. **Start with Phase 1** - Get comfortable with data distributions
2. **Move to Phase 2** - Understand xA vs xA+ differences
3. **Phase 3** - Answer tactical questions for football validity
4. **Phase 4** - Build evidence for submodel architecture

Each phase builds on the previous, and insights from earlier phases inform later analysis.

---

## Next Steps

1. Create analysis modules in `src/opponent_adjusted/analysis/cxa_analysis/`
2. Create a Jupyter notebook for interactive exploration
3. Generate output artifacts for each phase
4. Document key findings that inform modeling decisions
