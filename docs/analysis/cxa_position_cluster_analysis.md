# Position-Based Cluster Analysis for xA

## Overview

The cxA analysis has been refactored to focus on **position-based cluster analysis** rather than individual player analysis. This approach provides insights into how different team styles utilize different areas of the pitch for chance creation.

## Key Changes

### Removed
- **Individual player analysis** (`cxa_player_clusters_scatter.png` and related)
  - Player-level data lacks sufficient context for meaningful clustering
  - Individual player names are less actionable than position/style patterns

### Added
- **Position-based analysis** by team style
  - Position groups inferred from pass origin location
  - Analysis split by team style clusters
  - Both overall and cluster-specific position profiles

## Position Groups

Positions are inferred from the pitch location where passes originate:

| Position Group | x-coordinate Range | Pitch Region |
|---|---|---|
| **Defenders** | 0-40 | Defensive third |
| **Midfielders** | 40-80 | Middle third |
| **Forwards** | 80-120 | Attacking third |

This approach is more robust than trying to match player IDs across datasets, and directly reflects where teams create chances from.

## Team Style Clusters

Teams are grouped into 4 attacking style clusters based on their passing patterns:

1. **Wide Play Buildup** (Tunisia, South Korea, Colombia, etc.)
   - Cross%: 28.4%, Through%: 2.5%, Ground%: 46.8%
   - Moderate width, lower possession, attacking third focus
   
2. **Direct / Wide Attacking** (Serbia, Australia, Ecuador, etc.)
   - Cross%: 32.5%, Through%: 1.8%, Ground%: 35.4%
   - Highest crossing %, deepest passes (end_x: 104.7)
   - Direct, wide-focused attacking
   
3. **Possession Creators** (Switzerland, Argentina, Brazil, France, Germany, etc.)
   - Cross%: 23.9%, Through%: 4.8%, Ground%: 52.4%, xA/pass: 0.096
   - High through-ball usage, possession-oriented
   - Highest xA creation efficiency
   
4. **Possession / Conservative** (Denmark, Croatia, Egypt, etc.)
   - Cross%: 22.0%, Through%: 1.5%, Ground%: 58.3%
   - Highest ground pass %, lowest end_x (99.9)
   - Very possession-focused, less penetrative

## Key Findings

### Overall Position Distribution
- **Forwards create 98%+ of all xA** (390.5 xA+)
  - Mean xA/pass: 0.093
  - This includes all attacking-third passes
  
- **Midfielders contribute ~2%** (7.6 xA+)
  - Mean xA/pass: 0.044
  - Lower quality but some playmaking

- **Defenders contribute <1%** (0.08 xA+)
  - Mean xA/pass: 0.020
  - Rare attacking contributions

### Position Distribution by Team Style

#### Wide Play Buildup
- Forwards: 97.3% of xA
- Midfielders: 2.7% of xA
- Few defensive third actions

#### Direct / Wide Attacking
- Forwards: 98.9% of xA
- Midfielders: 1.1% of xA
- Most forward-focused style

#### Possession Creators
- Forwards: 98.0% of xA
- Midfielders: 2.0% of xA
- Defenders: 0.03% of xA
- Only style with some defensive contributions

#### Possession / Conservative
- Forwards: 97.7% of xA
- Midfielders: 2.3% of xA
- Very forward-focused despite possession style

## Outputs

### CSVs
- `cxa_position_groups_overall.csv` - Position stats for all teams
- `cxa_position_groups_cluster1.csv` through `cluster4.csv` - Per-cluster position stats

### Visualizations
- `cxa_position_groups_by_cluster.png` - Multi-panel comparison of position contribution by cluster
- `cxa_position_radar_by_cluster.png` - Radar charts showing position profiles for each team style

## Interpretation

The analysis reveals:

1. **Position is the strongest predictor of xA contribution**
   - Attacking third passes dominate chance creation
   - Positional focus (where teams build from) varies more by style than by individual players

2. **Team style is reflected in creation location**
   - Possession teams (cluster 3) still create most xA from forwards
   - But have slightly more midfield involvement (2.0% vs 1.1% for direct teams)
   - Suggests possession teams build through more layers

3. **No strong defensive xA creation**
   - Even for possession-heavy teams, defensive third plays minimal xA role
   - Teams that keep possession still need forward positioning for chances

## Related Analysis

- [Team Analysis](team_analysis.md) - Team style clustering and profiles
- [Spatial Analysis](spatial_analysis.md) - Zone-based and corridor analysis
- [Baseline Analysis](../methodology/01_data_ingestion.md) - Raw xA computation

## Methodology

Pass-level xA data is aggregated by:
1. **Position group** (inferred from pass origin x-coordinate)
2. **Team style cluster** (from hierarchical clustering on team passing patterns)

This provides cross-classified view of xA creation showing both the location (position) and context (team style) of chance creation.
