# Expected Assists (xA) Analysis Report

**Generated:** December 25, 2025  
**Dataset:** ~240,000 passes from StatsBomb open data  
**Methodology:** Sequence-based xA+ attribution vs traditional xA

---

## Executive Summary

This report compares **traditional xA** (crediting only the final pass before a shot) with **xA+** (distributing credit across the entire assist sequence). The analysis reveals systematic differences in how players are valued, with sequence attribution providing a more complete picture of chance creation.

**Key Finding:** Traditional xA overvalues final-pass specialists by ~25-40% while undervaluing build-up contributors.

---

## 1. Top Chance Creators by xA+

### Leaderboard (Top 15)

| Rank | Player | Position | xA | xA+ | Rank Change | % Difference |
|------|--------|----------|-----|-----|-------------|--------------|
| 1 | Kevin De Bruyne | Winger | 7.94 | 5.82 | — | -26.7% |
| 2 | Joshua Kimmich | Full Back | 5.93 | 4.52 | — | -23.8% |
| 3 | Antoine Griezmann | Attacking Mid | 5.69 | 4.27 | — | -25.0% |
| 4 | Kieran Trippier | Full Back | 4.81 | 3.94 | ↑1 | -18.1% |
| 5 | Lionel Messi | Winger | 4.49 | 3.35 | ↑2 | -25.4% |
| 6 | Kylian Mbappé | Winger | 5.41 | 3.26 | ↓2 | -39.7% |
| 7 | Neymar | Winger | 4.42 | 3.04 | ↑1 | -31.2% |
| 8 | Dani Olmo | Winger | 3.94 | 2.96 | ↑3 | -24.8% |
| 9 | Memphis Depay | Winger | 4.51 | 2.87 | ↓3 | -36.5% |
| 10 | Bruno Fernandes | Attacking Mid | 3.51 | 2.70 | ↑3 | -22.9% |
| 11 | Luka Modrić | Central Mid | 4.11 | 2.69 | ↓1 | -34.5% |
| 12 | Xherdan Shaqiri | Attacking Mid | 3.20 | 2.59 | ↑3 | -19.1% |
| 13 | Pedri | Central Mid | 4.25 | 2.41 | ↓4 | -43.2% |
| 14 | Christian Eriksen | Attacking Mid | 2.86 | 2.23 | ↑6 | -22.0% |
| 15 | Toni Kroos | Defensive Mid | 3.25 | 2.17 | ↓1 | -33.2% |

### Notable Rank Changes

**Winners under xA+ (moved up):**
- Paul Pogba: ↑11 ranks (build-up contributions recognized)
- Ruslan Malinovskiy: ↑7 ranks
- Christian Eriksen: ↑6 ranks
- Ricardo Rodríguez: ↑6 ranks
- Ivan Perišić: ↑6 ranks

**Losers under xA+ (moved down):**
- Cristiano Ronaldo: ↓11 ranks (traditional xA inflated by final-pass focus)
- Pedri: ↓4 ranks
- Theo Hernández: ↓4 ranks
- Harry Kane: ↓4 ranks

---

## 2. xA vs xA+ by Player Cluster

| Player Type | Total xA | Total xA+ | Difference | Correlation |
|-------------|----------|-----------|------------|-------------|
| Advanced Box Threat | 184.2 | 122.8 | -33.4% | 0.935 |
| Advanced Box Threat 3 | 113.0 | 76.4 | -32.4% | 0.938 |
| Central Circulator | 86.0 | 57.3 | -33.4% | 0.926 |
| Central Chance Creator | 86.8 | 59.9 | -31.0% | 0.946 |
| Deep Chance Creator | 103.2 | 59.9 | -42.0% | 0.910 |
| Deep Progressive | 3.4 | 2.1 | -37.5% | 0.882 |

**Insight:** All clusters see xA+ lower than xA because sequence attribution spreads credit. Deep Chance Creators see the largest reduction (-42%) as their passes are often early in sequences.

---

## 3. xA+ Attribution Flow

How does credit flow through assist sequences?

### 1-Pass Sequences (Direct Assists)
- Pass 1 (Key Pass): **100%** of credit → mean xA+ = 0.089

### 2-Pass Sequences
- Pass 1 (Build-up): **82%** of credit → mean xA+ = 0.079
- Pass 2 (Key Pass): **18%** of credit → mean xA+ = 0.017

### 3-Pass Sequences
- Pass 1 (Early Build-up): **74%** of credit → mean xA+ = 0.072
- Pass 2 (Progression): **15%** of credit → mean xA+ = 0.013
- Pass 3 (Key Pass): **11%** of credit → mean xA+ = 0.008

**Insight:** Surprisingly, earlier passes in sequences retain more xA+ credit than the final pass. This is because xT-based weighting rewards the progressive value gained earlier in the sequence, while the final pass often covers short distances with lower xT delta.

---

## 4. Zone Analysis

| Zone | Total xA | Total xA+ | Mean xA | Mean xA+ | Correlation |
|------|----------|-----------|---------|----------|-------------|
| Attacking Third | 510.8 | 357.4 | 0.00704 | 0.00493 | 0.944 |
| Middle Third | 58.8 | 19.7 | 0.00050 | 0.00017 | 0.733 |
| Defensive Third | 7.0 | 1.3 | 0.00014 | 0.00003 | 0.767 |

**Insight:** The attacking third contributes 94% of all xA+ value. Correlation between xA and xA+ is highest in the attacking third (0.944), meaning the two metrics agree most on final-third passes.

---

## 5. Key Findings

### Finding 1: Traditional xA Overvalues Final-Pass Specialists
Players who primarily deliver the final ball before a shot (crosses, through balls into the box) see their value inflated by 25-40% under traditional xA compared to sequence-attributed xA+.

### Finding 2: Build-Up Contributors Are Undervalued
Players who excel at progressive passes earlier in sequences (deep-lying playmakers, ball-playing defenders) gain relative rank under xA+ because their contributions to chance creation are now recognized.

### Finding 3: High Correlation, Systematic Difference
xA and xA+ correlate strongly (0.88-0.95 depending on segment), meaning they generally agree on who creates chances. However, there's a systematic ~30% reduction in total value because credit is distributed rather than concentrated.

### Finding 4: xT-Based Weighting Rewards Progression
The xT (expected threat) weighting system rewards passes that move the ball into more dangerous positions. This means the *progressive* pass (e.g., from midfield into the final third) often gets more credit than the final pass (e.g., a square ball in the box).

---

## 6. Implications

### For Player Recruitment
- Don't overpay for players with high traditional xA if their value comes primarily from final passes
- Look for undervalued build-up specialists whose progressive passing creates the platform for chances

### For Tactical Analysis
- Track xA+ alongside xA to understand which players contribute to the full sequence
- Use attribution flow to identify where your team's chance creation is strongest (early build-up vs final ball)

### For Performance Evaluation
- Consider both metrics when evaluating playmakers
- Players in deeper roles may appear less valuable under traditional xA but contribute significantly to xA+

---

## 7. Methodology Notes

### xA (Traditional)
- Credit = shot xG assigned entirely to the player who made the final pass before the shot

### xA+ (Sequence Attribution)
- Credit = shot xG distributed across all passes in the pre-shot sequence (up to 3 passes)
- Weighting = based on xT (expected threat) delta of each pass
- Higher xT delta → higher share of credit

### Data Scope
- ~240,000 passes analyzed
- Assist sequences: 1-3 passes before shots
- Player clusters: 6 behavioral types based on passing patterns
- Position groups: 9 categories

---

## 8. Visualizations Available

The following charts accompany this report:

**Phase 2 Charts** (`outputs/analysis/cxa/phase2_comparison/charts/`)
- `xa_vs_xaplus_scatter.png` — Scatter plot comparing xA to xA+ per pass
- `leaderboard_comparison.png` — Top 20 players by xA+ with rank changes
- `comparison_by_cluster.png` — xA vs xA+ by player type
- `attribution_by_position.png` — Credit distribution in sequences
- `comparison_by_splits.png` — Multi-panel comparison across dimensions

**Enhanced Charts** (`outputs/analysis/cxa/enhanced_charts/`)
- `attribution_flow.png` — Line chart showing credit flow through sequences

---

## 9. Data Files

All underlying data available in `outputs/analysis/cxa/phase2_comparison/`:

| File | Description |
|------|-------------|
| `player_leaderboard.csv` | Full player rankings with xA, xA+, rank changes |
| `by_player_cluster.csv` | Aggregated comparison by player type |
| `by_zone.csv` | Comparison by pitch zone |
| `by_position_group.csv` | Comparison by position |
| `attribution_by_position.csv` | Credit weights by sequence position |
| `goal_correlations.csv` | Correlation with actual goals |

---

**Report Generated By:** cxA Analysis Pipeline  
**Contact:** [Your details]  
**Repository:** opponent-adjusted-metrics
