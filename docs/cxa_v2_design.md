# cXA v2 Design Document

**Date:** February 2, 2026  
**Status:** Proposed  
**Author:** Auto-generated from Phase 0–5 analysis findings

---

## 1. Objective

Build a **unified Chance Creation Attribution (cXA)** system that:
- Credits both **passes** and **ball progression (carries/dribbles)** for chance creation
- Supports **all shots** (xG-weighted) as well as **goals only** (attribution)
- Serves **player profiling/scouting**, **team tactics**, and **opponent-adjusted creation** use-cases simultaneously

---

## 2. Key Findings From Phase Analysis

| Phase | Finding | Implication |
|-------|---------|-------------|
| Phase 0 | Goal populations differ: 369 (pass seq) vs 439 (action seq), overlap 360 | Must use consistent window definition |
| Phase 1 | Through-balls (5.3%) and into-box passes (1.8%) dominate assist rate | Pass type is predictive but not sufficient |
| Phase 2 | Baseline xA is stable (sum = assists) but pass-only | Good for passing value, not creation |
| Phase 4 | Carries account for ~40% of goal-creation credit | Excluding carries hides substantial creation |
| Phase 5 | Player "gainers" are carry-driven creators | Need carry features in scoring model |

---

## 3. Data Sources

| Source | Path | Description |
|--------|------|-------------|
| `shots.parquet` | `feature_store/cxa/shots.parquet` | 5,606 shots with `statsbomb_xg`, `is_goal`, `shot_id`, `possession` |
| `passes.parquet` | `feature_store/cxa/passes.parquet` | 240,105 passes with location, type flags, `possession`, timestamps |
| `possessions.parquet` | `feature_store/cxa/possessions.parquet` | Possession-level aggregates for context |
| Raw events (DB) | `events` table | Carries, dribbles, timestamps for window building |

---

## 4. Window Definition (Canonical Pre-Shot Slice)

**Rule:** For each shot, extract the **last N actions** (Pass, Carry, Dribble) in the **same possession**, up to **K seconds** before the shot.

**Recommended defaults:**
- `N = 8` (max actions)
- `K = 15` (max seconds before shot)
- Action types: `Pass`, `Carry`, `Dribble`
- Exclude: own-half actions unless they directly chain to an attacking action

This single window definition replaces the current "3-pass" and "5-action" separate builders.

---

## 5. Schema: `shot_action_windows.parquet`

### Shot-level columns (one row per shot)
| Column | Type | Description |
|--------|------|-------------|
| `shot_id` | int | Primary key linking to shots.parquet |
| `match_id` | int | Match identifier |
| `team_id` | int | Attacking team |
| `possession` | int | Possession number in match |
| `shot_minute` | int | Minute of shot |
| `shot_second` | int | Second of shot |
| `shot_x`, `shot_y` | float | Shot location |
| `statsbomb_xg` | float | StatsBomb xG (shot value for weighting) |
| `is_goal` | bool | Whether shot resulted in goal |
| `num_actions` | int | Actions in window (1–8) |

### Per-action columns (wide format, action1 = closest to shot)
For `i` in 1..8:
| Column | Type | Description |
|--------|------|-------------|
| `action{i}_type` | str | "Pass", "Carry", or "Dribble" |
| `action{i}_player_id` | int | Player who performed action |
| `action{i}_player_name` | str | Player name |
| `action{i}_start_x/y` | float | Action start location |
| `action{i}_end_x/y` | float | Action end location |
| `action{i}_distance_to_goal` | float | Distance from end to goal center |
| `action{i}_xt_delta` | float | Change in xT (if available) |
| `action{i}_is_into_box` | bool | Ends in penalty area |
| `action{i}_is_cross` | bool | Pass type: cross |
| `action{i}_is_through_ball` | bool | Pass type: through ball |
| `action{i}_under_pressure` | bool | Action under pressure |
| `action{i}_seconds_to_shot` | float | Time gap to shot |

---

## 6. Scoring Model

### Target
**Primary:** `is_assist` (action immediately preceding shot for all shots, not just goals)

This learns "what actions tend to be the final action before a shot" across all shots, giving a richer training signal than goal-only.

### Features (per action)
```
Geometry:
  end_x, end_y, distance_to_goal, angle_to_goal
  start_x, start_y (for carry length / pass length)

Progression:
  xt_delta (if available)
  is_into_box, is_progressive

Type:
  is_pass, is_carry, is_dribble
  is_cross, is_through_ball

Context:
  under_pressure
  seconds_to_shot (proximity to shot)
  action_position (1=closest, 2=second, ...)
```

### Algorithm
- Logistic Regression (interpretable, fast)
- Class weighting: `balanced` (is_assist is rare)
- Output: `P(is_final_action_before_shot | features)`

---

## 7. Credit Allocation

### Method: Softmax over log-odds
For each shot $s$, compute weights $w_{a,s}$ for actions in the window:

$$
w_{a,s} = \frac{\exp(\text{logit}(p_a) / T)}{\sum_{a' \in \text{window}} \exp(\text{logit}(p_{a'}) / T)}
$$

where $p_a$ is the scorer's predicted probability and $T$ is temperature (default 1.0).

### Two output modes

| Mode | Value per shot | Total credit | Use-case |
|------|----------------|--------------|----------|
| **cXA-xG** | $V_s = \text{statsbomb\_xg}$ | $\sum_s \text{xG}$ | Stable creation, scouting |
| **cXA-Goals** | $V_s = 1$ if goal else 0 | #goals | Attribution, storytelling |

Player $p$'s cXA:
$$
\text{cXA}(p) = \sum_{s} \sum_{a \in \text{actions}(p,s)} V_s \cdot w_{a,s}
$$

---

## 8. Outputs

### Feature store
- `feature_store/cxa/shot_action_windows.parquet` — canonical window data for all shots

### Analysis outputs (Phase 6)
```
outputs/analysis/cxa/phase6_cxa_xg/
  data/
    summary_metrics.csv          # total xG, credit by type, etc.
    credit_by_action_type.csv    # Pass/Carry/Dribble shares
    credit_by_action_position.csv
    player_leaderboard.csv       # player cXA-xG totals
    player_credit_by_type.csv    # per-player Pass vs Carry credit
  plots/
    credit_by_type.png
    top_players.png
  phase6_cxa_xg_report.md
```

---

## 9. Evaluation Checks

| Check | Method | Pass Criterion |
|-------|--------|----------------|
| **Conservation** | $\sum_{\text{actions}} \text{cXA-xG} = \sum_{\text{shots}} \text{xG}$ | Exact (within float precision) |
| **Alignment** | All shots have ≥1 action in window | >95% coverage |
| **Type balance** | Pass + Carry + Dribble shares | Should be in [50–70%, 25–45%, 0–5%] |
| **Scorer AUC** | ROC-AUC on held-out shots | >0.70 |
| **Calibration** | Reliability diagram for scorer | Close to diagonal |

---

## 10. Implementation Plan

1. **Window builder** (`src/opponent_adjusted/features/cxa/shot_action_windows.py`)
   - Query DB for events in same possession before each shot
   - Filter to Pass/Carry/Dribble, apply time/count limits
   - Write `shot_action_windows.parquet`

2. **cXA-xG scorer + allocator** (`src/opponent_adjusted/features/cxa/cxa_xg.py`)
   - `CxAScorer` class: fit on all shots, predict action scores
   - `compute_cxa_xg()`: load windows, score, allocate, return player credits

3. **Phase 6 analysis** (`src/opponent_adjusted/analysis/cxa/phase6_cxa_xg.py`)
   - Load windows + compute cXA-xG
   - Generate summaries, leaderboards, plots, markdown report

---

## 11. Migration Path

- Phase 3/4/5 remain valid for goal-only attribution comparisons
- Phase 6 (cXA-xG) becomes the recommended "stable creation" metric
- xA Baseline remains for pass-only contexts (e.g., passing networks)

---

*End of Design Document*
