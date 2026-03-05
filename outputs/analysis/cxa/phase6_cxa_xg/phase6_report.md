# Phase 6: cXA-xG Analysis Report

Generated: 2026-02-02 18:37

## Executive Summary

This analysis introduces **cXA-xG** (Created Expected Goals), a new metric that
attributes chance creation credit to all pre-shot actions, weighted by the
expected goals value of each shot.

**Key difference from prior xA+ metrics:**
- Trained on ALL shots (not just goals)
- Weighted by shot xG (stable, less noisy)
- Conservative: sum of credits = sum of shot xG

## Data Overview

| Metric | Value |
|--------|-------|
| Total shots | 5,606 |
| Total goals | 507 |
| Total xG | 519.44 |
| Shots with ≥1 action | 4,910 (87.6%) |
| Mean actions per shot | 2.92 |

## Credit by Action Type

| Type | cXA-xG | % of Total |
|------|--------|------------|
| Pass | 273.54 | 60.4% |
| Carry | 178.83 | 39.5% |
| Dribble | 0.35 | 0.1% |

**Key finding:** Passes account for 60% of creation credit,
with carries (40%) and dribbles (0%) making up the rest.

## Credit by Position in Sequence

Position 1 = action closest to the shot (typically the "assist").

|   action_position |   total_cxa_xg |   mean_share |   num_actions |   pct_of_total |
|------------------:|---------------:|-------------:|--------------:|---------------:|
|                 1 |      385.579   |    0.827316  |          4910 |      85.1687   |
|                 2 |       45.9169  |    0.137859  |          3571 |      10.1424   |
|                 3 |       12.8758  |    0.0721993 |          3069 |       2.84407  |
|                 4 |        5.60807 |    0.0312017 |          2589 |       1.23874  |
|                 5 |        2.74397 |    0.0238993 |          2227 |       0.606102 |

**Interpretation:** The final action before the shot receives the most credit
(position 1), but substantial credit flows to earlier actions in the buildup.

## Top 10 cXA-xG Creators

Players ranked by total xG-weighted creation credit:

|   player_id | player_name                   |   cXA_xG |   num_actions |   num_shots |
|------------:|:------------------------------|---------:|--------------:|------------:|
|         516 | Kevin De Bruyne               |  5.17588 |           144 |          94 |
|         236 | Kylian Mbappé Lottin          |  4.89319 |           169 |         102 |
|         226 | Antoine Griezmann             |  4.87206 |           124 |          90 |
|         146 | Memphis Depay                 |  4.71684 |            87 |          59 |
|          83 | Neymar da Silva Santos Junior |  4.43641 |           104 |          63 |

## Top 10 cXA-Goals Creators

Players ranked by goal-weighted creation credit (comparable to traditional assists):

|   player_id | player_name          |   cXA_goals |   num_actions |   num_shots |
|------------:|:---------------------|------------:|--------------:|------------:|
|         236 | Kylian Mbappé Lottin |     7.58397 |           169 |         102 |
|         226 | Antoine Griezmann    |     6.23781 |           124 |          90 |
|          11 | Xherdan Shaqiri      |     6.21535 |            73 |          53 |
|         149 | Cody Mathès Gakpo    |     5.82134 |            74 |          45 |
|         361 | Ivan Perišić         |     5.23053 |            76 |          57 |

## Calibration Check

The sum of attributed credit should equal the total xG (for cXA-xG) and
total goals (for cXA-Goals):

| metric                         |   expected |   attributed |        diff |
|:-------------------------------|-----------:|-------------:|------------:|
| Total xG (shots w/ actions)    |    452.724 |      452.724 | 5.68434e-14 |
| Total Goals (shots w/ actions) |    439     |      439     | 0           |

Note: Any difference is due to shots with no preceding actions in the window.

## Methodology

### Window Definition
- Last 8 actions (Pass, Carry, Dribble)
- Within 15 seconds before the shot
- Same possession

### Scorer
- Logistic regression predicting `is_final_action_before_shot`
- Features: end_x, end_y, distance_to_goal, angle_to_goal, action type flags,
  under_pressure, seconds_to_shot, is_into_box

### Credit Allocation
- Softmax over log-odds scores within each shot's action window
- Multiply by shot xG (cXA-xG) or 1.0 for goals (cXA-Goals)

## Files Generated

- `cxa_xg_credits.csv`: Per-action credits (xG-weighted)
- `cxa_goals_credits.csv`: Per-action credits (goals only)
- `player_leaderboard_xg.csv`: Player totals for cXA-xG
- `player_leaderboard_goals.csv`: Player totals for cXA-Goals
- `action_type_summary.csv`: Credit by Pass/Carry/Dribble
- `credit_by_position.csv`: Credit by action position
- `calibration_check.csv`: Conservation checks
