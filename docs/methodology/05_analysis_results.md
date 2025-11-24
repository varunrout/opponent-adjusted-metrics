# Analysis Results: Opponent-Adjusted Metrics

**Date:** November 24, 2025
**Module:** `opponent_adjusted/analysis`
**Source Data:** StatsBomb Open Data (PL 15/16 Subset)

---

## 1. Executive Summary

This document summarizes the key findings from the exploratory data analysis (EDA) phase. These insights directly informed the feature engineering and submodel architecture of the CxG (Contextual Expected Goals) system. The analysis covers geometric fundamentals, assist context, defensive pressure, sequence value, and game state dynamics.

## 2. Geometric Analysis

The relationship between shot location and goal probability is the foundational prior for any xG model. Our analysis confirms the expected non-linear decay of goal probability as distance increases.

### Key Findings
*   **The "Six-Yard" Cliff:** Goal probability drops precipitously outside the 6-yard box. Shots taken from <5 yards convert at **51.2%**, while shots from 5-10 yards convert at **19.2%**.
*   **The "Penalty Spot" Bump:** We observe a slight stabilization in conversion rates around the 10-15 yard range (**20.9%**), likely driven by penalties and high-quality cutbacks to the penalty spot.
*   **Long Range Inefficiency:** Shots from 25-30 yards have a conversion rate of just **3.3%**, yet they account for a significant volume of attempts.

**Table 1: Goal Rate by Distance Bin**

| Distance Bin | Shots | Goals | Mean xG | Goal Rate |
| :--- | :--- | :--- | :--- | :--- |
| **[0, 5)** | 86 | 44 | 0.520 | **51.2%** |
| **[5, 10)** | 868 | 167 | 0.197 | **19.2%** |
| **[10, 15)** | 1353 | 283 | 0.230 | **20.9%** |
| **[15, 20)** | 1086 | 88 | 0.083 | **8.1%** |
| **[20, 25)** | 1033 | 40 | 0.044 | **3.9%** |
| **[25, 30)** | 866 | 29 | 0.028 | **3.3%** |

*Reference Plot:* `outputs/analysis/cxg/plots/geometry_distance_vs_goal.png`

---

## 3. Assist Context Analysis

The "Assist Context" module analyzes the event immediately preceding the shot. This analysis validates the hypothesis that *how* the ball arrives matters as much as *where* it is.

### Key Findings
*   **The "Through Ball" Premium:** Through balls are the most dangerous assist type, generating shots with an average xG of **0.227** and a conversion rate of **29.7%**.
*   **Crosses vs. Cutbacks:** Cutbacks (passes backward from the byline) significantly outperform standard crosses. Cutbacks convert at **21.4%** compared to just **14.8%** for crosses.
*   **Unassisted Shots:** High conversion rate (**29.8%**) for unassisted shots is driven by penalties and rebounds, which are often classified as unassisted in this schema.

**Table 2: Goal Rate by Assist Type**

| Assist Category | Shots | Goals | Mean xG | Goal Rate | Avg Distance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Through Ball** | 64 | 19 | 0.227 | **29.7%** | 14.1y |
| **Unassisted** | 672 | 200 | 0.324 | **29.8%** | 17.5y |
| **Cutback** | 42 | 9 | 0.173 | **21.4%** | 12.6y |
| **Cross** | 324 | 48 | 0.144 | **14.8%** | 11.5y |
| **Ground Pass** | 1086 | 64 | 0.064 | **5.9%** | 22.9y |

*Reference Plot:* `outputs/analysis/cxg/plots/assist_context_goal_rates.png`

---

## 4. Defensive Pressure (Overlay)

We quantified defensive pressure using a "Defensive Overlay" model that categorizes the defensive state at the moment of the shot.

### Key Findings
*   **The Cost of Pressure:** Shots taken under "Pressure" (as defined by StatsBomb) have a conversion rate of **10.2%**, compared to **25.2%** for shots with "No immediate defensive trigger".
*   **Blocks are Effective:** When a defender is close enough to register a "Block" event, the conversion rate plummets to **0.3%** (3 goals from 1164 shots). This justifies treating "Blocked" shots as a distinct, low-probability state in the model.
*   **Ball Recovery Bonus:** Shots following a "Ball Recovery" (high turnover) convert at **16.9%**, significantly higher than standard possession play, confirming the value of pressing.

**Table 3: Impact of Defensive State**

| Defensive Label | Shots | Goals | Mean xG | Goal Rate | Lift vs xG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No Trigger** | 754 | 190 | 0.280 | **25.2%** | -2.8% |
| **Ball Recovery** | 472 | 80 | 0.147 | **16.9%** | +2.3% |
| **Pressure** | 964 | 98 | 0.099 | **10.2%** | +0.2% |
| **Block** | 1164 | 3 | 0.060 | **0.3%** | -5.8% |

*Reference Plot:* `outputs/analysis/cxg/plots/defensive_overlay_goal_rates.png`

---

## 5. Pass Value Chain (Sequence Analysis)

The "Pass Value Chain" analyzes the two-event sequence leading to a shot (e.g., `Carry -> Through Ball -> Shot`). This captures the momentum of the attack.

### Key Findings
*   **Dynamic Playmakers:** The combination of a **Carry + Through Ball** is the most lethal sequence in the dataset, with a massive **+10.7%** lift over expected conversion. This represents a player driving at the defense before slipping a teammate in.
*   **Direct Play:** **Direct + Through Ball** sequences also show a strong positive lift (**+6.8%**).
*   **Ineffective Possession:** **Carry + Switch** sequences (slow build-up switching play) show a negative lift (**-3.5%**), suggesting that allowing the defense to shift reduces shot quality.

**Table 4: Sequence Value Lift**

| Chain Label | Shots | Mean xG | Goal Rate | Lift |
| :--- | :--- | :--- | :--- | :--- |
| **Carry + Through Ball** | 72 | 0.213 | **31.9%** | **+10.7%** |
| **Direct + Through Ball** | 78 | 0.214 | **28.2%** | **+6.8%** |
| **Direct + Cross** | 333 | 0.147 | **15.3%** | +0.6% |
| **Carry + Ground Pass** | 1203 | 0.065 | **6.2%** | -0.2% |
| **Carry + Switch** | 155 | 0.067 | **3.2%** | **-3.5%** |

*Reference Plot:* `outputs/analysis/cxg/plots/pass_value_chain_lift_scatter.png`

---

## 6. Game State Dynamics

We analyzed how the scoreline affects conversion rates, testing the hypothesis that "Game State" influences finishing (e.g., composure vs. desperation).

### Key Findings
*   **Leading vs. Trailing:** Teams "Leading by 1" convert **Carry + Ground Pass** shots at **8.3%**, whereas teams "Trailing by 1" convert the same shots at **4.7%**. This suggests a "Composure Bonus" for leading teams or a "Desperation Penalty" for trailing teams.
*   **Tactical Context:** Trailing teams take more shots, but often of lower quality or under higher pressure, which is reflected in the lower conversion rates for similar shot types.

*Reference Plot:* `outputs/analysis/cxg/plots/game_state_score_goal_rates.png`
