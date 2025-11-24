# Methodology: Data Analysis and Feature Engineering

**Version:** 1.1.0  
**Module:** Analytics & Features  
**Technical Stack:** Pandas, NumPy, Scikit-Learn

---

## 1. Abstract

Following the ingestion of raw event data, the Data Analysis module is responsible for transforming atomic events into meaningful, predictive signals. This phase bridges the gap between raw telemetry (e.g., "Player X was at coordinate Y") and football intelligence (e.g., "The shooter was under high pressure in a transition phase"). This document outlines the mathematical definitions, heuristic algorithms, and feature engineering pipelines used to construct the analytical dataset, supported by empirical analysis of the PL 15/16 dataset.

## 2. Feature Engineering Framework

The feature engineering process is encapsulated in `scripts/build_shot_features.py`. We categorize features into three distinct domains: **Geometry**, **Context**, and **State**.

### 2.1 Geometric Features

The most fundamental predictors of goal probability are spatial. We define the goal mouth as a line segment on the goal line (x=120) spanning from y=36 to y=44 (8 yards wide).

*   **Shot Distance ($d$):** Calculated as the Euclidean distance from the shot location $(x, y)$ to the center of the goal $(120, 40)$.
    $$ d = \sqrt{(120 - x)^2 + (40 - y)^2} $$
*   **Shot Angle ($\theta$):** The visible angle of the goal mouth from the shooter's perspective. This is calculated using the law of cosines or the `arctan` difference between the two goal posts.
    $$ \theta = \arctan\left(\frac{y - 36}{120 - x}\right) - \arctan\left(\frac{y - 44}{120 - x}\right) $$
    *Note: We take the absolute value and normalize to degrees.*

#### Empirical Analysis: The "Six-Yard" Cliff
Our analysis confirms the expected non-linear decay of goal probability as distance increases. Goal probability drops precipitously outside the 6-yard box.

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

### 2.2 Contextual Features: Pressure and Density

A key innovation of this project is the rigorous quantification of defensive pressure using "Freeze Frame" data.

*   **Defender Density:** We define a "Shot Cone" as the triangular region formed by the ball and the two goal posts. We count the number of opponents within this cone and within a 5-yard radius of the shooter.
*   **Pressure State:** StatsBomb provides a binary `under_pressure` flag. We augment this by calculating a continuous `pressure_intensity` score based on the inverse distance to the nearest defender ($d_{nearest}$):
    $$ I_{pressure} = \frac{1}{1 + d_{nearest}} $$

#### Empirical Analysis: The Cost of Pressure
Shots taken under "Pressure" have a conversion rate of **10.2%**, compared to **25.2%** for shots with "No immediate defensive trigger". Furthermore, when a defender is close enough to register a "Block" event, the conversion rate plummets to **0.3%**.

**Table 2: Impact of Defensive State**

| Defensive Label | Shots | Goals | Mean xG | Goal Rate | Lift vs xG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No Trigger** | 754 | 190 | 0.280 | **25.2%** | -2.8% |
| **Ball Recovery** | 472 | 80 | 0.147 | **16.9%** | +2.3% |
| **Pressure** | 964 | 98 | 0.099 | **10.2%** | +0.2% |
| **Block** | 1164 | 3 | 0.060 | **0.3%** | -5.8% |

*Reference Plot:* `outputs/analysis/cxg/plots/defensive_overlay_goal_rates.png`

### 2.3 Game State Features

Football is path-dependent; the current score influences tactical behavior.

*   **Score Differential:** Defined from the perspective of the shooting team:
    $$ \Delta S = \text{Goals}_{\text{For}} - \text{Goals}_{\text{Against}} $$
*   **Gamestate Flags:**
    -   `is_leading`: $\Delta S > 0$
    -   `is_drawing`: $\Delta S = 0$
    -   `is_trailing`: $\Delta S < 0$

#### Empirical Analysis: Composure vs. Desperation
Teams "Leading by 1" convert **Carry + Ground Pass** shots at **8.3%**, whereas teams "Trailing by 1" convert the same shots at **4.7%**. This suggests a "Composure Bonus" for leading teams or a "Desperation Penalty" for trailing teams.

*Reference Plot:* `outputs/analysis/cxg/plots/game_state_score_goal_rates.png`

## 3. Advanced Analysis: The Pass Value Chain

To understand the quality of the *opportunity* (not just the shot), we analyze the sequence leading up to it. This is implemented in `src/opponent_adjusted/analysis/cxg_analysis/pass_value_chain.py`.

### 3.1 Assist Quality
Not all assists are equal. A through-ball that splits the defense creates a higher-xG chance than a lateral pass outside the box.
*   **Pass Type Classification:** We categorize the penultimate event (the assist) into:
    -   **Cross:** Pass from the wide channels into the box.
    -   **Cutback:** Pass from the byline backwards into the box (historically high conversion).
    -   **Through Ball:** Pass originating from central areas that penetrates the defensive line.

#### Empirical Analysis: The "Through Ball" Premium
Through balls are the most dangerous assist type, generating shots with an average xG of **0.227** and a conversion rate of **29.7%**. Cutbacks also significantly outperform standard crosses (21.4% vs 14.8%).

**Table 3: Goal Rate by Assist Type**

| Assist Category | Shots | Goals | Mean xG | Goal Rate | Avg Distance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Through Ball** | 64 | 19 | 0.227 | **29.7%** | 14.1y |
| **Unassisted** | 672 | 200 | 0.324 | **29.8%** | 17.5y |
| **Cutback** | 42 | 9 | 0.173 | **21.4%** | 12.6y |
| **Cross** | 324 | 48 | 0.144 | **14.8%** | 11.5y |
| **Ground Pass** | 1086 | 64 | 0.064 | **5.9%** | 22.9y |

*Reference Plot:* `outputs/analysis/cxg/plots/assist_context_goal_rates.png`

### 3.2 Sequence Analysis
We analyze the two-event sequence leading to a shot (e.g., `Carry -> Through Ball -> Shot`).

#### Empirical Analysis: Dynamic Playmakers
The combination of a **Carry + Through Ball** is the most lethal sequence in the dataset, with a massive **+10.7%** lift over expected conversion. Conversely, **Carry + Switch** sequences show a negative lift (**-3.5%**), suggesting that allowing the defense to shift reduces shot quality.

**Table 4: Sequence Value Lift**

| Chain Label | Shots | Mean xG | Goal Rate | Lift |
| :--- | :--- | :--- | :--- | :--- |
| **Carry + Through Ball** | 72 | 0.213 | **31.9%** | **+10.7%** |
| **Direct + Through Ball** | 78 | 0.214 | **28.2%** | **+6.8%** |
| **Direct + Cross** | 333 | 0.147 | **15.3%** | +0.6% |
| **Carry + Ground Pass** | 1203 | 0.065 | **6.2%** | -0.2% |
| **Carry + Switch** | 155 | 0.067 | **3.2%** | **-3.5%** |

*Reference Plot:* `outputs/analysis/cxg/plots/pass_value_chain_lift_scatter.png`

## 4. Team Style Profiling

To support the "Neutral Priors" modeling approach, we must profile teams without using their names. We derive a "Style Vector" for every team in every match.

### 4.1 Style Metrics
We construct a 5-dimensional "Style Vector" for every team in every match to capture their tactical identity. These features are normalized before clustering.

1.  **Possession Share ($P_{share}$):**
    $$ P_{share} = \frac{\text{Team Possession Duration}}{\text{Total Match Duration}} $$
    *Proxy for dominance and ball control.*
2.  **Press Intensity ($I_{press}$):**
    $$ I_{press} = \frac{\text{Count(Pressure Events)}}{\text{Match Minutes}} $$
    *Proxy for defensive aggression and work rate.*
3.  **Defensive Line Height ($H_{def}$):** The average X-coordinate of all team events, normalized to $[0, 1]$.
    $$ H_{def} = \frac{1}{N} \sum \frac{x_i}{120} $$
    *Indicates whether a team plays a High Line or a Low Block.*
4.  **Press Height ($H_{press}$):** The average X-coordinate specifically of `Pressure` events.
    *Distinguishes between a "High Press" (pressing in the opponent's third) and a "Mid Block" (pressing in the middle third).*
5.  **Average Shot Distance ($d_{avg}$):** The mean distance of shots taken by the team.
    *Distinguishes between teams that work the ball into the box (e.g., Arsenal) and teams that rely on long shots.*

### 4.2 Clustering Analysis
We apply **K-Means Clustering** ($k=6$) to these style vectors to identify tactical archetypes. This allows the model to learn interaction effects (e.g., "Counter-Attackers" scoring against "Dominant Pressers") without knowing the specific teams involved.

#### Cluster Interpretations (Archetypes)
Based on the feature centroids, we identify the following archetypes:
*   **Cluster 0: "Passive Low Block"** (Low $P_{share}$, Low $H_{def}$, Low $I_{press}$). Teams that sit deep and absorb pressure.
*   **Cluster 1: "Dominant Pressers"** (High $P_{share}$, High $H_{press}$, High $I_{press}$). Elite teams that control the game and win the ball back high up the pitch (e.g., Man City, Bayern).
*   **Cluster 2: "Direct Counter-Attackers"** (Low $P_{share}$, Low $H_{def}$, High $d_{avg}$). Teams that defend deep but transition quickly, often settling for longer range shots.
*   **Cluster 3: "Mid-Block Possession"** (Medium $P_{share}$, Medium $H_{def}$). Balanced teams that control possession but do not press aggressively.
*   **Cluster 4: "Aggressive Underdogs"** (Low $P_{share}$, High $I_{press}$). Teams that lack quality on the ball but compensate with extreme physical effort.
*   **Cluster 5: "Box Crashers"** (High $P_{share}$, Low $d_{avg}$). Teams that dominate territory and refuse to shoot from distance.

## 5. Conclusion

The Data Analysis module transforms raw coordinates into a rich, semantic representation of the game. By engineering features that capture geometry, pressure, game state, and team style, and validating them against empirical data, we provide the downstream models with the necessary context to distinguish between a "statistically unlikely" goal and a "tactically created" high-quality chance.
