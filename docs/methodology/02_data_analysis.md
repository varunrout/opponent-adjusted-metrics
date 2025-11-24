# Methodology: Data Analysis and Feature Engineering

**Version:** 1.0.0  
**Module:** Analytics & Features  
**Technical Stack:** Pandas, NumPy, Scikit-Learn

---

## 1. Abstract

Following the ingestion of raw event data, the Data Analysis module is responsible for transforming atomic events into meaningful, predictive signals. This phase bridges the gap between raw telemetry (e.g., "Player X was at coordinate Y") and football intelligence (e.g., "The shooter was under high pressure in a transition phase"). This document outlines the mathematical definitions, heuristic algorithms, and feature engineering pipelines used to construct the analytical dataset.

## 2. Feature Engineering Framework

The feature engineering process is encapsulated in `scripts/build_shot_features.py`. We categorize features into three distinct domains: **Geometry**, **Context**, and **State**.

### 2.1 Geometric Features

The most fundamental predictors of goal probability are spatial. We define the goal mouth as a line segment on the goal line (x=120) spanning from y=36 to y=44 (8 yards wide).

*   **Shot Distance ($d$):** Calculated as the Euclidean distance from the shot location $(x, y)$ to the center of the goal $(120, 40)$.
    $$ d = \sqrt{(120 - x)^2 + (40 - y)^2} $$
*   **Shot Angle ($\theta$):** The visible angle of the goal mouth from the shooter's perspective. This is calculated using the law of cosines or the `arctan` difference between the two goal posts.
    $$ \theta = \arctan\left(\frac{y - 36}{120 - x}\right) - \arctan\left(\frac{y - 44}{120 - x}\right) $$
    *Note: We take the absolute value and normalize to degrees.*
*   **Spatial Binning:** To capture non-linear effects (e.g., the sharp drop-off in conversion rate beyond 18 yards), we discretize these continuous variables into bins (e.g., `distance_bin`: "0-6y", "6-12y", "12-18y", "18+y").

### 2.2 Contextual Features: Pressure and Density

A key innovation of this project is the rigorous quantification of defensive pressure using "Freeze Frame" data.

*   **Defender Density:** We define a "Shot Cone" as the triangular region formed by the ball and the two goal posts. We count the number of opponents within this cone and within a 5-yard radius of the shooter.
*   **Pressure State:** StatsBomb provides a binary `under_pressure` flag. We augment this by calculating a continuous `pressure_intensity` score based on the inverse distance to the nearest defender ($d_{nearest}$):
    $$ I_{pressure} = \frac{1}{1 + d_{nearest}} $$
    This allows the model to distinguish between "loose marking" (2 yards away) and "tight marking" (contact).

### 2.3 Game State Features

Football is path-dependent; the current score influences tactical behavior.

*   **Score Differential:** Defined from the perspective of the shooting team:
    $$ \Delta S = \text{Goals}_{\text{For}} - \text{Goals}_{\text{Against}} $$
*   **Gamestate Flags:**
    -   `is_leading`: $\Delta S > 0$
    -   `is_drawing`: $\Delta S = 0$
    -   `is_trailing`: $\Delta S < 0$
*   **Temporal Decay:** We include `minute` as a feature. Analysis shows that conversion rates for similar shots fluctuate slightly towards the end of halves due to fatigue and tactical desperation.

## 3. Advanced Analysis: The Pass Value Chain

To understand the quality of the *opportunity* (not just the shot), we analyze the sequence leading up to it. This is implemented in `src/opponent_adjusted/analysis/cxg_analysis/pass_value_chain.py`.

### 3.1 Assist Quality
Not all assists are equal. A through-ball that splits the defense creates a higher-xG chance than a lateral pass outside the box.
*   **Pass Type Classification:** We categorize the penultimate event (the assist) into:
    -   **Cross:** Pass from the wide channels into the box.
    -   **Cutback:** Pass from the byline backwards into the box (historically high conversion).
    -   **Through Ball:** Pass originating from central areas that penetrates the defensive line.
*   **Pass Value:** We assign a heuristic "Threat Score" to the assist based on its origin and destination zones. This score feeds into the `assist_quality` submodel.

### 3.2 Defensive Triggers
We analyze the 10-second window preceding the shot to identify "Defensive Triggers"—moments of disorganization.
*   **High Turnover:** A ball recovery in the attacking third.
*   **Fast Break:** A sequence where the ball travels vertically >50 yards in <10 seconds.
*   **Set Piece Aftermath:** Shots occurring within 5 seconds of a corner or free kick clearance (often chaotic).

## 4. Team Style Profiling

To support the "Neutral Priors" modeling approach, we must profile teams without using their names. We derive a "Style Vector" for every team in every match.

### 4.1 Style Metrics
1.  **Possession Share:**
    $$ P_{share} = \frac{\text{Team Duration}}{\text{Total Match Duration}} $$
2.  **Press Intensity:** Defined as the number of `Pressure` events per minute of opponent possession. This proxies the team's defensive aggression.
3.  **Defensive Line Height:** The average X-coordinate of defensive actions (tackles, interceptions). A high value (>50) indicates a High Line; a low value (<35) indicates a Low Block.
4.  **Build-up Directness:** The ratio of forward distance to total distance in pass sequences.

### 4.2 Clustering Analysis
We apply **K-Means Clustering** to these style vectors to identify archetypes.
*   **Cluster 1 (e.g., "Dominant Pressers"):** High possession, high line, high press (e.g., Man City, Bayern).
*   **Cluster 2 (e.g., "Counter-Attackers"):** Low possession, deep line, high directness (e.g., Leicester 15/16).
*   **Cluster 3 (e.g., "Passive Low Block"):** Low possession, deep line, low press.

These clusters allow the model to learn interaction effects (e.g., "Counter-Attackers" scoring against "Dominant Pressers") without knowing the specific teams involved.

## 5. Exploratory Data Analysis (EDA)

Before modeling, we validated the feature set through rigorous EDA.

### 5.1 Correlation Analysis
We examined the Pearson correlation coefficient between features to identify multicollinearity.
*   **Finding:** `shot_distance` and `shot_angle` are correlated ($\rho \approx -0.7$), but not perfectly. A shot from the wing has a tight angle but medium distance; a shot from central deep has a wide angle but long distance. Both are retained.
*   **Finding:** `pressure_state` is weakly correlated with `shot_distance`. This is intuitive; defenders close down shots in the box more aggressively than long-range efforts.

### 5.2 Distribution Checks
We verified that the distributions of key features matched football intuition:
*   **Shot Distance:** Right-skewed (Gamma distribution), peaking around 12-15 yards.
*   **Goal Rate vs. Distance:** Exponential decay.
*   **Goal Rate vs. Angle:** Sigmoidal relationship (rapid increase as angle opens up).

## 6. Conclusion

The Data Analysis module transforms raw coordinates into a rich, semantic representation of the game. By engineering features that capture geometry, pressure, game state, and team style, we provide the downstream models with the necessary context to distinguish between a "statistically unlikely" goal and a "tactically created" high-quality chance.
