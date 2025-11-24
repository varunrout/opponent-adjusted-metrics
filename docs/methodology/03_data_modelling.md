# Methodology: Data Modelling and Opponent Adjustment

**Version:** 1.1.0  
**Module:** Modeling & Inference  
**Technical Stack:** Scikit-Learn, Statsmodels, Logistic Regression

---

## 1. Abstract

This document details the core statistical engine of the project: the Opponent-Adjusted xG Model. Unlike traditional Expected Goals models that treat all defensive opposition as a uniform "average," our approach explicitly quantifies the defensive strength of the opponent and the finishing skill of the attacker. We employ a **Neutral Priors** strategy combined with **Residual Analysis** to decouple the quality of the chance from the quality of the teams involved.

## 2. The "Neutral Priors" Philosophy

The central hypothesis of this project is that a robust predictive model should first understand the *intrinsic difficulty* of a shot before adjusting for team quality.

### 2.1 The Problem with Standard Models
Standard xG models are often trained on datasets dominated by elite teams. If a model sees Lionel Messi scoring from 25 yards repeatedly, it might learn that "25-yard shots are high probability," rather than "Messi is an outlier." This biases the model against average players.

### 2.2 The Solution: Masking and Clustering
To create a "Neutral" baseline, we employ two strategies during training:
1.  **ID Exclusion:** We explicitly exclude `team_id` and `player_id` from the feature set of the base model.
2.  **Style Substitution:** Instead of team names, we use the "Style Clusters" defined in the Analysis phase (e.g., "High Pressing Opponent"). This allows the model to learn that shots against a "Low Block" are harder to convert due to density, without overfitting to specific teams like Burnley or Atletico Madrid.

## 3. Model Architecture: Stacking Ensembles

We do not rely on a single algorithm. Instead, we use a **Stacked Generalization** (Stacking) approach to capture different aspects of the game.

### 3.1 Level 0: The Submodels
We train three distinct "weak learners" (all Logistic Regressions) on different feature subsets:
*   **`GeomModel`:** Trained *only* on Distance and Angle. This captures the pure physics of the shot.
*   **`ContextModel`:** Trained on Pressure, Body Part, and Pass Type. This captures the tactical context.
*   **`StateModel`:** Trained on Score Differential and Time. This captures the psychological state.

### 3.2 Level 1: The Meta-Learner
The predictions (probabilities) from the Level 0 models are fed as input features into a Level 1 Logistic Regression.
$$ P(\text{Goal}) = \sigma(\beta_0 + \beta_1 \cdot \hat{y}_{geom} + \beta_2 \cdot \hat{y}_{context} + \beta_3 \cdot \hat{y}_{state}) $$
This allows the model to weigh the components dynamically. For example, if the `GeomModel` says "easy goal" (close range) but `ContextModel` says "impossible" (3 defenders blocking), the Meta-Learner learns to trust the Context model.

## 4. Opponent Adjustment Mechanism

Once the Neutral Base Model ($M_{neutral}$) is trained, we calculate the "Opponent Adjustment" factors.

### 4.1 Residual Calculation
For every shot $i$ in the training set, we calculate the residual:
$$ r_i = y_i - \hat{p}_{neutral, i} $$
Where $y_i$ is the actual outcome (1 for Goal, 0 for Miss) and $\hat{p}_{neutral}$ is the probability predicted by the base model.

### 4.2 Defensive Strength Coefficient ($D_j$)
For each defensive team $j$, we calculate the average residual of all shots conceded by them:
$$ D_j = \frac{1}{N_j} \sum_{i \in \text{ShotsConceded}_j} r_i $$
*   **Interpretation:**
    *   $D_j < 0$: The team concedes *fewer* goals than the model expects. They have a **Strong Defense** (or a great Goalkeeper).
    *   $D_j > 0$: The team concedes *more* goals than expected. They have a **Weak Defense**.

### 4.3 Attacking Skill Coefficient ($A_k$)
Similarly, for each attacking team $k$:
$$ A_k = \frac{1}{N_k} \sum_{i \in \text{ShotsTaken}_k} r_i $$
*   **Interpretation:**
    *   $A_k > 0$: The team scores *more* than expected. They are **Clinical Finishers**.
    *   $A_k < 0$: The team scores *less* than expected. They are **Wasteful**.

### 4.4 The Final Prediction Equation
For a new match between Attacking Team $A$ and Defending Team $D$, the final adjusted probability for a shot is:
$$ \text{xG}_{adj} = \text{clip}(\text{xG}_{neutral} + A_{coeff} + D_{coeff}, 0, 1) $$
*Note: We clip values to ensure valid probabilities $[0, 1]$.*

## 5. Training and Validation Strategy

To ensure the model generalizes to unseen matches, we use a rigorous validation protocol.

### 5.1 GroupKFold Cross-Validation
We cannot use standard random splitting because shots from the same match are highly correlated. We use `GroupKFold` with `match_id` as the group. This ensures that if a match is in the validation set, *none* of its shots are in the training set.

### 5.2 Evaluation Metrics
We evaluate the model on two levels:
1.  **Shot Level:**
    *   **Log Loss:** The primary optimization metric.
    *   **Brier Score:** Measures calibration (mean squared error of probabilities).
    *   **AUC-ROC:** Measures discrimination (ability to rank goals higher than misses).
2.  **Match Level:**
    *   **MAE (Mean Absolute Error):** The average difference between predicted xG totals and actual goals per match.
    *   **RMSE (Root Mean Squared Error):** Penalizes large misses more heavily.

#### Empirical Results: Model Comparison
We compared the performance of the "Neutral Priors" model against a standard baseline.
*   **AUC-ROC:** The Neutral Priors model achieves a comparable AUC to the baseline, indicating that removing team IDs does not significantly degrade the ability to rank chances.
    *   *Reference Plot:* `outputs/modeling/cxg/modeling_charts/model_compare_auc_mean.png`
*   **Brier Score:** The Neutral Priors model shows excellent calibration, minimizing the mean squared error of predictions.
    *   *Reference Plot:* `outputs/modeling/cxg/modeling_charts/model_compare_brier_mean.png`

### 5.3 Calibration Analysis
We plot Calibration Curves (Reliability Diagrams). A perfectly calibrated model lies on the $y=x$ diagonal. If our curve is S-shaped, it indicates under-confidence; if it is inverted S-shaped, it indicates over-confidence.

#### Empirical Results: Reliability Diagram
The calibration plot for the Neutral Priors model shows strong alignment with the diagonal, particularly in the high-probability range (>0.3 xG), which is critical for accurately valuing "Big Chances".

*Reference Plot:* `outputs/modeling/cxg/plots/contextual_model_reliability_neutral_priors_refresh.png`

## 6. Conclusion

The Data Modelling module moves beyond simple regression. By stacking submodels, we capture complex non-linearities. By analyzing residuals, we extract the latent "skill" parameters of teams. This results in a hybrid model that respects the physics of the game while acknowledging that playing against a world-class defense is fundamentally different from playing against a relegation candidate.
