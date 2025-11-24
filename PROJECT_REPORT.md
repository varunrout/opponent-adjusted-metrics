# Opponent-Adjusted Metrics: Comprehensive Project Report

**Date:** November 24, 2025  
**Author:** Varun Rout  
**Repository:** `opponent-adjusted-metrics`  
**Version:** 1.0.0

---

## 1. Executive Summary

This report details the end-to-end development, implementation, and evaluation of a contextual, opponent-adjusted Expected Goals (CxG) system. The primary objective was to move beyond standard geometric xG models by incorporating rich contextual factors—defensive pressure, game state, and team style—while ensuring the model remains "neutral" to specific team identifiers. This neutrality allows the model to generalize across competitions and seasons without overfitting to historical team performance, a critical requirement for accurate opponent adjustment.

The project successfully established a robust data engineering pipeline using StatsBomb Open Data, ingesting match events into a normalized PostgreSQL schema. We developed a hierarchical modeling approach where submodels (finishing bias, concession bias, pressure effects) feed into a primary logistic regression classifier.

Key achievements include:
-   **Data Infrastructure:** A scalable SQLAlchemy/PostgreSQL architecture handling thousands of matches and millions of events.
-   **Neutral Priors:** A novel "neutralization" technique replacing explicit Team IDs with rolling performance windows, style archetypes (K-Means clustering), and player-specific lift components.
-   **Model Performance:** The enriched contextual model achieved a **ROC AUC of 0.865** and **Brier Score of 0.0631**, significantly outperforming a geometric baseline (AUC 0.739) and showing strong calibration against the provider's proprietary xG (AUC 0.799 on the same sample).
-   **Case Study Validation:** Applied to the 2015/16 Premier League season, the model correctly identified the underlying strength of title-winners Leicester City and highlighted the finishing over-performance of teams like West Ham, providing granular match-by-match insights.

---

## 2. Introduction

### 2.1 Background
Expected Goals (xG) has become the standard metric for quantifying chance quality in football. However, traditional models often rely heavily on shot location (distance and angle) and basic event qualifiers (header vs. foot). They frequently overlook the *context* of the chance:
-   Was the shooter under intense pressure?
-   Did the defensive line collapse deep, or was it a high turnover?
-   Is the team chasing a lead (game state effects)?
-   Who is the opponent, and how "soft" is their defense typically?

### 2.2 The Opponent-Adjustment Problem
Standard "opponent adjustment" often involves post-hoc mathematical adjustments to xG totals based on team strength ratings. This project takes a different approach: **Contextual Modeling**. By feeding the model features that describe the *defensive context* (e.g., "concession bias" derived from defensive form, pressure intensity), the model inherently adjusts the probability of a goal based on the difficulty of the situation, rather than just the location of the shot.

### 2.3 Project Scope
The system is built on **StatsBomb Open Data**, specifically focusing on:
-   **Training Data:** FIFA World Cup and UEFA Euro championships (high-quality, neutral ground data).
-   **Test/Validation Data:** Premier League 2015/16 season (a distinct league environment to test generalization).

The technical stack includes Python 3.12, PostgreSQL, SQLAlchemy, Scikit-Learn, and Pandas, orchestrated via a Makefile and Poetry environment.

---

## 3. Data Engineering & Architecture

### 3.1 Database Schema
The foundation of the project is a relational database designed to normalize the nested JSON structure of StatsBomb data. The schema was designed to balance query performance with data integrity, utilizing SQLAlchemy ORM models (`src/opponent_adjusted/db/models.py`) to ensure type safety.

*   **`competitions` & `matches`**: These tables store metadata about tournaments and fixtures. The `matches` table is particularly important as it serves as the primary partition key for many downstream analyses. It includes attributes like `match_date`, `kick_off`, and `stadium`, which are vital for temporal splitting in our cross-validation strategy.
*   **`teams` & `players`**: Reference tables for entities. We maintain a strict separation between entity metadata and their performance metrics.
*   **`events`**: The core table, storing every on-ball action. This table is heavily indexed on `match_id`, `team_id`, and `player_id`. It uses a JSONB column for flexible attribute storage (like `pass_end_location` or `foul_committed_card`), allowing us to evolve the schema without costly migrations.
*   **`shots`**: A specialized view/table for shot-specific attributes. This table flattens the complex `shot` dictionary found in the raw events, extracting critical features like `freeze_frame` (positions of all players at the moment of the shot), `technique`, `body_part`, and `outcome`.
*   **`possessions`**: A derived table aggregating events into continuous phases of play. This is crucial for our "Style" analysis, allowing us to calculate metrics like "Average Possession Duration" and "Passes per Possession" which feed into the K-Means clustering.

### 3.2 Ingestion Pipeline
The ingestion process (`scripts/ingest_*.py`) follows a strict ETL (Extract, Transform, Load) pattern designed for idempotency and error resilience:

1.  **Extract:** The system fetches JSON files from the StatsBomb repository. A "Discovery" module scans the directory tree to identify new or updated match files.
2.  **Transform:**
    -   **Complex Parsing:** The most challenging aspect is parsing the `shot.freeze_frame`. This is a list of dictionaries representing every player on the pitch. Our pipeline transforms this into a structured format, calculating the "Goalkeeper Location" and "Defender Density" relative to the shooter.
    -   **Coordinate Normalization:** StatsBomb uses a 120x80 yard pitch. We normalize all coordinates to this standard, ensuring that data from different sources (if added later) would align correctly.
    -   **Entity Mapping:** Categorical IDs (e.g., "Play Pattern: Regular Play") are mapped to database foreign keys to enforce referential integrity.
3.  **Load:** We use SQLAlchemy's `Session` management with bulk insert operations (`session.bulk_save_objects`) to handle thousands of events per match efficiently. The pipeline includes a "Rollback" mechanism: if any event in a match fails validation, the entire match transaction is rolled back to prevent partial/corrupt data states.

### 3.3 Feature Engineering
Raw data is transformed into modeling features via `scripts/build_shot_features.py`. This step bridges the gap between the raw database and the machine learning models. Key feature groups include:

*   **Geometry:**
    -   `shot_distance`: Euclidean distance to the center of the goal line.
    -   `shot_angle`: The visible angle of the goal mouth from the shooter's perspective.
    -   `distance_bin` / `angle_bin`: We bin these continuous variables to allow the linear model to capture non-linear effects (e.g., the sharp drop-off in probability at very tight angles).

*   **Game State:**
    -   `score_diff_at_shot`: The goal difference from the shooter's perspective. This is a proxy for "Game Script"—teams leading by 2 goals often face different defensive structures than teams chasing a draw.
    -   `is_leading`, `is_trailing`, `is_drawing`: Boolean flags derived from the score difference.
    -   `minute`: Game time is used both as a continuous variable and bucketed (e.g., "Late Game") to capture fatigue effects.

*   **Pressure & Defense:**
    -   `pressure_state`: Derived from StatsBomb's `under_pressure` attribute. We also calculate a "Defender Proximity" score using the freeze frame data, measuring the distance to the nearest opponent and the number of defenders in the "Shot Cone" (the triangle between the ball and the goal posts).
    -   `def_line_height`: The average X-coordinate of the defensive team's players. This helps distinguish between shots taken against a "Low Block" (packed defense) versus a "High Line" (potential for through balls).

---

## 4. Methodology: The Contextual Model

The core of the project is the **Contextual xG Model**. Unlike a "black box" gradient booster, we opted for a **Stacked Logistic Regression** approach. This offers interpretability and allows us to explicitly control how different signals (finishing skill, defensive weakness) enter the final probability.

### 4.1 Architecture
The model is composed of several "Submodels" that generate priors (logits), which are then used as features in the final calibration layer.

#### 4.1.1 Neutral Finishing & Concession Priors
*File: `src/opponent_adjusted/modeling/cxg/submodels/train_finishing_bias_model.py`*

This submodel is the cornerstone of our "Neutral" approach. In traditional opponent-adjusted models, one might include a `team_id` feature (one-hot encoded) to capture that "Team A is a strong defensive team." However, this fails when applying the model to a new league or season where "Team A" has changed significantly or doesn't exist in the training set.

Our solution is to replace the explicit `team_id` with a composite "Prior" derived from three neutral components:

1.  **Rolling Form Component:**
    We calculate the "Finishing Lift" (Goals vs. xG) and "Concession Lift" (Goals Conceded vs. xG Conceded) for each team over rolling windows of 3, 5, and 8 matches.
    The math for the "Lift" is based on the Log-Odds ratio:
    $$ \text{Lift} = \ln\left(\frac{\text{Goals}}{\text{Shots}}\right) - \ln\left(\frac{\text{xG}}{\text{Shots}}\right) $$
    This effectively measures how much a team is over- or under-performing the baseline expectation. We weight these windows (giving more weight to recent form) to produce a single `rolling_component`.

2.  **Style Archetype Component:**
    Teams are clustered into 6 distinct "Styles" using K-Means clustering. The features for clustering include:
    -   `possession_share`: Do they dominate the ball?
    -   `press_intensity`: Pressure events per minute.
    -   `avg_shot_distance`: Do they shoot from deep or work it into the box?
    -   `def_line_height`: How high do they defend?
    
    Once clustered, we calculate the aggregate Finishing and Concession Bias for *all teams in that cluster*. If "Cluster 1" (e.g., High-Pressing Dominant Teams) tends to score 10% more than expected, every team in that cluster inherits a positive bias. This allows the model to understand that "Teams playing *like this* usually score more," without knowing the team's name.

3.  **Player Component:**
    Finally, we account for individual brilliance. For every match, we identify the top 3 shooters (by volume) in the lineup. We calculate their personal rolling finishing lift over a 35-match window. The average lift of these top 3 players forms the `player_component`. This captures the "Messi Effect"—a team might be average, but if they field a world-class finisher, their probability of scoring increases.

**Synthesis:**
These three components are combined via a weighted sum to produce the final `finishing_bias_logit` and `concession_bias_logit`.
$$ \text{Bias Logit} = w_1 \cdot \text{Rolling} + w_2 \cdot \text{Style} + w_3 \cdot \text{Player} $$
These logits are then fed as features into the main Contextual Model.

#### 4.1.2 Other Submodels
-   **Assist Quality:** This submodel estimates the probability of a goal based *solely* on the pass that created the shot. It considers the `pass_type` (Cross, Cutback, Through Ball) and the "Pass Value" (a metric derived from our `pass_value_chain.py` analysis). This helps the model distinguish between a "lucky" long shot and a tap-in created by a brilliant play.
-   **Pressure Model:** Estimates the difficulty of the shot based on defender proximity. It outputs a `pressure_logit` that quantifies how much the defensive pressure reduces the expected conversion rate.
-   **Defensive Trigger:** Analyzes the 10 seconds leading up to the shot. Was it a "High Turnover"? A "Fast Break"? This submodel captures the disorganization of the defense during transition moments.

### 4.2 The Enriched Dataset
The `enrich_cxg_with_submodels.py` script performs the critical task of merging the "Neutral Priors" and other submodel outputs onto the shot-level dataset. This process effectively "stacks" the submodels into the final feature set.

**Technique: Logit Stacking**
Instead of feeding raw probabilities (0 to 1) from submodels into the final model, we convert them to **Logits** (Log-Odds).
$$ \text{Logit}(p) = \ln\left(\frac{p}{1-p}\right) $$
This is crucial because the final model is a Logistic Regression, which operates linearly in log-odds space. By providing logits, we allow the final model to simply learn a coefficient (weight) for each submodel. If the coefficient is 1.0, the submodel is trusted perfectly. If it's 0.5, the submodel is dampened.

**Merge Logic:**
-   **Finishing/Concession Priors:** Merged on `match_id` and `team_id`.
-   **Pressure/Defensive Triggers:** Merged on specific buckets (e.g., `pressure_bucket`) derived from the raw features.
-   **Missing Data Handling:** If a prior is missing (e.g., a new team with no history), we impute a **Neutral Logit of 0.0** (implying no deviation from the average) and a **Reliability Score of 0.0**. This ensures the model falls back to the baseline geometric probability when context is unavailable.

### 4.3 Training Strategy
*File: `src/opponent_adjusted/modeling/cxg/contextual_model.py`*

We employ a robust Scikit-Learn pipeline to ensure reproducibility and prevent data leakage.

**Algorithm:** Logistic Regression
-   **Solver:** `lbfgs` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno). We chose this optimizer for its efficiency on datasets of this size (~20k rows) and its ability to handle L2 regularization.
-   **Regularization:** L2 (Ridge) with `C=0.5`. This prevents overfitting, especially given the high correlation between some features (e.g., `shot_distance` and `distance_bin`).

**Preprocessing Pipeline:**
We use a `ColumnTransformer` to apply specific transformations to different feature types:
-   **Numeric Features:** `StandardScaler` (Z-score normalization). This is essential for Logistic Regression so that coefficients are comparable and the optimizer converges quickly.
-   **Categorical Features:** `OneHotEncoder` (with `handle_unknown='ignore'`). This converts features like `body_part` (Head, Foot) into binary columns.
-   **Binary Features:** Passed through unchanged (or imputed with mode).

**Validation Strategy: GroupKFold**
Standard K-Fold cross-validation is dangerous in football data because shots from the same match are highly correlated (same weather, same pitch, same defensive form). Random splitting could leak information.
-   **Technique:** We use `GroupKFold` with `groups=match_id`.
-   **Effect:** This ensures that all shots from Match X are either *entirely* in the training set or *entirely* in the validation set. The model is tested on matches it has never seen before, simulating the real-world prediction task.

---

## 5. Model Evaluation

We evaluated three primary model configurations on the training set (World Cup + Euros):
1.  **Baseline Geometry:** Only distance and angle.
2.  **Contextual (Filtered):** Contextual features but without the rich submodel priors.
3.  **Contextual (Enriched):** The full model with neutral priors.

### 5.1 Aggregate Metrics
We utilize three primary metrics to quantify model performance, each capturing a different aspect of quality:

1.  **ROC AUC (Receiver Operating Characteristic Area Under Curve):**
    -   **Value:** 0.865 (Enriched Model)
    -   **Definition:** The probability that the model ranks a randomly chosen "Goal" higher than a randomly chosen "No Goal".
    -   **Significance:** This measures **Discrimination**. A high AUC means the model is excellent at sorting chances from best to worst, regardless of the absolute probability values.

2.  **Brier Score:**
    -   **Value:** 0.0631 (Enriched Model)
    -   **Definition:** The mean squared difference between the predicted probability ($p$) and the actual outcome ($y \in \{0,1\}$):
        $$ \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2 $$
    -   **Significance:** This measures **Calibration** and **Refinement**. Unlike AUC, Brier Score penalizes a model for being confident and wrong. It is the most "honest" metric for probabilistic forecasts. The improvement from 0.0760 (Baseline) to 0.0631 is substantial in the context of rare-event modeling.

3.  **Log Loss (Cross-Entropy):**
    -   **Value:** 0.2188 (Enriched Model)
    -   **Definition:** Measures the uncertainty of the probabilities.
        $$ \text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \ln(p_i) + (1-y_i) \ln(1-p_i)] $$
    -   **Significance:** Heavily penalizes being "surprised" (e.g., assigning 0.01 probability to a shot that becomes a goal).

### 5.2 Reliability and Calibration
*Reference Chart: `outputs/modeling/cxg/modeling_charts/cxg_reliability_overlay.png`*

A model is "calibrated" if events predicted with 30% probability actually happen 30% of the time. We assess this using a **Reliability Diagram**.

**Method:**
1.  **Binning:** We divide the predictions into 10 bins (0-10%, 10-20%, ..., 90-100%).
2.  **Calculation:** For each bin, we calculate:
    -   Mean Predicted Probability ($\bar{p}$)
    -   Observed Event Frequency ($\bar{y}$)
3.  **Plotting:** We plot $\bar{y}$ vs. $\bar{p}$. A perfectly calibrated model lies on the $y=x$ diagonal.

**Result:** The Enriched Contextual model tracks the diagonal extremely well. Notably, in the high-probability range (0.45 - 0.85), where sample sizes are smaller and models often drift, our model remains tight to the line. This indicates that when the model says "this is a big chance," it really is.

### 5.3 Feature Importance
*Reference Data: `outputs/modeling/cxg/contextual_feature_effects_enriched.csv`*

Since we use Logistic Regression with standardized features, the coefficients ($\beta$) directly indicate feature importance in terms of **Log-Odds**.

1.  **`statsbomb_xg` (Base Probability):** The coefficient is positive and large. This is expected; the provider's geometry model is a strong baseline.
2.  **`finishing_bias_logit`:** Highly positive coefficient. This confirms that the "Neutral Prior" is working. A team with a high finishing bias (running hot) increases the log-odds of scoring.
3.  **`pressure_logit`:** Significant negative coefficient. This validates the hypothesis that defensive pressure suppresses goal probability. A shot taken with a defender 1 meter away is far less likely to go in than one with 5 meters of space, even if the angle is identical.
4.  **`is_trailing`:** Positive coefficient. This captures the "Game State" effect. Teams chasing a lead often face defenses that are "protecting what they have," potentially sitting deeper but inviting more dangerous pressure, or taking more risks in attack that lead to chaotic rebounds.

---

## 6. Case Study: Premier League 2015/16

To validate the "Neutral Priors" approach, we applied the model—trained *only* on international tournaments—to the 2015/16 Premier League season. This is a rigorous test of generalization.

### 6.1 The "Leicester City" Test
The 2015/16 Premier League season is the ultimate stress test for any football model. Leicester City's title win is often dismissed as a "miracle" or a statistical anomaly. A robust model should be able to peer through the noise and determine if their underlying performance supported their results.

**Results (from `team_aggregates.csv`):**
-   **Leicester City:**
    -   **CxG For:** ~68.0 (Rank: 4th)
    -   **Goals For:** 68
    -   **CxG Difference (For - Against):** +16.0
    -   **Verdict:** The model validates Leicester's performance, but with nuance. They were *not* the best team by pure chance creation (Arsenal and Spurs were higher), but they were elite. Crucially, their "Goals For" perfectly matched their "CxG For" (68 vs 68). This implies they didn't "get lucky" with finishing; they simply created high-quality chances (likely from counter-attacks, which our model rewards heavily due to the `def_trigger` and `pressure` features) and converted them at a sustainable rate.

-   **The "True" Best Teams:**
    -   **Arsenal:** CxG Diff +29.5. They were the statistical champions, creating far more than they conceded. Their failure to win the league was a failure of converting dominance into points, not a lack of underlying performance.
    -   **Tottenham:** CxG Diff +24.9. Similar to Arsenal, they were statistically superior to Leicester but fell short in key moments.

### 6.2 Finishing Variance & Relegation
*Reference Chart: `outputs/modeling/cxg/prediction_runs/pl_2015_16_club/charts/finishing_delta.png`*

This chart visualizes `Goals Scored - CxG`, effectively measuring "Finishing Luck" or "Skill" (depending on your philosophy).

-   **Over-performers:**
    -   **West Ham (+14.0):** The standout over-performer. Dimitri Payet's free-kicks and long-range screamers broke the model. While the model saw "low probability shot from 30 yards," Payet saw a goal. This +14 goal swing likely propelled them much higher up the table than their chance creation warranted.
    -   **Manchester City (+11.5):** A classic sign of elite talent. With Aguero and De Bruyne, City consistently scored from chances that an "average" team (which the model assumes) would miss.

-   **Under-performers (The Relegation Battle):**
    -   **Aston Villa (-1.7):** Villa's season was a disaster on all fronts. They had the worst CxG Difference (-29.4) *and* they under-finished. There was no "bad luck" here; they were simply the worst team.
    -   **Newcastle United (-11.6):** A fascinating case. Their CxG Difference was bad, but not "worst in the league" bad. However, they conceded ~11 more goals than expected (or scored fewer, depending on the split). This suggests a fragility—perhaps poor goalkeeping or defensive errors leading to "easy" chances that the model didn't fully capture.

### 6.3 Provider Comparison
*Reference Chart: `outputs/modeling/cxg/prediction_runs/pl_2015_16_club/charts/cxg_vs_provider_scatter.png`*

We compared our CxG totals against the Provider's xG totals for every team.
-   **Correlation:** Very high (>0.95). This confirms our model captures the fundamental "truth" of the game similarly to established providers.
-   **Deviation:** Our model tends to be slightly more conservative on "low quality" shots but rewards "high context" chances more generously. For example, a tap-in after a "High Turnover" might get 0.85 CxG in our model vs 0.75 in the provider model, because we explicitly account for the disorganized defense via the `def_trigger` submodel.

### 6.4 Neutral vs. PL-Inclusive Priors
We ran a controlled experiment (`pl_2015_16_bias_comparison.json`) to see if including PL data in the *priors* generation (but not the contextual model training) improved accuracy.

-   **Match MAE (Exclude PL):** 0.740
-   **Match MAE (Include PL):** 0.740
-   **Team Bias (Goals - CxG):** +0.53 (Exclude) vs +0.56 (Include)

**Conclusion:** Including the specific PL history in the priors didn't significantly reduce match-level error. This is a **positive result** for the Neutral Priors approach. It implies that the "Style Clusters" and "Rolling Form" derived from international play (World Cup/Euros) are robust enough to describe Premier League teams. We don't *need* to know that "Arsenal is Arsenal"; knowing that "This team plays like a High-Possession/High-Press Style 1 team" is sufficient to predict their finishing characteristics accurately. This validates the portability of our model to new leagues without extensive retraining.

---

## 7. Operational Workflow

The project delivers a reproducible command-line interface (CLI) for analysts.

### 7.1 Generating Predictions
The prediction pipeline (`src/opponent_adjusted/prediction/run_pipeline.py`) is designed for production-grade inference. It handles the complexity of loading models, validating schemas, and aggregating results.

**Workflow:**
1.  **Model Loading:** The script loads the trained model artifact (`.joblib`) and its corresponding metadata (`.json`).
2.  **Feature Contract Enforcement:** This is a critical step. The metadata contains the exact list of features (numeric, binary, categorical) used during training. The pipeline ensures that the inference dataset matches this schema exactly—ordering columns correctly and filling missing columns with defaults if necessary—to prevent "feature mismatch" errors.
3.  **Scoring:** The `predict_proba` method is called on the prepared dataset.
4.  **Aggregation:** The raw shot-level probabilities are aggregated into two levels:
    -   **Match Level:** Summing CxG per team per match.
    -   **Team Level:** Summing CxG across the entire season.

**Command:**
```bash
# 1. Ingest Data
poetry run python scripts/ingest_events.py --competition 2 --season 27

# 2. Build Features
poetry run python scripts/build_shot_features.py --version-tag cxg_v1

# 3. Run Pipeline
poetry run python -m opponent_adjusted.prediction.run_pipeline \
    outputs/modeling/cxg/cxg_dataset_enriched.parquet \
    --tag my_new_run
```

### 7.2 Visualization
The visualization suite (`src/opponent_adjusted/prediction/plot_reports.py`) automates the generation of insight-ready charts. It uses `matplotlib` and `seaborn` to produce high-quality static assets.

**Key Charts Generated:**
1.  **Team Totals (Bar Chart):** Compares Goals vs. CxG vs. Provider xG for the top N teams. This gives an immediate view of the "League Table of Justice."
2.  **Finishing Delta (Diverging Bar Chart):** Plots `Goals - CxG`. Bars to the right (Green) indicate over-performance; bars to the left (Red) indicate under-performance. This is the primary tool for identifying "lucky" or "clinical" teams.
3.  **Scatter Plot (CxG vs. Provider):** A regression plot to check alignment with the industry standard. Outliers here indicate matches or teams where our Contextual Model strongly disagrees with the geometric baseline, warranting further investigation.

**Command:**
```bash
poetry run python -m opponent_adjusted.prediction.plot_reports \
    outputs/modeling/cxg/prediction_runs/my_new_run/team_aggregates.csv
```

---

## 8. Conclusion & Future Work

### 8.1 Summary
This project has successfully demonstrated that **Contextual, Opponent-Adjusted xG** can be built using open data and a neutral modeling framework. By decoupling the model from specific Team IDs, we created a flexible tool that adapts to new leagues and seasons instantly. The "Enriched" model's superior metrics (AUC 0.865) validate the hypothesis that context (pressure, defensive form) matters just as much as location.

### 8.2 Limitations
-   **Tracking Data:** We rely on "freeze frames" for pressure. Full 25fps tracking data would allow for velocity-based pressure models, likely improving accuracy further.
-   **Sample Size:** The "Neutral Priors" rely on style clusters. With only WC/Euro data for training, the variety of styles is limited compared to a full domestic league database.

### 8.3 Next Steps
1.  **Expand Training Corpus:** Ingest more open data (e.g., FA WSL, NWSL) to robustify the style clusters.
2.  **Goalkeeper Model:** Currently, the model assumes an "average" keeper. Adding a "Goalkeeper Saving Ability" submodel (similar to the Finishing Bias model) would refine Concession Bias.
3.  **Live Inference:** Wrap the `run_pipeline.py` in a FastAPI endpoint for real-time match scoring.

---
*End of Report*
