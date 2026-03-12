# Contextual Expected Goals (CxG) Modelling Roadmap

_Revision: 21 Nov 2025_

## 0. Purpose & Scope
This document specifies the sequential, academically grounded plan to convert the completed exploratory analysis (see `docs/analysis/cxg/cxg_analysis_report.md`) into a production‑grade Contextual Expected Goals (CxG) model. CxG aims to predict the probability that a shot results in a goal while incorporating multi‑layer match context: attacking sequence (pass value chain), game state, set‑piece microstates, assist and receiver attributes, defensive overlay, spatial geometry, and team/opponent latent tendencies.

We divide the roadmap into ten logical phases: Data Governance, Feature Engineering, Sub‑Models, Baseline Modelling, Hierarchical Extension, Hybrid Stacking, Calibration & Reliability, Evaluation Protocol, Deployment & Monitoring, and Future Extensions.

---
## 1. Data Governance & Preparatory Steps

### 1.1 Dataset Definition
- **Temporal scope:** Fix competition(s) and match range. Ensure reproducible sample (e.g., all 2022 tournament matches contained in current StatsBomb subset).
- **Entity completeness:** Each shot must link to: `ShotFeature` (distance, angle, score diff flags), `Event` (location, minute, second, under_pressure), `RawEvent` (pass/shot metadata), `Team`.
- **Exclusions:** Own goals, penalties optionally modelled separately; shots lacking distance or angle removed (geometry‑level analysis shows strong monotonicity—missing geometry precludes calibration).

### 1.2 Integrity Audits
- **Referential checks:** No orphan shot IDs; consistent possession IDs for chain construction.
- **Time resolution:** Confirm `minute`+`second` are non‑negative and increasing within match; required for defensive trigger time gaps.
- **Pressure flag consistency:** Validate under_pressure distribution across categories (report proportion per assist category; large anomalies flagged).
- **Set-piece classification accuracy:** Cross‑verify play pattern mappings (penalties should never appear as indirect free kicks).

### 1.3 Version Control
- Freeze a **Feature Schema v1** capturing stable columns: 
  - Geometry: `shot_distance`, `shot_angle`
  - Sequence: `chain_label`, `assist_category`, `pass_style`, `set_piece_category`, `set_piece_phase`, `def_label`
  - Context: `score_state`, `minute_bucket_label`, `pressure_state` (derived from `under_pressure`), `is_leading/is_trailing/is_drawing`
  - Entities: `team_id`, `opponent_team_id`, `match_id`
  - Outcomes: `is_goal`, `statsbomb_xg`
- Tag feature extraction code (ShotFeature generation scripts) with version hash.

### 1.4 Leakage Avoidance
- Ensure no post‑shot events influence features. Defensive triggers bounded to ≤5 seconds **before** shot (see defensive overlay module). Rebounds classified using key pass markers only, not subsequent outcomes.

### 1.5 Data Splitting Strategy
- **Primary split:** Match‑level stratified k‑fold (e.g., k=5) to avoid within‑match leakage.
- **Temporal holdout:** Last 10–15% of matches for prospective calibration check.
- **Competitional stratification:** If multiple tournaments, ensure each fold contains samples from all.

---
## 2. Feature Engineering Framework

We adopt a layered design: base geometry, contextual categorical factors, engineered interactions, smoothed spatial surfaces, and latent random effects.

### 2.1 Geometry Features
- Raw: `shot_distance`, `shot_angle` (radians). Exploratory distance bins show monotonic decline (see `geometry_distance_bins.csv`). Angle‑distance ridge for shallow cutback angles (plots: `geometry_angle_distance_heatmap.png`).
- Derived: 
  - `distance_sq_root` (stabilize marginal effect), 
  - `angle_sin`, `angle_cos` (circular encoding), 
  - `distance × angle`, 
  - Lateral offset normalized: `(location_y / pitch_width)`.

### 2.2 Sequence & Pass Chain Features
- `chain_label` hierarchical decomposition: preceding action (Carry/Dribble/Duel/Direct) + `pass_style` (Through Ball, Cutback, Cross, Switch, High/Ground Pass, etc.). Through‑ball and cutback high lift (see §1.1 efficiency tiers).
- Reduced cardinality encoding: Map rare chains (<30 shots) to prototype parent class (e.g., "Direct + Other Pass").
- Binary indicators for high‑impact patterns: `is_through_ball`, `is_cutback`, `is_unassisted`.

### 2.3 Assist & Receiver Context
- Assist category + pressure interaction: empirical gap between unpressured vs pressured crosses (goal rate 0.171 vs 0.098).
- Receiver foot assumption (if available): `is_weak_foot_finish` (penalty to base probability ~6–9 pp). If footedness incomplete, fallback to body_part categories.
- Pass geometry: `pass_length`, `pass_angle`, `pass_height` (categorical), endpoints `(pass_end_x, pass_end_y)` as spatial vector.

### 2.4 Set-Piece Microstates
- `set_piece_category` × `set_piece_phase` × `score_state` × `minute_bucket_label`. Second‑phase throw‑ins and other restarts late (75–90) show elevated conversion.
- Direct penalties separated: model as independent Bernoulli with distinct calibration.

### 2.5 Game State Dynamics
- `score_state` (Trailing 2+, Trailing 1, Level, Leading 1, Leading 2+) and `minute_bucket_label` interactions. Chains invert performance under different states (e.g., Direct + Cross leading vs trailing). Encode as joint factor or two effect terms plus interaction penalty.

### 2.6 Defensive Overlay
- Main triggers: Pressure, Ball Recovery, Carry (opponent ball progression), Block – Deflection, No immediate trigger. Use one‑hot with potential grouping for extremely sparse triggers.
- Time gap buckets: 0–2s vs 2–5s prior events; earlier triggers may signal transition exploitation vs settled defense.

### 2.7 Team & Opponent Effects
- Latent random intercepts: `(1 | team_id)` (finishing bias) and `(1 | opponent_team_id)` (defensive concession bias). Partial pooling to manage sparse teams.
- Optional dynamic component: Rolling prior using exponential decay over matches for online updating.

### 2.8 Interaction Inventory (Curated)
- Cutback × shallow angle × distance (10–15m)
- Through Ball × trailing state (counter context effect)
- Cross × pressure_state (performance degradation when pressured)
- Set Piece Phase × minute_bucket_label (late rebounds vs early phases)
- Defensive trigger × chain archetype (e.g., Ball Recovery preceding Through Ball) 
- Team random effect × chain (varying slopes if data permits; advanced extension)

### 2.9 Encoding & Regularization
- Continuous: StandardScaler (for linear/logit) or left raw for tree models.
- High cardinality categorical (team/opponent): use pure random effect in Bayesian model; use target (leave-one-out) encoding for GBM baseline (with K-fold blending to avoid leakage).
- Interaction pruning: Apply L1 penalty in logistic or use tree model feature importance to remove weak interactions.

### 2.10 Monotonic Constraints
- For gradient boosting (LightGBM / XGBoost monotonic): enforce negative monotonicity for distance; positive for angle (within appropriate domain). Prevent physically implausible reversals.

---
## 3. Sub‑Models (Prior Estimators)
Building lower‑dimensional sub‑models yields interpretable priors and stabilized effects fed into the main CxG model.

| Sub‑Model | Objective | Inputs | Output Feature | Rationale |
| --- | --- | --- | --- | --- |
| Team Finishing Bias | Estimate team finishing over baseline | statsbomb_xg, team_id | `team_finish_uplift` | Captures systematic over/under performance |
| Opponent Concession Bias | Defensive concession tendency | statsbomb_xg, opponent_team_id | `opp_concede_uplift` | Adjust probabilities vs weaker defenses |
| Set-Piece Phase Model | Phase conversion probability | set_piece_category, phase | `set_piece_phase_prior` | Distinct dynamics of first vs second phase |
| Assist Quality Uplift | Extra value of pass geometry | pass_length, pass_angle, pass_style | `assist_quality_delta` | Quantifies pass geometry beyond chain label |
| Pressure Penalty Model | Effect of pressure on conversion | pressure_state, geometry | `pressure_penalty` | Isolates pressure influence |
| Spatial Surface (GP / TPS) | Smooth location baseline | (x, y) | `spatial_prior` | Reduces noise in distance/angle raw features |
| Defensive Trigger Uplift | Disruption effect on outcome | def_label, time_gap | `def_trigger_uplift` | Incorporates recent defensive context |

Outputs (posterior means or predictions) are appended as numerical predictors to the integrated model.

---
## 4. Baseline Model (Level 1)

### 4.1 Specification
Logistic regression with elastic net regularization.

`logit(p_goal) = β0 + f(distance, angle) + Σ β_chain[chain] + Σ β_assist[assist] + β_pressure * pressure_state + β_setpiece[category × phase] + β_score[score_state] + β_minute[minute_bucket]`

### 4.2 Training Protocol
- Fit on K‑fold (match‑stratified) splits.
- Use class weighting only if extreme imbalance (<5% goals); prefer probability calibration metrics.
- Evaluate incremental gain: geometry only vs + chain vs + set-piece vs + state vs + pressure.

### 4.3 Deliverables
- Coefficient table with 95% bootstrap intervals.
- Residual diagnostics: predicted vs actual goal rate across distance bins.

---
## 5. Hierarchical Bayesian Model (Level 2)

### 5.1 Model Form
`logit(p_goal_i) = α + s_geom(distance_i, angle_i) + γ_chain[c_i] + δ_assist[a_i] + θ_state[score_i, minute_i] + φ_setpiece[cat_i, phase_i] + ψ_def[def_i] + u_team[team_i] + v_opp[opp_i]`

- `s_geom`: Thin‑plate spline or low‑rank Gaussian Process on (distance, angle) or (x, y).
- Random intercepts: `u_team ~ Normal(0, σ_team)`, `v_opp ~ Normal(0, σ_opp)`.
- Optional random slope: selected high‑impact chain (e.g., Through Ball) varying per team if data volume permits.

### 5.2 Priors
- Fixed effects: `Normal(0, 0.5)` (weakly informative given logit scale).
- Spline coefficients: penalized with smoothing prior (e.g., second‑order random walk / intrinsic Gaussian prior).
- Variances: `σ_team, σ_opp ~ HalfStudentT(ν=4, scale=0.3)`.

### 5.3 Inference
- Use NUTS (Stan/PyMC); monitor convergence (R‑hat < 1.01, effective sample size > 400).
- If computationally heavy, fall back to variational inference for prototype, then refine with subset HMC.

### 5.4 Advantages
- Partial pooling controls overfitting of rare chains and sparse teams.
- Posterior uncertainty communicates reliability of uplift adjustments.

---
## 6. Hybrid Ensemble (Level 3)

### 6.1 Components
- Hierarchical Bayesian posterior mean predictions.
- Gradient Boosting Machine (LightGBM) with monotonic constraints (distance ↓, angle ↑) and early stopping.
- Spatial prior surface (GP/TPS) as an external feature.

### 6.2 Stacking Meta‑Learner
Simple logistic regression on out‑of‑fold predicted probabilities: 
`logit(p_final) = β0 + β1 p_bayes + β2 p_gbm + β3 spatial_prior + ε`

### 6.3 Calibration Alignment
Post‑stack isotonic regression or beta calibration to refine sharpness without distorting relative ranking.

---
## 7. Calibration & Reliability

### 7.1 Metrics
- **Brier Score**, **Log Loss**, **AUC** (secondary), **Expected Calibration Error (ECE)**.
- **Calibration slope** (ideal ≈1) and intercept (ideal ≈0).

### 7.2 Subgroup Calibration
Evaluate reliability curves for: chain_label, assist_category, pressure_state, set_piece_category, distance bins, score_state. Identify systematic under/overestimation to adjust priors or introduce interaction terms.

### 7.3 Recalibration Strategy
If a subgroup shows consistent misfit (e.g., cutbacks late vs early), apply subgroup‑specific recalibration (piecewise isotonic) or add hierarchical interaction term in next iteration.

---
## 8. Evaluation Protocol

### 8.1 Cross‑Validation
- Match‑stratified folds; ensure consistent goal proportion per fold.
- Nested CV for hyperparameter search (outer 5 folds, inner 3 folds for GBM).

### 8.2 Temporal Holdout
- Final season segment reserved for unbiased future performance check.

### 8.3 Ablation Study
Record ΔBrier and ΔLogLoss after adding each feature block: Geometry → Chain → Assist → Set-piece → Game State → Defensive Overlay → Random Effects → Ensemble.

### 8.4 Statistical Testing
- Paired bootstrap on match sets (≥1,000 resamples) for Brier difference significance.
- McNemar test in classification threshold framing (optional).

### 8.5 Error Diagnostics
- Plot predicted vs empirical goal rate across deciles.
- Investigate largest residual clusters (e.g., high predicted probability but miss) for modeling gaps (keeper effects, shot technique).

---
## 9. Deployment & Monitoring

### 9.1 Inference Pipeline
1. Precompute slow priors (team finish, opponent concede, spatial surface) nightly.
2. For each new shot event: extract real‑time features (distance, angle, chain, assist, pressure, game state, set-piece context, defensive trigger if within window) → feed into stacked model → output probability.

### 9.2 Latency Considerations
- Logistic + GBM prediction: sub‑millisecond per shot.
- Spatial prior lookup: constant time via grid index.
- Random effect retrieval: dictionary access.

### 9.3 Monitoring Metrics
Weekly: overall Brier, calibration slope, distribution drift for distance and chain_label (KS test). Alert thresholds on slope deviation >0.05 or Brier > baseline +0.01.

### 9.4 Model Card
Publish: data version, training dates, feature list, performance metrics, known failure modes, calibration plots, ethical considerations (avoid misuse in player valuation without confidence intervals).

---
## 10. Future Extensions

| Extension | Concept | Expected Benefit | Considerations |
| --- | --- | --- | --- |
| Keeper Interaction Model | Add GK positioning & action features | Capture save difficulty | Requires additional labeled data |
| Post‑Shot xG Layer | Model shot velocity & placement (if available) | Improves end‑to‑end scoring probability | Data availability uncertain |
| Sequential Transformer | Encode last N events of possession | Richer tactical context | Lower interpretability |
| Real‑Time Adaptation | Online Bayesian updating of team priors | Adaptive accuracy during tournaments | Risk of drift if data sparse |
| Counterfactual Simulator | Alter chain or pressure states to project outcomes | Tactical scenario planning | Requires stable causal assumptions |

---
## 11. Recommended Initial Execution Order
1. Data audit & feature schema freeze.
2. Geometry baseline logistic + calibration plots.
3. Pass value chain & assist integration (analyze coefficient stability).
4. Add game state + minute interactions; prune weak terms.
5. Integrate set-piece microstates and phase priors.
6. Defensive overlay triggers and pressure penalties.
7. Introduce team/opponent random effects (Bayesian fit prototype).
8. Train GBM with monotonic constraints; perform stacking.
9. Full calibration (isotonic/beta) and subgroup reliability verification.
10. Deployment packaging + monitoring dashboard specification.

---
## 12. Risk Register & Mitigations
| Risk | Description | Impact | Mitigation |
| --- | --- | --- | --- |
| Sparse Categories | Rare chains inflate variance | Unstable predictions | Partial pooling; collapse <30-shot chains |
| Temporal Drift | Style shifts over season | Calibration decay | Rolling recalibration; time covariate |
| Data Leakage | Post-event info leaks | Inflated metrics | Strict pre-shot window enforcement |
| Overfitting Interactions | Excessive combinatorial terms | Poor generalization | L1 regularization; hierarchical shrinkage |
| Misclassified Set Pieces | Incorrect phase/category | Bias phase priors | Validation sample manual review |
| Pressure Flag Noise | Inconsistent tagging | Unreliable penalty effect | Robustness check; sensitivity analysis |

---
## 13. Academic Framing
The CxG system operationalizes a **hierarchical generalized additive logistic model** enriched by **contextual categorical priors** and **non-parametric spatial smoothing**, then ensemble‑calibrated. It treats contextual constructs (chains, phases, pressure) as structured covariates while regularizing via partial pooling to avoid variance inflation in sparse regimes. Calibration is treated as a first-class objective (probabilistic integrity over ranking), supported by reliability decomposition across contextual strata. This aligns with best practices in sports analytics for interpretable yet performance‑oriented probabilistic modelling.

---
## 14. Output Artifacts Checklist
- Baseline geometry model coefficients & diagnostics
- Context increment analysis (delta metrics per feature block)
- Hierarchical Bayesian posterior summary (random intercept SDs)
- Stacked ensemble final metrics (Brier, ECE, AUC)
- Reliability plots per key subgroup
- Model card + risk register
- Monitoring specification document

---
## 15. Next Action
Proceed to implement Phase 1 (Audit + Feature Schema Freeze) and scaffold baseline geometry model script under `src/opponent_adjusted/modeling/cxg/`.

---
_Authored for internal modelling reference. Questions or revision proposals can be logged via the active pull request discussion._
