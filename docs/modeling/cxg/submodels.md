# CxG Sub-Model Blueprint

This note expands the roadmap into concrete modeling modules that can be implemented on top of the filtered feature matrix now available under `outputs/modeling/cxg/cxg_dataset_filtered.{parquet,csv}`. Each sub-model isolates a contextual factor, produces an interpretable adjustment, and feeds the unified CxG stack.

## Current Sub-Model Outputs

The first descriptive passes for each sub-model now live under `outputs/modeling/cxg/submodels/`. These CSVs will act as priors until the probabilistic fits described below are implemented.

- **Finishing Bias** (`finishing_bias_by_team.csv`, `concession_bias_by_opponent.csv`)
  - Ghana (+0.10 lift), Colombia (+0.066), Russia (+0.055) headline the over-performers; Iceland, Canada, Panama lag by −0.06 to −0.07.
  - On the defensive side Canada (+0.114 conceded lift) and Qatar (+0.100) allow well above expected finishing while Colombia (−0.074) suppresses chances.
  - Use these lifts as shrinkage targets when fitting the Bayesian team/opponent random-effect model.
  - **Modeled priors**: `python -m opponent_adjusted.modeling.cxg.submodels.train_finishing_bias_model` fits an L2-regularized logistic on `statsbomb_xg + team_id + opponent_team_id`, saves a reusable `.joblib`, metrics JSON, and bias tables with logit coefficients/odds multipliers for each team.
- **Set-Piece Phase** (`set_piece_phase_uplift.csv`)
  - Direct free kicks at level score lines between 45–60' show +0.085 lift, whereas late direct attempts trend below xG.
  - Indirect first-phase restarts spike when trailing 2+ in stoppage (+0.14) and sag when trailing 1 between 60–75' (−0.093).
  - Throw-in second phases at level states deliver the largest uplift (+0.10 in stoppage), suggesting under-modeled restart danger.
  - **Modeled priors**: `python -m opponent_adjusted.modeling.cxg.submodels.train_set_piece_phase_model` stores `set_piece_phase_model.joblib`, metrics JSON, and `set_piece_phase_modeled.csv` with shrinkage-adjusted logits per category × phase × score/minute bucket.
- **Assist Quality** (`assist_quality_summary.csv`)
  - Unpressured set-piece through balls (+0.102) and open-play cutbacks (+0.071) provide the clearest boosts.
  - High passes, switches, and pressured crosses consistently underperform (−0.02 to −0.04), reinforcing the need for pressure-aware pass features.
  - **Modeled priors**: `python -m opponent_adjusted.modeling.cxg.submodels.train_assist_quality_model` exports `assist_quality_model.joblib`, metrics, and `assist_quality_modeled.csv` keyed by assist category × pass style × pressure state.
- **Pressure Influence** (`pressure_influence_summary.csv`)
  - Central shots inside 12 m lose ~0.026 probability when pressured despite similar base xG, giving us a clear multiplicative penalty.
  - Beyond 18 m, pressure effects flatten; we can taper penalties with distance to avoid over-correction on long shots.
  - **Modeled priors**: `python -m opponent_adjusted.modeling.cxg.submodels.train_pressure_influence_model` builds per-bucket coefficients over `pressure_state × distance_bin × angle_bin`, saving outputs to `pressure_influence_modeled.csv`.
- **Defensive Trigger** (`defensive_trigger_uplift.csv`)
  - Same-possession carries within 5–10 s of the defensive action add +0.037 lift; 0–2 s ball recoveries also over-perform (+0.024).
  - Immediate blocks create a −0.058 crash, highlighting sequences to deprioritize when building counter-press priors.
  - **Modeled priors**: `python -m opponent_adjusted.modeling.cxg.submodels.train_defensive_trigger_model` persists `defensive_trigger_model.joblib`, diagnostics, and `defensive_trigger_modeled.csv` for `def_label × time_gap_bucket × possession_state` combinations.

Each CSV includes counts, goal rates, aggregated xG, and a `lift_vs_xg` column to make it easy to convert into additive logit adjustments or multiplicative weights.

## Enriching the Modelling Dataset

Run `python -m opponent_adjusted.modeling.cxg.enrich_cxg_with_submodels` to merge every modeled prior onto the latest CxG dataset (preferring the filtered version when present). The script writes `cxg_dataset_enriched.parquet/csv` under `outputs/modeling/cxg/` with the following new feature groups:

- `finishing_bias_logit`, `finishing_bias_multiplier`, `concession_bias_*` joined by `team_id`/`opponent_team_id`.
- `set_piece_bucket`, `set_piece_logit`, `set_piece_multiplier`, `set_piece_modeled_prob` keyed by category × phase × score/minute bucket.
- `assist_bucket`, `assist_quality_logit`, `assist_quality_multiplier`, `assist_quality_modeled_prob` keyed by assist/pass/pressure state.
- `pressure_bucket`, `pressure_logit`, `pressure_multiplier`, `pressure_modeled_prob` built from pressure state + distance/angle bins.
- `def_trigger_bucket`, `def_trigger_logit`, `def_trigger_multiplier`, `def_trigger_modeled_prob` covering defensive label × time-gap × possession alignment.

Missing matches default to logit `0.0` (no adjustment) and multiplier `1.0`, keeping the enriched table backward-compatible with existing modelling pipelines.

## 1. Team Finishing Bias Model
- **Objective**: Estimate finishing over/under-performance for shooting teams and concession bias for opponents after controlling for baseline StatsBomb xG.
- **Inputs**: `statsbomb_xg`, `is_goal`, `team_id`, `opponent_team_id`, `match_id` (for grouped CV).
- **Model**: Bayesian logistic (PyMC/Stan) or regularized GLMM.
  - `logit(p_goal) = α + β·statsbomb_xg + u_team[team_id] + v_opp[opponent_team_id]`
  - Priors: `u ~ Normal(0, σ_team)`, `v ~ Normal(0, σ_opp)`.
- **Outputs**:
  - Posterior means for `u_team`, `v_opp` → `finishing_bias_team`, `concession_bias_opp`.
  - Posterior predictive distribution for goal probability residuals.
- **Integration**: Add bias terms as numeric features or convert to multiplicative uplift (`exp(u)`), pass downstream to baseline/contextual/GBM fits.

## 2. Set-Piece Phase Model
- **Objective**: Quantify conversion lift for direct/first/second-phase restarts, independent of geometry.
- **Inputs**: `set_piece_category`, `set_piece_phase`, `score_state`, `minute_bucket_label`, `shot_distance`, `shot_angle`.
- **Model**: Multilevel logistic by category × phase × (score/minute) with shrinkage threshold to handle sparse combinations.
- **Outputs**:
  - `set_piece_uplift` per category-phase bucket.
  - Reliability tables for coaches.
- **Integration**: Provide `set_piece_uplift` as additive logit term or multiplier used in stacking/ensembles.

## 3. Assist Quality Model
- **Objective**: Produce a continuous measure of pass-driven shot quality uplift beyond raw geometry.
- **Inputs**: `assist_category`, `pass_style`, `pressure_state`, `chain_label`; (future) raw pass metrics from `passes` table (length, angle, height, end_x/y).
- **Model**: Gradient boosted regression or logistic that predicts `goal - base_geom_prob` using pass descriptors.
- **Outputs**:
  - `assist_quality_score` (centered at zero) per shot.
  - Optional categorical breakdown (cutback vs cross vs through ball under pressure).
- **Integration**: Add as feature to contextual logistic / GBM, allows quick ablation to show pass contribution.

## 4. Pressure Influence Model
- **Objective**: Isolate penalty or lift due to defensive pressure flags controlling for geometry and assist.
- **Inputs**: `pressure_state`, `shot_distance`, `shot_angle`, `assist_category`, `chain_label`, `score_state`.
- **Model**: Simple logistic with interaction terms (pressure × distance/angle). Could start with regularized GLM before moving to Bayesian partial pooling by chain.
- **Outputs**:
  - `pressure_penalty` coefficient(s) or per-bucket multipliers (Under Pressure vs Not).
  - Calibration check to ensure effect is monotonic with geometry.
- **Integration**: Additional feature representing expected delta vs no-pressure baseline.

## 5. Defensive Trigger Model
- **Objective**: Measure how recent defensive actions (pressure, block, tackle, carry) influence conceded xG within the time gap.
- **Inputs**: `def_label`, `time_gap_seconds`, `possession_match`, `chain_label`, `shot_distance`, `shot_angle`.
- **Model**: Logistic with categorical `def_label` effects + spline/step for `time_gap_seconds` (0–2s, 2–5s, >5s). Partial pooling for low-volume labels.
- **Outputs**:
  - `def_trigger_uplift` per label + gap bucket.
  - Possession alignment penalty indicator.
- **Integration**: Pass the uplift as feature; also use for defensive scouting dashboards.

## 6. Spatial Surface Baseline
- **Objective**: Produce a smooth geometry-only probability surface to serve as the initial logit term.
- **Inputs**: `location_x`, `location_y` (converted to StatsBomb pitch coordinates), optionally `body_part`.
- **Model**: Thin-plate spline GAM or 2D Gaussian Process; enforce monotonic decrease with distance if using GBM with monotonic constraints later.
- **Outputs**:
  - `p_geom_surface` baseline probability per shot.
  - Residuals `goal - p_geom_surface` for downstream models.
- **Integration**: Use as base logit in hierarchical logistic; residualize other features for stability.

## Data & Engineering Notes
- Current filtered dataset already includes most categorical drivers (`chain_label`, `pass_style`, `assist_category`, `def_label`, `set_piece_phase`, `score_state`).
- Team/opponent random effects require consistent IDs; we have `team_id`, `opponent_team_id` (54 each) → adequate volume.
- Assist quality may need richer pass metadata (length, angle). Plan to extend `build_cxg_dataset` with pass joins once ingestion scripts expose `PassEvent` fields.
- Pressure model benefits from additional continuous pressure metrics (`recent_def_actions_count`, `pressure_proxy_score`) stored in `shot_features`; consider adding them to the dataset.

## Integration Strategy
1. **Train sub-models independently** using filtered dataset + any extra joins. Persist outputs under `outputs/modeling/cxg/submodels/`.
2. **Augment dataset** with sub-model scores (e.g., add columns `finishing_bias_team`, `set_piece_uplift`, `assist_quality_score`, `pressure_penalty`, `def_trigger_uplift`, `p_geom_surface`).
3. **Update contextual_model.py** to ingest these engineered features and log ablation improvements (ΔBrier, Δlog-loss).
4. **Stacking layer**: Create `stacked_model.py` that takes probabilities from (a) spatial surface baseline, (b) hierarchical logistic with context features, (c) tree-based model. Fit meta-logistic to calibrate final CxG.
5. **Reporting**: Add sub-model metrics and insights to `docs/modelling/cxg/modelling_roadmap.md` appendix + `IMPLEMENTATION_SUMMARY.md`.

This blueprint keeps the modelling hierarchy modular: each sub-model is independently testable, produces meaningful scouting insights, and improves the overall CxG estimation when stacked or used as feature inputs.
