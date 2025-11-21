# CxG Modelling Results Summary

_Revision: 21 Nov 2025_

## 1. Runs Covered
- **Baseline Geometry:** Logistic regression on distance & angle (5-fold, match-stratified).
- **Contextual (Filtered):** Contextual pipeline trained on filtered dataset (excludes low-signal events).
- **Contextual (Enriched Priors):** Same architecture with sub-model priors (finishing, set-piece, assist, pressure, defensive triggers).
- **StatsBomb Provider xG:** Raw `statsbomb_xg` probabilities evaluated on the same sample.

## 2. Aggregate Metrics
| Model | ROC AUC | Brier Score | Log Loss |
| --- | --- | --- | --- |
| Baseline Geometry | 0.739 | 0.0760 | 0.2728 |
| Contextual (Filtered) | 0.827 | 0.0678 | 0.2378 |
| Contextual (Enriched Priors) | **0.865** | **0.0631** | **0.2188** |
| StatsBomb Provider xG | 0.799 | 0.0679 | 0.2448 |

**Key deltas:**
- Enriched contextual model reduces Brier by **0.0129** vs baseline and **0.0047** vs filtered contextual.
- Log-loss improves by **0.054** over baseline and **0.019** over filtered contextual.
- ROC AUC gain of **0.126** vs baseline and **0.066** vs provider xG shows stronger ranking of true goals.

## 3. Reliability Insights
- Multi-line calibration plot: `outputs/modeling/cxg/modeling_charts/cxg_reliability_overlay.png`.
- Enriched contextual curve stays closest to the ideal diagonal across bins, particularly 0.45–0.85 probability range.
- Provider xG overstates probabilities above 0.65; baseline geometry underestimates mid-range (0.15–0.35) bins.

## 4. Supporting Artifacts
- Comparison bars: `outputs/modeling/cxg/modeling_charts/model_compare_{brier_mean,log_loss_mean,auc_mean}.png`.
- Metrics table CSV: `outputs/modeling/cxg/modeling_charts/model_metric_table.csv`.
- Contextual feature effect exports: `outputs/modeling/cxg/contextual_feature_effects{,_filtered,_enriched}.csv`.
- Reliability curves per run under `outputs/modeling/cxg/plots/`.

## 5. Next Steps
1. Promote enriched contextual model as default scoring engine and persist pipeline artifacts (see §6).
2. Integrate results into monitoring dashboards (include Brier/log-loss target bands).
3. Expand documentation with subgroup calibration (assist category, set-piece phase) once available.

## 6. Model Persistence Checklist
- Serialize trained scikit-learn pipelines (baseline + contextual variants) to `outputs/modeling/cxg/models/`.
- Store accompanying metadata JSON (training dataset version, feature lists, CV metrics, timestamp).
- Version model artifacts using semantic tags (`cxg_contextual_enriched_v1.joblib`).
- Document loading example for inference services.
