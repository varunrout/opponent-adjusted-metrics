# CxG Modelling Results Summary

_Revision: 24 Nov 2025_

## 1. Runs Covered
- **Baseline Geometry:** Logistic regression on distance & angle (5-fold, match-stratified).
- **Contextual (Filtered):** Contextual pipeline trained on filtered dataset (excludes low-signal events).
- **Contextual (Enriched Priors):** Same architecture with sub-model priors (finishing, set-piece, assist, pressure, defensive triggers).
- **Contextual (Neutral Priors Refresh):** Retrained contextual model that consumes the new neutral finishing/concession priors (match/side keyed, no team IDs) generated on the WC/Euro corpus (`contextual_model_neutral_priors_refresh`).
- **Contextual (Neutral Priors + PL15/16 Finishing Bias):** Same as above but with `CXG_INCLUDE_PL1516_PRIORS=1` so the finishing bias submodel can learn directly from Premier League 2015/16 match-sides; downstream contextual artifact is `contextual_model_neutral_priors_refresh_plfb`.
- **StatsBomb Provider xG:** Raw `statsbomb_xg` probabilities evaluated on the same sample.

The neutral priors refresh replaces team identifiers with rolling form, style clusters, and player-level lifts. Use it for any competitions where you want neutralized opponent context without leaking explicit club features.

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

## 3. Slice-Level Evaluation (ECE / Calibration)
- Script: `poetry run python -m opponent_adjusted.modeling.cxg.evaluation.slice_metrics`
- Outputs: `outputs/modeling/cxg/evaluation/slice_metrics.csv` + `summary.json`

| Model | Dataset | Brier | Log Loss | ROC AUC | ECE |
| --- | --- | --- | --- | --- | --- |
| Contextual (Enriched Priors) | Enriched | **0.0599** | **0.2072** | **0.8787** | **0.0056** |
| Contextual (Filtered) | Filtered | 0.0650 | 0.2274 | 0.8432 | 0.0104 |
| StatsBomb Provider xG | Enriched | 0.0679 | 0.2448 | 0.7991 | 0.0066 |
| Contextual (Raw) | Raw | 0.0710 | 0.2432 | 0.8689 | 0.0103 |
| Baseline Geometry | Filtered | 0.0757 | 0.2717 | 0.7368 | 0.0030 |

Highlights:
- **Under pressure** slice: enriched contextual Brier 0.0537 / ECE 0.0078 vs provider 0.0600 / 0.0178 and baseline 0.0684 / 0.0480.
- **Trailing state**: enriched Brier 0.0528 and AUC 0.893 > filtered contextual (0.0577 / 0.839) and provider (0.0607 / 0.783).
- **Long range** (>20m) shows biggest lift: enriched Brier 0.0281 vs baseline 0.0306 and provider 0.0303 while boosting AUC from 0.605 (provider) to 0.816.
- **Close range** remains the toughest slice; contextual models still over-forecast a bit (ECE ~0.01–0.02) suggesting future calibration or feature tweaks around cutbacks and rebounds.

## 4. Reliability Insights
- Multi-line calibration plot: `outputs/modeling/cxg/modeling_charts/cxg_reliability_overlay.png`.
- Enriched contextual curve stays closest to the ideal diagonal across bins, particularly 0.45–0.85 probability range.
- Provider xG overstates probabilities above 0.65; baseline geometry underestimates mid-range (0.15–0.35) bins.

## 5. Supporting Artifacts
- Comparison bars: `outputs/modeling/cxg/modeling_charts/model_compare_{brier_mean,log_loss_mean,auc_mean}.png`.
- Metrics table CSV: `outputs/modeling/cxg/modeling_charts/model_metric_table.csv`.
- Contextual feature effect exports: `outputs/modeling/cxg/contextual_feature_effects{,_filtered,_enriched}.csv`.
- Reliability curves per run under `outputs/modeling/cxg/plots/`.

## 6. Batch Scoring & Inference
- **Artifacts:** `outputs/modeling/cxg/models/contextual_model.joblib` (enriched default) plus `contextual_model_filtered.joblib` and `baseline_geometry_model.joblib` for fallbacks.
- **Feature contract:** Feature lists are recorded in each artifact’s JSON (`contextual_model.json`, etc.). Always select columns in that order before calling `predict_proba`.
- **Example:**

```python
import json
import joblib
import pandas as pd

MODEL_DIR = "outputs/modeling/cxg/models"
model = joblib.load(f"{MODEL_DIR}/contextual_model.joblib")
meta = json.load(open(f"{MODEL_DIR}/contextual_model.json"))
features = meta["features"]["numeric"] + meta["features"]["binary"] + meta["features"]["categorical"]

df = pd.read_parquet("/path/to/new_cxg_dataset_enriched.parquet")
scores = model.predict_proba(df[features])[:, 1]
df.assign(cxg_pred=scores).to_parquet("/path/to/new_scores.parquet", index=False)
```

- **Data prep:** For new competitions run `build_cxg_dataset.py` → `enrich_cxg_with_submodels.py` to emit the enriched schema before inference.

### 6.0 Neutral Priors Workflow Cheat Sheet
- **Toggle PL training data:** Set `CXG_INCLUDE_PL1516_PRIORS=1` when running `train_finishing_bias_model.py` if you want those match-sides included. Omit or set to `0`/`false` to keep the WC/Euro-only baseline. Each run emits metrics in `outputs/modeling/cxg/submodels/finishing_bias_model_metrics*.json` so you can verify which mode was used (`include_pl1516_in_priors`).
- **Regenerate priors + enriched dataset:**
	```bash
	poetry run python -m opponent_adjusted.modeling.cxg.submodels.train_finishing_bias_model
	poetry run python -m opponent_adjusted.modeling.cxg.enrich_cxg_with_submodels
	```
	Save copies of the CSV/Parquet outputs if you need to swap between modes quickly (see `outputs/modeling/cxg/submodels/*_include_pl1516.csv`).
- **Train contextual model with a run label:**
	```bash
	CXG_CONTEXTUAL_RUN_NAME=neutral_priors_refresh \
		poetry run python -m opponent_adjusted.modeling.cxg.contextual_model
	```
	Repeat with `neutral_priors_refresh_plfb` (or any slug) once you rebuild priors that include PL match-sides.
- **Score downstream competitions:** call `prediction.run_pipeline` with `--model-label contextual_model_neutral_priors_refresh` (or `_plfb`) so the inference stage uses the matching artifact+feature contract.

### 6.1 Prediction Module Workflow
1. **Fetch & ingest** the target competition so the database mirrors StatsBomb open data:
	- `poetry run python scripts/fetch_statsbomb_subset.py --competition <id> --season <year>`
	- `poetry run python scripts/ingest_competitions.py && poetry run python scripts/ingest_matches.py`
	- `poetry run python scripts/ingest_events.py --competition <id>` (runs per match scope automatically)
	- `poetry run python scripts/normalize_events.py` to populate relational tables.
2. **Build features** using the same transforms as training:
	- `poetry run python scripts/build_shot_features.py --version-tag cxg_v1`
	- `poetry run python scripts/build_opponent_profiles.py --version-tag cxg_v1`
	- `poetry run python scripts/run_cxg_analysis.py --dataset-type enriched --output /tmp/new_dataset.parquet`
3. **Score & aggregate** with the new prediction runner:
	- `poetry run python -m opponent_adjusted.prediction.run_pipeline /tmp/new_dataset.parquet --tag <competition_slug>`
	- Outputs land in `outputs/modeling/cxg/prediction_runs/<tag-or-timestamp>/`:
	  - `scored_shots.parquet`: per-shot predictions (CxG + provider xG/goals references)
	  - `match_aggregates.csv`: one row per team per match with CxG/provider/goals/shot deltas
	  - `team_aggregates.csv`: season table with for/against splits, CxG deltas, and goal differences

The runner automatically selects the feature contract recorded in the model metadata, ensuring the same column order/calibration used during training. Override the artifact via `--model-label contextual_model_filtered` if you need to match an alternate training sample.

### 6.2 Premier League 2015/16 Case Study

**Objective.** Score every Premier League 2015/16 shot with the contextual model that was trained on World Cup + Euro data, without retraining anything, and compare team outputs to observed goals and provider xG.

**Methodology.**
1. **Ingest** StatsBomb competition 2 / season 27 (Premier League 15/16) using the standard fetch → ingest_* → normalize pipeline so shots/events exist in the relational store.
2. **Refresh features** for tag `v1` (`build_shot_features.py`, `build_opponent_profiles.py`). These populate the same feature columns used in training.
3. **Assemble datasets**
   - `poetry run python -m opponent_adjusted.modeling.cxg.build_cxg_dataset`
   - Filter to PL matches only and rerun the governance filters:
	 ```bash
	 poetry run python - <<'PY'
	 import pandas as pd
	 from pathlib import Path
	 from opponent_adjusted.db.session import session_scope
	 from opponent_adjusted.db.models import Match, Competition
	 from opponent_adjusted.modeling.cxg.data_filters import apply_modeling_filters
	 from opponent_adjusted.modeling.cxg.enrich_cxg_with_submodels import enrich_dataset

	 base = Path('outputs/modeling/cxg/cxg_dataset.parquet')
	 df = pd.read_parquet(base)
	 with session_scope() as session:
		 rows = (
			 session.query(Match.id, Competition.statsbomb_competition_id, Competition.season)
			 .join(Competition, Match.competition_id == Competition.id)
			 .all()
		 )
	 pl_ids = {mid for mid, comp, season in rows if comp == 2 and season.startswith('2015/2016')}
	 filtered, _ = apply_modeling_filters(df, allowed_match_ids=pl_ids)
	 out_dir = Path('outputs/modeling/cxg/prediction_runs/pl_2015_16')
	 out_dir.mkdir(parents=True, exist_ok=True)
	 filtered.to_parquet(out_dir / 'cxg_dataset_pl1516_filtered.parquet', index=False)
	 enrich_dataset(filtered).to_parquet(out_dir / 'cxg_dataset_pl1516_enriched.parquet', index=False)
	 PY
	 ```
4. **Score + aggregate** the enriched parquet:
   ```bash
   poetry run python -m opponent_adjusted.prediction.run_pipeline \
	   outputs/modeling/cxg/prediction_runs/pl_2015_16/cxg_dataset_pl1516_enriched.parquet \
	   --tag pl_2015_16_club
   ```

This emits PL-only artifacts under `outputs/modeling/cxg/prediction_runs/pl_2015_16_club/` (`scored_shots.parquet`, `match_aggregates.csv`, `team_aggregates.csv`).

### 6.3 Results & Interpretation (PL 2015/16)

- **Top underlying sides (CxG diff = CxG for − CxG against):** Arsenal (+29.5), Tottenham (+24.9), Liverpool (+20.5), Manchester City (+19.9), Leicester (+16.0), Southampton (+9.4). These clubs created far more contextual value than they conceded, even after accounting for opponent effects.
- **Relegation signals:** Aston Villa (−29.4), Sunderland (−16.4), Stoke (−16.5), Norwich (−15.0), Newcastle (−11.6), Watford (−10.1), Crystal Palace (−10.0) all conceded significantly more CxG than they generated.
- **Finishing vs expectation (Goals − CxG):** West Ham (+14.0), Man City (+11.5), Everton (+9.8), Spurs (+8.7), Liverpool (+8.6) ran hot relative to the contextual model, while Watford (−4.9), Crystal Palace (−1.9), Aston Villa (−1.7) under-finished.
- **Match-level diagnostics:** Each row of `match_aggregates.csv` contains CxG, goals, provider xG, and the deltas (`cxg_delta`, `provider_delta`) for both teams, enabling fixture-by-fixture storytelling (e.g., Leicester opening the campaign with ~2.23 CxG but no goals vs Bournemouth).

Because the contextual model was trained on neutralized WC/Euro data and simply applied to PL shots, differences highlight how Premier League chance quality compares to that international baseline without any retraining. The workflow above can be repeated for any competition by swapping the ingest filters and dataset slice.

#### Neutral vs PL-inclusive finishing priors
We ran the PL 2015/16 scoring pipeline twice — once with the WC/Euro-only finishing priors (`pl_2015_16_exclude`) and once with PL match-sides allowed into the finishing bias fit (`pl_2015_16_plfb`). Aggregate deltas are saved in `outputs/modeling/cxg/prediction_runs/pl_2015_16_bias_comparison.json` for reproducibility.

| Scope | Variant | CxG MAE | CxG RMSE | Provider MAE | Provider RMSE | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Match | Exclude PL | 0.740 | 0.939 | 0.797 | 0.996 | Baseline neutral priors trained on WC/Euro only |
| Match | Include PL | 0.740 | 0.939 | 0.797 | 0.996 | Identical match error; priors shift per-team calibration |
| Team season totals | Exclude PL | 3.63 | 4.44 | 3.61 | 4.32 | Goals − CxG bias: +0.53 |
| Team season totals | Include PL | **3.60** | **4.39** | 3.61 | 4.32 | Goals − CxG bias: +0.56 |

Key takeaways:
- Including PL match-sides in the finishing bias submodel barely changes match-level accuracy but softens the priors (finishing multipliers rise from 0.974 → 0.983 on average), making the contextual model less conservative for PL teams.
- Provider metrics stay constant between runs, so any change in MAE/RMSE is attributable to the neutral priors adjustments.
- Keep both runs handy if you need to explain differences in club narratives; the JSON comparison plus the `team_aggregates.csv` files contain the raw numbers cited above.

### 6.4 Visualization Toolkit
- **Script:** `poetry run python -m opponent_adjusted.prediction.plot_reports <team_csv> [--output-dir <dir>] [--top-n 12]`
- **Inputs:** the `team_aggregates.csv` produced by `run_pipeline` (club-only or mixed).
- **Outputs:** three PNGs per run (team totals grouped bar, finishing deltas, CxG vs provider scatter) saved next to the CSV or the provided directory.
- **Usage example:**
	```bash
	poetry run python -m opponent_adjusted.prediction.plot_reports \
			outputs/modeling/cxg/prediction_runs/pl_2015_16_club/team_aggregates.csv \
			--output-dir outputs/modeling/cxg/prediction_runs/pl_2015_16_club/charts \
			--top-n 12
	```
These quick-look charts make it easy to compare contextual vs provider expectations and highlight finishing variance for any prediction batch.

## 7. Next Steps
1. Wire slice/evaluation outputs into the monitoring dashboard (surface Brier/log-loss/ECE thresholds, highlight pressure & long-range slices).
2. Add bootstrap confidence intervals and/or temporal holdout runs if stakeholders request statistical significance checks.
3. Extend subgroup calibration reporting (assist category, set-piece phase) using the evaluation script framework.

## 8. Model Persistence Checklist
- Serialize trained scikit-learn pipelines (baseline + contextual variants) to `outputs/modeling/cxg/models/`.
- Store accompanying metadata JSON (training dataset version, feature lists, CV metrics, timestamp).
- Version model artifacts using semantic tags (`cxg_contextual_enriched_v1.joblib`).
- Document loading example for inference services.
