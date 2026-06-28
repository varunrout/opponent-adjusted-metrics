# V1 Results Summary

This page summarizes the current post-PR43 v1 sample run. Values come from the local SQLite database, generated parquet files, and generated report JSON files.

These are reproducible sample-run results, not hard-coded guarantees for every future data subset.

## Dataset Coverage

From `outputs/reports/ingestion/db_status.json`:

| Area | Count |
|---|---:|
| competitions | 5 |
| teams | 74 |
| players | 2,038 |
| matches | 610 |
| raw events | 2,143,146 |
| normalized events | 2,143,146 |
| possessions | 110,432 |
| shots | 15,623 |
| shot features | 15,623 |
| action features | 1,091,388 |

Largest raw event types in the current run:

| Event type | Count |
|---|---:|
| Pass | 604,861 |
| Ball Receipt | 563,697 |
| Carry | 466,258 |
| Pressure | 184,749 |
| Shot | 15,623 |

## CxG Results

Model version: `cxg_contextual_20260628`

| Metric | Value |
|---|---:|
| shot rows | 15,623 |
| goals | 1,639 |
| goal rate | 0.1049 |
| mean predicted CxG | 0.1051 |
| Brier | 0.07325 |
| Log loss | 0.26110 |
| ROC AUC | 0.80920 |
| grouped validation matches | 606 |

Provider `statsbomb_xg` was available for comparison and scored ROC AUC 0.81957 in the current run. That makes it a useful benchmark for the simple CxG baseline.

CxG generated:

- 15,623 DB `shot_predictions` rows.
- 1,453 player aggregate file rows.
- 74 team aggregate file rows.

## CxA Results

Model version: `cxa_baseline_20260628`

| Metric | Value |
|---|---:|
| action rows | 1,091,388 |
| positive shot-created rows | 54,569 |
| positive rate | 0.05000 |
| mean predicted probability | 0.05000 |
| Brier | 0.04046 |
| Log loss | 0.15080 |
| ROC AUC | 0.85814 |
| total attributed CxA | 4,944.835 |
| mean CxA | 0.004531 |
| max CxA | 0.072891 |

The current attribution method is `simple_action_level_baseline_attribution`. Each action receives expected CxA value from the baseline model, with sequence/possession shares normalized within generated groups.

CxA generated:

- 1,091,388 DB `action_features` rows.
- 1,091,388 DB `action_predictions` rows.
- 2,168 player aggregate file rows.
- 74 team aggregate file rows.
- 133,791 sequence aggregate file rows.

## CxT Results

Model version: `cxt-baseline-v1`

| Metric | Value |
|---|---:|
| action rows | 1,091,388 |
| players | 2,025 |
| teams | 74 |
| total CxT | 7,848.171 |
| mean CxT | 0.007191 |
| min CxT | -0.239401 |
| max CxT | 0.245658 |
| positive actions | 515,593 |
| negative actions | 264,073 |
| zero actions | 311,722 |

Interpretation summary:

| Area | CxT |
|---|---:|
| pass CxT | 6,737.874 |
| carry CxT | 1,110.297 |
| final-third entry CxT | 2,205.979 |
| box-entry CxT | 4,898.689 |
| progressive-action CxT | 4,947.281 |

Baseline CxT is location-threat movement: `cxt_value = end_threat - start_threat`.

CxT generated:

- 1,091,388 DB `action_threat_predictions` rows.
- 2,168 player aggregate file rows.
- 74 team aggregate file rows.
- 1,212 sequence aggregate file rows.

## DB Persistence Summary

Current `model_registry`:

| Model | Version | Algorithm |
|---|---|---|
| cxg | `cxg_contextual_20260628` | `contextual_logistic` |
| cxa | `cxa_baseline_20260628` | `baseline_action_classifier` |
| cxt | `cxt-baseline-v1` | `baseline_grid_threat` |

Prediction rows by model:

| DB table | Model | Rows |
|---|---|---:|
| `shot_predictions` | CxG | 15,623 |
| `action_predictions` | CxA | 1,091,388 |
| `action_threat_predictions` | CxT | 1,091,388 |
| `evaluation_metrics` | CxG | 20 |
| `evaluation_metrics` | CxA | 45 |
| `evaluation_metrics` | CxT | 13 |

## What A Reviewer Should Learn

- The repository now has a reproducible event-to-model pipeline, not only isolated notebooks or static docs.
- CxG evaluates shot quality.
- CxA evaluates chance-creation action value.
- Baseline CxT evaluates threat added by ball progression.
- File outputs and DB persistence are both generated locally and ignored by Git.
- The Streamlit dashboard can inspect generated outputs when they exist and degrade gracefully when they do not.

## Limitations

- CxG is a simple contextual logistic baseline and provider xG remains a strong benchmark.
- CxA is baseline action-level attribution, not causal assist credit.
- CxT is baseline grid threat, not contextual/opponent-adjusted CxT.
- No v1 metric uses tracking data.
- CxT+, Contextual CxT, Advanced CxT, OD-CxT, and production deployment are deferred.
