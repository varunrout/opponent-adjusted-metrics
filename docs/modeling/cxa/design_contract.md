# CxA Design And Contract

This document defines the implemented v1 baseline contract for contextual expected assist (CxA). It should not be read as evidence of a final CxA attribution system.

## Definition

CxA estimates the expected chance-creation value of eligible attacking actions. In v1 it asks:

> Did this action help create a shot within a short same-team possession window, and what expected chance value should be attributed to it?

CxA is event-data compatible and works with StatsBomb Open Data. It must not require tracking data.

## Eligible Actions

The implemented v1 action feature builder uses:

- Pass
- Carry
- Dribble

Ball receipts are loaded as context/detail data but the current generated CxA action-feature table contains pass, carry, and dribble rows. Shots, goalkeeper events, fouls, stoppages, tactical shifts, and post-shot events are excluded from CxA training rows.

Current v1 sample action-feature rows: 1,091,388.

## Target

The v1 target is deterministic and leakage-controlled:

- Same team.
- Same possession.
- Maximum 5 downstream actions.
- Maximum 15 seconds from action to shot.
- First eligible downstream shot is the target event.

Generated target/reference columns:

- `shot_created`: binary indicator that the action created an eligible downstream shot.
- `created_shot_cxg`: value of the created downstream shot; zero when no eligible shot exists.
- `created_shot_id`: optional reference identifier for audit and attribution, not a training feature.

Current v1 sample target summary:

| Field | Value |
|---|---:|
| action rows | 1,091,388 |
| positive shot-created rows | 54,569 |
| positive rate | 0.050000 |
| positive mean created-shot CxG | 0.090648 |

## Attribution Logic

The implemented attribution method is simple and explicit:

- Actions are linked to the first eligible downstream shot inside the same-team/same-possession window.
- Model predictions produce action-level expected CxA values.
- `cxa_share` is normalized within generated sequence/possession groups.
- Attribution is baseline action-level credit, not causal assist credit.

## Feature Families

Core feature families are observable at the action timestamp:

- Identity: action, event, sequence, match, possession, team, and player identifiers.
- Location and movement: start/end coordinates, distance, angle, x/y progression.
- Action descriptors: action type, body part, pass height, cross/cutback/through-ball flags, carry/dribble indicators.
- Progression: final-third entry, penalty-area entry, zone 14 entry, switches of play.
- Sequence context: action position, sequence length so far, seconds since possession start.
- Match context: minute, second, score state, play pattern.
- Optional proxies: pressure flags and opponent defensive profile placeholders where available.

## Leakage Risks

Prohibited leakage fields remain excluded from model inputs:

- Created shot outcome or goal outcome.
- Post-shot xG.
- Future possession value.
- Future sequence length.
- Number/type of future actions.
- Any feature derived from a shot after it happens, except explicit target/reference columns excluded from training.

## Baseline Model Plan

The implemented v1 model is a logistic-regression baseline classifier:

- Model version: `cxa_baseline_20260628`
- Estimator: `logistic_regression`
- Target: `shot_created`
- Value column: `created_shot_cxg`
- Split: grouped by `match_id` where available

The model predicts shot-creation probability and converts it into baseline CxA value using the observed created-shot value scale.

## Current V1 Metrics

From `outputs/modeling/cxa/reports/metrics.json`:

| Metric | Value |
|---|---:|
| rows | 1,091,388 |
| positive count | 54,569 |
| positive rate | 0.050000 |
| mean predicted probability | 0.050004 |
| Brier | 0.040464 |
| Log loss | 0.150804 |
| ROC AUC | 0.858136 |
| baseline probability | 0.050000 |
| baseline CxA | 0.004532 |

The fold-level metrics are computed with grouped match splits. All current fold log-loss and ROC-AUC statuses are `computed`.

## Attribution Summary

From `outputs/modeling/cxa/reports/attribution_summary.json`:

| Attribution field | Value |
|---|---:|
| action count | 1,091,388 |
| sequence count | 107,020 |
| total attributed CxA | 4,944.835 |
| mean CxA | 0.004531 |
| max CxA | 0.072891 |
| high-value threshold | 0.004784 |

Attribution method:

```text
simple_action_level_baseline_attribution
```

Each action receives its baseline model expected CxA value. Sequence and possession shares are normalized within generated action groups. Downstream shot value is available in the current v1 sample run, so attribution includes observed shot-value references.

## Output Contract

Generated files:

```text
feature_store/cxa/action_features.parquet
feature_store/cxa/pipeline_metadata.json
outputs/modeling/cxa/models/baseline_model.joblib
outputs/modeling/cxa/models/baseline_model.json
outputs/modeling/cxa/reports/metrics.json
outputs/modeling/cxa/reports/attribution_summary.json
outputs/modeling/cxa/predictions/action_predictions.parquet
outputs/modeling/cxa/aggregates/player_cxa.parquet
outputs/modeling/cxa/aggregates/team_cxa.parquet
outputs/modeling/cxa/aggregates/sequence_cxa.parquet
```

DB tables:

- `action_features`: 1,091,388 rows in the current sample run.
- `model_registry`: model row for `cxa_baseline_20260628`.
- `action_predictions`: 1,091,388 rows in the current sample run.
- `aggregates_player`: CxA player aggregate rows.
- `aggregates_team`: CxA team aggregate rows.
- `aggregates_sequence`: CxA sequence aggregate rows.
- `evaluation_metrics`: 45 CxA metric rows in the current sample run.

## Validation Plan

Validation should continue to report:

- Row count and eligible-action count.
- Shot creation rate.
- Mean predicted probability/value.
- Brier, log loss, and ROC AUC when safe.
- Grouped validation by match.
- Slice summaries by action type and zone where available.
- Clear skip statuses where class-dependent metrics are unsafe.

## Limitations

- Event-only CxA cannot observe off-ball movement, defender spacing, passing lanes, or receiver separation directly.
- The baseline target depends on future shots, so leakage guardrails are essential.
- Current attribution is simple action-level baseline attribution, not causal assist credit.
- Future work may introduce CxA+, richer sequence credit, better defensive context, and more advanced chance-value attribution.
