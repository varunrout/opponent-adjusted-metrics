# CxT Design And Contract

## Definition

Contextual expected threat (CxT) values ball progression into more threatening pitch states. It is a territorial/progression metric: it asks how much an action improves the attacking state before a shot happens.

In v1, CxT is a baseline grid-threat implementation. It is not CxT+, Contextual CxT, Advanced CxT, OD-CxT, or OD-CxT+.

CxT is distinct from the other metric families:

- CxG = shot quality at the moment of a shot.
- CxA = chance creation actions that create or progress toward shots.
- CxT = territorial and threat progression from one ball state to another.

The baseline CxT path is implemented with a simple zone/grid threat value approach; CxT+ / Contextual / Advanced / OD-CxT remain future enhancements.

## Baseline CxT

The implemented v1 baseline uses:

```text
cxt_value = end_threat - start_threat
```

The model maps start and end locations into a deterministic pitch grid, looks up the start/end zone threat values, and assigns the difference to the action.

## CxT+

CxT+ is deferred after v1. It may add action-completion context, pressure, pass height, body part, or other event-data context known at action time. It is not implemented in the current baseline.

## Contextual CxT

Contextual CxT is deferred after v1. It may add match and possession context such as period, minute, score state, play pattern, prior action context, and opponent tendencies. It is not implemented in the current baseline.

## Advanced CxT

Advanced CxT is deferred after v1. It may estimate richer state values before and after an action rather than relying only on fixed grid threat values. It is not implemented in the current baseline.

## OD-CxT

OD-CxT means opponent defensive adjusted CxT. OD-CxT and OD-CxT+ are deferred after v1 and are not implemented in the current baseline.

## Input Contract

The v1 CxT pipeline uses real generated action features. It no longer persists synthetic fixture rows in production paths.

Primary command:

```bash
make cxt-baseline
```

The runner resolves generated inputs in this order:

1. `feature_store/cxt/progressions_featured.parquet`
2. `feature_store/cxt/progressions.parquet`
3. `feature_store/cxa/action_features.parquet`

The current v1 sample run used real generated action rows and produced 1,091,388 CxT action-threat predictions.

## Eligible Actions

The baseline covers ball-progression actions with start/end locations, including:

- pass
- carry
- dribble/progression rows when represented in the generated action features

Shots belong to CxG, not CxT. Foul, goalkeeper, injury, and administrative event types should not be treated as CxT progression actions.

## Feature Families

Required feature families are:

- Identifiers: action/event ID, match ID, team ID, player ID, and possession/sequence ID where available.
- Locations: `start_x`, `start_y`, `end_x`, and `end_y`.
- Zones: `start_zone` and `end_zone`.
- Threat values: `start_threat`, `end_threat`, and `cxt_value`.
- Action descriptors: action type and success/progression flags where available.

## Leakage Guardrails

Baseline CxT uses action type, identifiers, and start/end locations. Future shot or goal outcomes are not used as action-level inputs.

Prohibited action-level input fields include:

- `future_shot_xg`
- `future_shot_location`
- `future_goal`
- `future_shot_outcome`
- `next_action_is_shot`
- `actions_until_shot`
- `total_future_possession_length`
- `goal_outcome`
- `shot_outcome`

The key rule is simple: CxT may learn zone values from historical data, but a scored action row cannot peek at its own future.

## Current V1 Metrics

From `outputs/modeling/cxt/reports/metrics.json`:

| Metric | Value |
|---|---:|
| model version | `cxt-baseline-v1` |
| actions | 1,091,388 |
| players | 2,025 |
| teams | 74 |
| total CxT | 7,848.171 |
| mean CxT | 0.007191 |
| min CxT | -0.239401 |
| max CxT | 0.245658 |
| positive actions | 515,593 |
| negative actions | 264,073 |
| zero actions | 311,722 |

From `outputs/modeling/cxt/reports/interpretation_summary.json`:

| Interpretation metric | Value |
|---|---:|
| pass CxT | 6,737.874 |
| carry CxT | 1,110.297 |
| final-third entry CxT | 2,205.979 |
| box-entry CxT | 4,898.689 |
| progressive-action CxT | 4,947.281 |

## Validation Plan

Baseline validation should continue to check:

- Action row count.
- Required start/end location coverage.
- Valid pitch coordinates.
- No prohibited leakage fields in action-level inputs.
- Reconciliation of `cxt_value = end_threat - start_threat`.
- Player/team/sequence aggregates.
- Zone-transition and top-action interpretation reports.

## Output Contract

Generated files:

```text
feature_store/cxt/action_features.parquet
outputs/modeling/cxt/threat_grid.parquet
outputs/modeling/cxt/predictions/action_threat.parquet
outputs/modeling/cxt/aggregates/player_cxt.parquet
outputs/modeling/cxt/aggregates/team_cxt.parquet
outputs/modeling/cxt/aggregates/sequence_cxt.parquet
outputs/modeling/cxt/reports/metrics.json
outputs/modeling/cxt/reports/zone_transition_summary.csv
outputs/modeling/cxt/reports/top_actions.csv
outputs/modeling/cxt/reports/interpretation_summary.json
```

DB tables:

- `model_registry`: model row for `cxt-baseline-v1`.
- `action_threat_predictions`: 1,091,388 rows in the current v1 sample run.
- `aggregates_player`: CxT player aggregate rows.
- `aggregates_team`: CxT team aggregate rows.
- `aggregates_sequence`: CxT sequence aggregate rows.
- `evaluation_metrics`: 13 CxT metric rows in the current v1 sample run.

## Interpretation

Baseline CxT should be read as a location-threat movement model:

- Positive values indicate movement into more threatening grid zones.
- Negative values indicate movement away from threat.
- Zero values often mean the action stayed within the same grid-threat state.

This is useful for reviewing progression profiles at action, player, team, and sequence level. It is not a full possession-state value model.

## Deferred Variants Summary

The following are roadmap items and are not implemented in v1:

- CxT+
- Contextual CxT
- Advanced CxT
- OD-CxT
- OD-CxT+

Future variants may add richer action context, opponent defensive adjustment, state-value modelling, and more advanced validation. They must preserve the no-future-leakage rule.

## Limitations

- Event-only CxT cannot observe defensive shape, receiver separation, passing lanes, or off-ball movement.
- Grid values are explainable but coarse.
- The current model is not opponent-adjusted.
- Sequence aggregate persistence is idempotent by model/version and may not be a one-to-one mirror of every generated sequence file row.
- Baseline CxT is a reproducible v1 reporting surface, not a production-grade threat model.
