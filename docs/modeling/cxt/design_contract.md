# CxT Design and Contract

## Definition

Contextual expected threat (CxT) values moving the ball into more threatening pitch states. It is a territorial and progression metric: it asks how much an action improves the attacking state before a shot happens.

CxT is distinct from the other metric families:

- CxG = shot quality at the moment of a shot.
- CxA = chance creation actions that create or progress toward shots.
- CxT = territorial and threat progression from one ball state to another.

This document defines the leakage-safe CxT design and contract. The baseline CxT path is now implemented with a simple zone/grid threat value approach; CxT+ / Contextual / Advanced / OD-CxT remain future enhancements.

## Baseline CxT

The baseline CxT idea is:

```text
CxT = threat_value(end_location) - threat_value(start_location)
```

The baseline implementation uses event-data-compatible pass, carry, dribble, cross, and progressive movement actions. It maps start and end locations into a deterministic 12x8 grid and assigns each action the difference between the end-state and start-state threat.

## CxT+

CxT+ is a future enhancement of baseline CxT. It can account for action completion, action type, pressure, pass height, body part, and other context known at action time. CxT+ should still preserve the core idea that value comes from moving into more dangerous states, not from reading future outcomes from the same action row.

## Contextual CxT

Contextual CxT adds match and possession context available when the action occurs. Examples include period, minute, score state, play pattern, pressure, prior action context, and zone-level opponent tendencies when available.

Contextual features must be observable at the action timestamp. They must not include future shot labels, future possession length, or downstream outcome fields as action-level inputs.

## Advanced CxT

Advanced CxT is a future state-value formulation. Instead of only comparing fixed zone values, it can estimate `state_value_before` and `state_value_after` from richer event state. The action value is the improvement in estimated state value.

This variant may use future outcomes during offline value-estimation or evaluation, but the row-level model inputs must remain leakage-safe.

## OD-CxT

OD-CxT means opponent defensive adjusted CxT. It should adjust threat progression for defensive context and opponent suppression effects. OD-CxT+ is the future enhanced version combining opponent adjustment with the richer CxT+ feature set.

These variants are roadmap items. They are not implemented by the baseline CxT PR.

## Eligible Actions

The baseline contract covers ball-progression actions:

- pass
- carry
- dribble
- cross when represented
- progressive pass or carry when available

Shots belong to CxG, not CxT. Foul, goalkeeper, injury, and administrative event types should not be treated as CxT progression actions.

## Feature Families

Required identifiers should include match, team, player, possession, and action/event identifiers where available. Required location fields are `start_x`, `start_y`, `end_x`, and `end_y`.

Baseline value fields are:

- `start_zone`
- `end_zone`
- `start_threat`
- `end_threat`
- `cxt_value`

Future enhancement fields include `cxt_plus`, `state_value_before`, `state_value_after`, `advanced_cxt`, `od_cxt`, and `od_cxt_plus`.

## Leakage Guardrails

Future outcomes may be used to estimate zone or state values and to evaluate CxT quality. They must not be used as action-level model input features.

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

The key rule is simple: CxT can learn from historical futures when estimating the value map, but a scored action row cannot peek at its own future.

## Validation Plan

The baseline implementation should validate:

- row count
- valid pitch coordinates
- required start/end locations
- no prohibited leakage fields in model inputs
- baseline value reconciliation between start/end threat and `cxt_value`
- grouped validation by `match_id` where modelling is introduced
- slice summaries by action type and zone
- sequence or possession summaries

## Output Contract

Expected generated paths for the baseline CxT path and future CxT extensions:

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

These paths are generated outputs and remain ignored by Git.

## Limitations

This contract is event-data compatible and does not require tracking data. That makes it reproducible with StatsBomb Open Data, but it also limits what CxT can know about defensive shape, receiver separation, passing lanes, and off-ball movement.

Baseline CxT is implemented as an explainable, leakage-safe starting point, not a production-grade threat model. Player, team, sequence, zone-transition, top-action, and interpretation reports are baseline reporting surfaces. CxT+, Advanced CxT, OD-CxT, and OD-CxT+ are roadmap items rather than completed model surfaces.
