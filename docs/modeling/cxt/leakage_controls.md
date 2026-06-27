# CxT Leakage Controls

This note records leakage controls for future CxT modelling work. The current
source of truth for the CxT design-stage contract is
[design_contract.md](design_contract.md).

## Core risk

CxT values moving the ball into more threatening pitch states. The core
leakage risk is using information that is only known after the action as an
input feature for the same action.

Future model designs may include completion, value-gain, state-value, or
opponent-adjusted components. Each component must keep row-level inputs limited
to information available at the action timestamp.

## Explicit rules

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

If a future implementation introduces `xt_delta`, it must not be used as a
completion-model input because it can encode the action outcome. Value targets
and value-map estimation may use historical outcomes offline, but the scored
action row cannot peek at its own future.

## Required validation before release

Before CxT is marked implemented, validation should confirm:

- model inputs exclude prohibited leakage fields
- start/end locations are available and valid
- zone or state value estimation is documented separately from action-level
  scoring features
- train/test splits are grouped by `match_id`
- slice metrics and sequence or possession summaries are regenerated

## Acceptance criteria for CxT completion

CxT can be considered implemented only when:

- the model config saved with the artefact shows no leakage-sensitive inputs
- tests assert forbidden columns are absent from model inputs
- generated threat grids, predictions, reports, and aggregates match the
  contract
- player and team CxT leaderboards are generated from the leakage-safe path
