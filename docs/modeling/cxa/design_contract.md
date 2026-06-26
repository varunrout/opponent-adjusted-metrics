# CxA Design And Contract

This document defines the design contract for contextual expected assist (CxA). It is a guardrail for future implementation work, not evidence that a final CxA model has been trained or validated.

## Definition

CxA estimates the expected chance-creation value of eligible attacking actions. In this repository, the first baseline target is the CxG value of a downstream shot created by the action inside a fixed event-data window. Actions that do not create an eligible shot receive zero target value.

CxA is event-data compatible and must work with StatsBomb Open Data. It must not require tracking data.

## Target

The design-stage target is `created_shot_cxg`.

- `shot_created`: binary indicator that the action created an eligible downstream shot.
- `created_shot_cxg`: CxG value of the first eligible downstream shot; zero when no eligible shot exists.
- `created_shot_id`: optional reference identifier for audit and attribution, not a training feature.

The preferred shot-quality source is the generated CxG output. A future baseline may use provider xG as an explicit fallback only when CxG is unavailable, and the fallback must be recorded in model metadata and validation reports.

## Eligible Actions

Eligible rows should represent attacking events that can create or materially progress toward shots:

- Passes, including crosses, through balls, cutbacks, switches and set-piece deliveries.
- Carries and dribbles that progress the ball or change shot-creation context.
- Ball receipts or pressure evasions only when they are represented cleanly enough to attribute action value.

Shots, goalkeeper events, fouls, stoppages, tactical shifts and post-shot events are excluded from CxA training rows.

## Attribution Logic

The baseline attribution window should be deterministic:

- Same team only.
- Same possession preferred.
- Maximum 5 downstream actions.
- Maximum 15 seconds from action to shot.
- First eligible downstream shot is the baseline target event.

Future PRs may add multi-action credit allocation, but the baseline should start with a single target per action so validation remains legible.

## Feature Families

Core features should be available at the action timestamp:

- Identity: action, event, sequence, match, possession, team and player identifiers.
- Location and movement: start/end coordinates, distance, angle, x/y progression, final-third and penalty-area entries.
- Action descriptors: action type, body part, pass height, cross/cutback/through-ball flags and carry/dribble indicators.
- Sequence context: action position, sequence length so far and seconds since possession start.
- Match context: minute, second, score state and play pattern.
- Optional opponent context: opponent defensive profile or nearby defensive-action proxies when available from event data.

Future generated datasets should follow `configs/feature_contracts/cxa_v1.json`.

## Leakage Risks

CxA is especially leakage-prone because its target depends on future events. Training features must not include:

- Created shot outcome or goal outcome.
- Post-shot xG.
- Future possession value.
- Future sequence length.
- Number or type of actions after the current action.
- Any feature derived from the shot after it happens, except explicit target/reference columns excluded from training.

Validation must group by `match_id` to reduce match-level leakage.

## Baseline Model Plan

The next implementation PR should build a reproducible baseline with:

- A model-ready action feature table under `feature_store/cxa/`.
- A simple, deterministic baseline model before richer sequence attribution.
- Model metadata that records target source, feature columns, split group and generated output paths.
- Prediction outputs at action grain.
- Player/team aggregates based on summed predicted CxA and CxA above baseline.

This plan does not require a CxA API endpoint.

## Validation Plan

Generated validation should report:

- Row count and eligible-action count.
- Shot creation rate.
- Mean target CxA and mean predicted CxA.
- Main metrics appropriate for continuous probability/value targets.
- Grouped validation by `match_id`.
- Slice metrics by action type, start/end zone, score state and pressure context where available.
- Baseline comparison against traditional assists, uncontextualized created-shot xG or a simple location-progression baseline when available.

If a baseline comparison is unavailable, validation should state that clearly rather than inventing values.

## Output Contract

Future generated outputs should be ignored by Git and live under:

```text
feature_store/cxa/
outputs/modeling/cxa/
outputs/modeling/cxa/models/
outputs/modeling/cxa/reports/
outputs/modeling/cxa/predictions/
outputs/modeling/cxa/aggregates/
```

Expected future files are defined in `configs/feature_contracts/cxa_v1.json`. This PR documents and tests the contract shape only; it does not require those files to exist.

## Limitations

Event-only CxA cannot observe off-ball movement, defender spacing, passing lanes or receiver separation directly. Pressure and defensive context are proxies. Results should be described as a reproducible event-data baseline until a future PR implements and validates the model.
