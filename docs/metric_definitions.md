# Metric Definitions

This document defines the metric families used by the project. It separates current implementation status from future metric intent.

## CxG: Contextual Expected Goals

CxG estimates the probability that a shot results in a goal using shot geometry and contextual features.

Current status:

- CxG has a reproducible baseline training, validation, and export runner.
- Generated CxG features and model outputs are not committed.
- The current implementation is a pragmatic baseline, not a final calibrated production model.

Primary commands:

```bash
poetry run python scripts/run_cxg_pipeline.py
poetry run python scripts/run_cxg_end_to_end.py
```

Current generated locations:

- `feature_store/cxg/`
- `outputs/modeling/cxg/`

Conceptual formulation:

```text
CxG = P(goal | shot_geometry, shot_context)
```

Typical feature groups:

- Geometry: distance, angle, centrality.
- Context: game state, period/minute, pressure flags where available.
- Baseline benchmark signals where available, such as provider xG.

Future refinement areas:

- Calibration beyond the current baseline.
- Monitoring and drift checks.
- Broader slice validation.
- Stable registry-backed aggregate serving.

## CxA: Contextual Expected Assists

CxA is planned as a chance-creation metric that credits passes and/or build-up actions for the expected shot value they help create.

Current status:

- CxA is not complete.
- Feature contracts and planning notes exist.
- Old generated CxA reports and completion summaries were removed to avoid overclaiming.

Conceptual formulation:

```text
CxA = P(shot created within window | action context) * E(CxG of created shot)
```

Future completion needs:

- A reproducible sequence/action dataset.
- Final window and attribution rules.
- Validation with match-grouped splits.
- Player/team aggregate outputs.
- CxA model card and limitations.

## CxT: Contextual Expected Threat

CxT is planned as an action-value metric for ball progressions that accounts for completion risk, expected value gain, and context.

Current status:

- CxT is not complete.
- Leakage controls are documented in `docs/modeling/cxt/leakage_controls.md`.
- Old generated CxT reports and completion summaries were removed to avoid overclaiming.

Conceptual formulation:

```text
CxT = P(action completes | pre-action context) * E(value gain | completed action, pre-action context)
```

Future completion needs:

- Verified exclusion of post-action leakage features from model inputs.
- Separate completion and value-gain validation.
- Regenerated slice metrics and player/team outputs.
- CxT model card and limitations.

## Dashboard And v1 Release

Dashboard integration and v1 packaging are not complete. Dashboard views should only be presented as current once they run against stable regenerated outputs and are covered by smoke checks.
