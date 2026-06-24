# CxT Leakage Controls

This note records the CxT leakage controls used by the contextual xT model.

## Core risk

CxT is a two-stage model:

1. completion model: estimates whether a progression action completes
2. value-gain model: estimates expected xT gain for completed actions

The key leakage risk is using a feature that is only known after the action outcome to predict completion.

## Explicit rule

`xt_delta` must not be used in the completion model.

Reason: `xt_delta` is only observed after the action has resolved. If it is used to predict completion, the model can learn from the answer rather than from pre-action context.

## Current code behaviour

The CxT training path removes `xt_delta` from:

- completion model features
- xT gain model input features

The value-gain model still uses `xt_delta` as the regression target for completed actions. That is valid because the model is learning to estimate the value gain, not using the value gain as an input.

## Required validation before release

Before CxT is marked complete, the final CxT report should confirm:

- completion features exclude `xt_delta`
- completion features exclude `success`, `completed`, `action_outcome`, and post-action labels
- value-gain features exclude `xt_delta`
- `xt_delta` is only used as the regression target
- train/test splits are grouped by `match_id`
- slice metrics are regenerated after the guardrail is applied

## Acceptance criteria for CxT completion

CxT can be considered complete only when:

- the model config saved with the artefact shows no leakage-sensitive inputs
- the evaluation report no longer includes unresolved leakage warnings
- tests assert the forbidden columns are absent from completion features
- player and team CxT leaderboards are regenerated from the corrected model
