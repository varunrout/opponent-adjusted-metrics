# CxA Pre-Model Analysis

The CxA diagnostic analysis layer sits between action feature engineering and
model training:

```text
raw / normalized events
-> CxA action feature engineering
-> CxA diagnostic analysis
-> CxA modelling decisions / model training
-> post-model prediction and attribution reporting
```

It reads the pre-model `action_features` table and writes diagnostics under
`outputs/analysis/cxa/`. It does not read `action_predictions`, `model_registry`,
post-model CxA aggregates, leaderboards, dashboard outputs, or attribution reports.

Run:

```bash
make analysis-cxa
```

The report at `outputs/analysis/cxa/report.md` explains target usability, target
sparsity, action coverage, action-type signal, movement and spatial signal,
sequence-window stability, feature redundancy, slice stability, data quality,
leakage risks, and modelling recommendations.

Downstream-shot fields such as created-shot IDs, created-shot CxG/value, and
window timing/count fields are treated as target or reference fields. They help
diagnose target construction but should not enter model training as ordinary
candidate inputs.

Modelling should consume this report by encoding cleaning rules, excluding leakage
or reference columns, deciding how to pool rare action types, validating sparse
positive targets, and adding missing sequence-window fields when attribution
diagnostics are incomplete.
