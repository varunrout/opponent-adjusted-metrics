# CxT Pre-Model Ball Progression Analysis

The ball progression diagnostic layer is the pre-model CxT analysis stage:

```text
raw / normalized events
-> progression / CxT feature engineering
-> CxT diagnostic analysis
-> CxT modelling decisions / threat-value construction
-> post-model CxT reporting
```

It studies pre-model action/progression features and transition support before
any threat predictions, aggregate outputs, leaderboards, or dashboard stories are
produced.

Run:

```bash
make analysis-cxt
```

The loader prefers the DB-backed `action_features` table and can fall back to
local pre-model parquet files such as `feature_store/cxt/progressions_featured.parquet`,
`feature_store/cxt/progressions.parquet`, or `feature_store/cxa/action_features.parquet`.
It does not read post-model CxT prediction outputs.

The report at `outputs/analysis/cxt/report.md` explains action coverage, spatial
coverage, feature distributions, target/proxy availability, progression structure,
feature redundancy, transition stability, slice stability, data quality, leakage
checks, and modelling recommendations.

If no supervised CxT target/proxy exists, the report explicitly says so and
recommends constructing a target such as `threat_delta`, `xt_delta`,
`future_shot_value`, or `possession_value_change` before supervised CxT modelling.

Transition stability is the core diagnostic: sparse start/end zone pairs may need
coarser zones, smoothing, or a different value-construction strategy before model
training.
