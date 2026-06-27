# Feature Contracts

Feature contracts define the expected shape of model-ready datasets before training, scoring, and API inference.

## Contracts

| Metric | Contract | Grain |
| --- | --- | --- |
| CxG | `configs/feature_contracts/cxg_v1.json` | one row per shot |
| CxA | `configs/feature_contracts/cxa_v1.json` | one row per eligible attacking action with a fixed lookahead window for shot creation |
| CxT | `configs/feature_contracts/cxt_v1.json` | one row per progression action |

## What each contract defines

Each contract can include identity columns, target columns, required features, optional reference features, columns that must not be used for model training, nullable columns, the split group column, and methodology notes.

## Validation

Use:

```bash
poetry run python scripts/validate_feature_contract.py \
  --contract configs/feature_contracts/cxg_v1.json \
  --data feature_store/cxg/shots.parquet
```

Write a validation report:

```bash
poetry run python scripts/validate_feature_contract.py \
  --contract configs/feature_contracts/cxg_v1.json \
  --data feature_store/cxg/shots.parquet \
  --output outputs/reports/feature_contracts/cxg_v1_validation.json
```

The validator checks that required columns exist, excluded columns are absent, the split group column exists, and row and column counts are recorded.

## CxG notes

CxG rows must be available at shot time. Post-shot information should not enter training features. Validation should group by `match_id`.

## CxA notes

CxA is currently in design-contract stage. The contract defines the target, eligible actions, leakage guardrails, validation expectations, and future output paths. Shot quality labels should come from the CxG pipeline when available, not post-shot outcomes.

## CxT notes

CxT is currently in leakage-safe design-contract stage. The contract defines baseline CxT, CxT+, contextual CxT, advanced CxT, and OD-CxT roadmap variants, required location/value fields, future output paths, and fields that must not be used as action-level model inputs.

Future outcomes may be used to estimate historical zone or state values and to evaluate the metric. They must not be used as row-level action features for the action being scored.

## Completion role

These contracts are not the final feature store implementation. They are guardrails that future data and modelling PRs should use when generating datasets, training models, and adding API inference.
