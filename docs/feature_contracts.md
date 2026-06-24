# Feature Contracts

Feature contracts define the expected shape of model-ready datasets before training, scoring, and API inference.

## Contracts

| Metric | Contract | Grain |
| --- | --- | --- |
| CxG | `configs/feature_contracts/cxg_v1.json` | one row per shot |
| CxA | `configs/feature_contracts/cxa_v1.json` | one row per attacking action in a shot-linked sequence |
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

CxA uses the public-facing name `sequence-adjusted CxA`. Shot quality labels should come from the CxG pipeline, not post-shot outcomes.

## CxT notes

CxT has a specific completion-model constraint: `xt_delta` must not be used in the completion model because it can encode the completion outcome. Completion and value-gain models use separate feature lists.

## Completion role

These contracts are not the final feature store implementation. They are guardrails that future data and modelling PRs should use when generating datasets, training models, and adding API inference.
