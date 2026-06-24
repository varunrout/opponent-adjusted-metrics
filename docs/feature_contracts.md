# Feature Contracts

Feature contracts define the minimum column expectations and leakage rules for each metric family.

They are intentionally simple. They provide a shared validation layer that can be used by training scripts, batch scoring jobs and API inference.

## Why contracts matter

Football event datasets often carry columns that are valid for analysis but invalid for modelling. For example, a post-shot outcome column can be useful in reporting but must not be used as an input feature for CxG. CxT has a similar risk where realised `xt_delta` can accidentally imply whether an action completed.

Contracts make these rules explicit.

## Implemented contracts

| Contract | Purpose |
| --- | --- |
| `cxg_base` | CxG training and scoring base contract. |
| `cxa_action` | CxA action-sequence modelling contract. |
| `cxt_action` | CxT completion and value-gain modelling contract. |

## Usage

```python
import pandas as pd
from opponent_adjusted.features.contracts import CXG_BASE_CONTRACT, validate_contract

df = pd.read_parquet("feature_store/cxg/shots.parquet")
validate_contract(df, CXG_BASE_CONTRACT)
```

For stricter validation:

```python
CXG_BASE_CONTRACT.validate(df, allow_extra=False)
```

## Contract principles

1. Required columns must be present.
2. Forbidden columns must not be present.
3. Target columns are tracked separately from feature columns.
4. Extra columns are allowed by default because training datasets often include IDs and audit fields.
5. Inference pipelines should use the ordered feature list saved in model metadata when available.

## Next steps

- Wire `cxg_base` into CxG dataset building and model training.
- Wire `cxa_action` into CxA sequence dataset generation.
- Wire `cxt_action` into the CxT leakage-fix PR.
- Save the active contract name and feature list into every model metadata JSON.
