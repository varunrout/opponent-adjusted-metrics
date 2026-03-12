"""CxT Feature Module.

Expected Threat (xT) and Contextual xT features for ball progression analysis.

Components:
- xt_model: Static 12x8 xT grid
- cxt_features: Contextual feature engineering
"""

from __future__ import annotations

from .xt_model import (
    XT_GRID,
    XT_GRID_X,
    XT_GRID_Y,
    get_zone,
    get_xt_value,
    get_xt_delta,
    add_xt_features,
)
from .cxt_features import (
    engineer_cxt_features,
    get_feature_columns,
    load_opponent_profiles,
)

__all__ = [
    "XT_GRID",
    "XT_GRID_X",
    "XT_GRID_Y",
    "get_zone",
    "get_xt_value",
    "get_xt_delta",
    "add_xt_features",
    "engineer_cxt_features",
    "get_feature_columns",
    "load_opponent_profiles",
]