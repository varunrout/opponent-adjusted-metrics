import pandas as pd
import pytest

from opponent_adjusted.features.contracts import (
    CXG_BASE_CONTRACT,
    CXT_ACTION_CONTRACT,
    FeatureContractError,
    validate_contract,
)


def test_cxg_contract_accepts_minimal_valid_frame():
    frame = pd.DataFrame(
        [
            {
                "location_x": 104.0,
                "location_y": 40.0,
                "distance_to_goal": 16.0,
                "centrality": 0.0,
                "body_part": "Right Foot",
                "shot_type": "Open Play",
                "minute": 44,
                "under_pressure": False,
                "is_goal": 0,
            }
        ]
    )

    assert validate_contract(frame, CXG_BASE_CONTRACT).equals(frame)


def test_cxg_contract_rejects_missing_required_columns():
    frame = pd.DataFrame([{"location_x": 104.0}])

    with pytest.raises(FeatureContractError, match="missing required columns"):
        CXG_BASE_CONTRACT.validate(frame)


def test_cxg_contract_rejects_forbidden_leakage_columns():
    frame = pd.DataFrame(
        [
            {
                "location_x": 104.0,
                "location_y": 40.0,
                "distance_to_goal": 16.0,
                "centrality": 0.0,
                "body_part": "Right Foot",
                "shot_type": "Open Play",
                "minute": 44,
                "under_pressure": False,
                "is_goal": 0,
                "shot_outcome": "Goal",
            }
        ]
    )

    with pytest.raises(FeatureContractError, match="forbidden columns"):
        CXG_BASE_CONTRACT.validate(frame)


def test_contract_can_disallow_extra_columns():
    frame = pd.DataFrame(
        [
            {
                "location_x": 104.0,
                "location_y": 40.0,
                "distance_to_goal": 16.0,
                "centrality": 0.0,
                "body_part": "Right Foot",
                "shot_type": "Open Play",
                "minute": 44,
                "under_pressure": False,
                "is_goal": 0,
                "unexpected": 1,
            }
        ]
    )

    with pytest.raises(FeatureContractError, match="unexpected columns"):
        CXG_BASE_CONTRACT.validate(frame, allow_extra=False)


def test_cxt_feature_columns_exclude_targets():
    assert "completed" not in CXT_ACTION_CONTRACT.feature_columns
    assert "xt_delta" not in CXT_ACTION_CONTRACT.feature_columns
