"""Regression tests for CxT leakage controls."""

import pandas as pd

from opponent_adjusted.modeling.cxt.contextual_model import train_cxt_model


def test_cxt_training_excludes_xt_delta_from_model_inputs():
    """`xt_delta` is the value-gain target, not an input feature."""

    df = pd.DataFrame(
        [
            {
                "match_id": 1,
                "start_xt": 0.01,
                "xt_delta": 0.02,
                "minute_normalized": 0.10,
                "under_pressure": False,
                "is_progressive": True,
                "action_type": "Pass",
                "start_third": "middle",
                "macro_zone_start": "central",
                "success": 1,
            },
            {
                "match_id": 1,
                "start_xt": 0.02,
                "xt_delta": -0.01,
                "minute_normalized": 0.20,
                "under_pressure": True,
                "is_progressive": False,
                "action_type": "Pass",
                "start_third": "defensive",
                "macro_zone_start": "wide",
                "success": 0,
            },
            {
                "match_id": 2,
                "start_xt": 0.03,
                "xt_delta": 0.04,
                "minute_normalized": 0.30,
                "under_pressure": False,
                "is_progressive": True,
                "action_type": "Carry",
                "start_third": "middle",
                "macro_zone_start": "central",
                "success": 1,
            },
            {
                "match_id": 2,
                "start_xt": 0.04,
                "xt_delta": -0.01,
                "minute_normalized": 0.40,
                "under_pressure": True,
                "is_progressive": False,
                "action_type": "Carry",
                "start_third": "defensive",
                "macro_zone_start": "wide",
                "success": 0,
            },
            {
                "match_id": 3,
                "start_xt": 0.05,
                "xt_delta": 0.03,
                "minute_normalized": 0.50,
                "under_pressure": False,
                "is_progressive": True,
                "action_type": "Dribble",
                "start_third": "attacking",
                "macro_zone_start": "central",
                "success": 1,
            },
            {
                "match_id": 3,
                "start_xt": 0.06,
                "xt_delta": -0.02,
                "minute_normalized": 0.60,
                "under_pressure": True,
                "is_progressive": False,
                "action_type": "Dribble",
                "start_third": "middle",
                "macro_zone_start": "wide",
                "success": 0,
            },
            {
                "match_id": 4,
                "start_xt": 0.07,
                "xt_delta": 0.05,
                "minute_normalized": 0.70,
                "under_pressure": False,
                "is_progressive": True,
                "action_type": "Pass",
                "start_third": "attacking",
                "macro_zone_start": "central",
                "success": 1,
            },
            {
                "match_id": 4,
                "start_xt": 0.08,
                "xt_delta": -0.03,
                "minute_normalized": 0.80,
                "under_pressure": True,
                "is_progressive": False,
                "action_type": "Pass",
                "start_third": "middle",
                "macro_zone_start": "wide",
                "success": 0,
            },
        ]
    )

    model, _ = train_cxt_model(df, n_splits=2)

    assert "xt_delta" not in model.completion_features
    assert "xt_delta" not in model.gain_features
