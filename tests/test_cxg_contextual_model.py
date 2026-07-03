import pandas as pd

from opponent_adjusted.modeling.cxg.contextual_model import _filter_features


def test_filter_features_excludes_statsbomb_xg_from_training_selection():
    frame = pd.DataFrame(
        [
            {
                "shot_distance": 12.4,
                "shot_angle": 0.33,
                "statsbomb_xg": 0.21,
                "is_leading": 1.0,
                "chain_label": "slow",
            }
        ]
    )

    numeric, binary, categorical = _filter_features(frame)

    assert "shot_distance" in numeric
    assert "shot_angle" in numeric
    assert "statsbomb_xg" not in numeric
    assert "is_leading" in binary
    assert "chain_label" in categorical
