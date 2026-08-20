from opponent_adjusted.features.cxg.three_sixty_composite import (
    F15_FEATURES,
    derive_composite_360_context,
)


def test_exact_governed_f15_membership():
    assert len(F15_FEATURES) == 3
    assert set(F15_FEATURES) == {
        "transition_space_decay",
        "defensive_reset_index",
        "gk_setness_proxy",
    }


def test_all_null_when_components_missing():
    values = derive_composite_360_context({}, {}, None)
    assert all(v is None for v in values.values())


def test_transition_space_decay_requires_delta_and_positive_age():
    dynamic = {"nearest_defender_distance_delta": -2.0}
    values = derive_composite_360_context(dynamic, {}, possession_age_s=4.0)
    assert values["transition_space_decay"] == 0.5

    values_zero_age = derive_composite_360_context(dynamic, {}, possession_age_s=0.0)
    assert values_zero_age["transition_space_decay"] is None


def test_defensive_reset_index_penalizes_looser_compactness():
    dynamic_tight = {"rest_defence_reset_fraction": 0.8, "defensive_compactness_delta": -100.0}
    dynamic_loose = {"rest_defence_reset_fraction": 0.8, "defensive_compactness_delta": 1200.0}
    tight = derive_composite_360_context(dynamic_tight, {}, None)["defensive_reset_index"]
    loose = derive_composite_360_context(dynamic_loose, {}, None)["defensive_reset_index"]
    assert tight == 0.8
    assert loose < tight


def test_gk_setness_proxy_bounded_and_decreasing_with_displacement():
    still = derive_composite_360_context({"gk_total_displacement": 0.0}, {}, None)[
        "gk_setness_proxy"
    ]
    moved = derive_composite_360_context({"gk_total_displacement": 5.0}, {}, None)[
        "gk_setness_proxy"
    ]
    assert still == 1.0
    assert 0.0 < moved < still
