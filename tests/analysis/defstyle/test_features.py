import pytest

from opponent_adjusted.analysis.defstyle.features import (
    ACTION_TYPE_TO_FEATURE,
    ACTION_TYPES,
    DUEL_SUBTYPE_AVAILABLE,
    MIN_ACTIONS,
    STYLE_FEATURES,
    action_shares,
    feature_vector,
    meets_threshold,
    total_actions,
)


def test_seven_action_types_one_feature_each():
    assert len(ACTION_TYPES) == 7
    assert len(STYLE_FEATURES) == 7
    assert len(set(STYLE_FEATURES)) == 7


def test_duel_is_a_single_combined_rate():
    # Duel sub-type (aerial/ground, won/lost) is not available in oam_core --
    # there is no duel outcome column and no qualifiers side-table. If that
    # ever changes this flag flips and the taxonomy gains sub-typed entries.
    assert DUEL_SUBTYPE_AVAILABLE is False
    assert ACTION_TYPE_TO_FEATURE["Duel"] == "duel_share"
    assert not any(f.startswith("duel_aerial") or f.startswith("duel_won") for f in STYLE_FEATURES)


def test_no_identity_or_xg_feature_leaks_into_the_style_vector():
    for banned in ("player_id", "team_id", "statsbomb_xg", "xg"):
        assert not any(banned in feature for feature in STYLE_FEATURES)


def test_shares_sum_to_one():
    counts = {
        "Pressure": 60,
        "Duel": 20,
        "Interception": 10,
        "Clearance": 5,
        "Block": 3,
        "Foul Committed": 1,
        "50/50": 1,
    }
    shares = action_shares(counts)
    assert shares is not None
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["pressure_share"] == pytest.approx(0.6)


def test_missing_action_types_count_as_zero():
    shares = action_shares({"Pressure": 30})
    assert shares is not None
    assert shares["pressure_share"] == pytest.approx(1.0)
    assert shares["duel_share"] == pytest.approx(0.0)


def test_below_threshold_returns_none_not_a_fallback_vector():
    assert action_shares({"Pressure": MIN_ACTIONS - 1}) is None
    assert action_shares({}) is None
    assert action_shares({"Pressure": MIN_ACTIONS}) is not None


def test_threshold_boundary_is_inclusive():
    assert meets_threshold(MIN_ACTIONS)
    assert not meets_threshold(MIN_ACTIONS - 1)


def test_untracked_event_types_do_not_inflate_the_denominator():
    # A caller passing a wider count dict must not have Pass/Shot counted.
    counts = {"Pressure": 30, "Pass": 900, "Shot": 40}
    assert total_actions(counts) == 30
    shares = action_shares(counts)
    assert shares is not None
    assert shares["pressure_share"] == pytest.approx(1.0)


def test_feature_vector_follows_canonical_order():
    counts = dict.fromkeys(ACTION_TYPES, 10)
    shares = action_shares(counts)
    assert feature_vector(shares) == [pytest.approx(1 / 7)] * 7
