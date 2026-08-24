from opponent_adjusted.analysis.defprofile.features import (
    CLUSTER_FEATURES,
    DEFENSIVE_360_CANDIDATES,
    DROPPED_REDUNDANT,
    LINE_SHAPE_360_CANDIDATES,
)


def test_cluster_features_is_fixed_size():
    assert len(CLUSTER_FEATURES) == 13


def test_cluster_features_has_no_duplicates():
    assert len(CLUSTER_FEATURES) == len(set(CLUSTER_FEATURES))


def test_dropped_redundant_columns_excluded_from_final_set():
    assert set(DROPPED_REDUNDANT).isdisjoint(CLUSTER_FEATURES)


def test_cluster_features_subset_of_candidates():
    candidates = set(DEFENSIVE_360_CANDIDATES) | set(LINE_SHAPE_360_CANDIDATES)
    assert set(CLUSTER_FEATURES).issubset(candidates)


def test_box_occupation_family_excluded():
    # F3 box-occupation columns were excluded for >95% null rate, not just redundancy.
    excluded_f3 = {
        "central_defenders_between_ball_and_goal",
        "defenders_goal_side",
        "box_numerical_balance",
        "attackers_in_box",
        "defenders_in_box",
    }
    assert excluded_f3.isdisjoint(CLUSTER_FEATURES)
