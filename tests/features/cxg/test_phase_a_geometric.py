from opponent_adjusted.analysis.odi.contracts import bucket_defender_role
from opponent_adjusted.features.cxg.phase_a_geometric import (
    NULL_REASON_FEWER_THAN_2_DEFENDERS,
    DefenderCandidate,
    _flip_to_own_attacking_frame,
    compute_shot_features,
    nearest_defenders,
    role_centroids,
)


def test_bucket_defender_role_every_observed_position_name():
    expected = {
        "Goalkeeper": "GK",
        "Center Back": "CB",
        "Left Center Back": "CB",
        "Right Center Back": "CB",
        "Left Back": "Fullback_WingBack",
        "Right Back": "Fullback_WingBack",
        "Left Wing Back": "Fullback_WingBack",
        "Right Wing Back": "Fullback_WingBack",
        "Center Midfield": "Midfield",
        "Center Attacking Midfield": "Midfield",
        "Center Defensive Midfield": "Midfield",
        "Left Midfield": "Midfield",
        "Right Midfield": "Midfield",
        "Left Attacking Midfield": "Midfield",
        "Right Attacking Midfield": "Midfield",
        "Left Center Midfield": "Midfield",
        "Right Center Midfield": "Midfield",
        "Left Defensive Midfield": "Midfield",
        "Right Defensive Midfield": "Midfield",
        "Center Forward": "Attack",
        "Left Center Forward": "Attack",
        "Right Center Forward": "Attack",
        "Left Wing": "Attack",
        "Right Wing": "Attack",
        "Secondary Striker": "Attack",
    }
    for position_name, bucket in expected.items():
        assert bucket_defender_role(position_name) == bucket, position_name


def test_bucket_defender_role_none_for_missing():
    assert bucket_defender_role(None) is None


def test_role_centroids_weighted_not_naive_mean():
    # Two position_names in the same bucket (CB) with very different sample sizes --
    # the centroid must be the true weighted mean, not an average of the two per-position
    # means (which would incorrectly give the rare position_name equal say).
    stats = {
        "Center Back": (900, 900 * 20.0, 900 * 40.0),  # mean (20, 40), n=900
        "Left Center Back": (100, 100 * 60.0, 100 * 80.0),  # mean (60, 80), n=100 (out of range y, just for math clarity)
    }
    centroids = role_centroids(stats)
    cx, cy = centroids["CB"]
    expected_x = (900 * 20.0 + 100 * 60.0) / 1000
    expected_y = (900 * 40.0 + 100 * 80.0) / 1000
    assert abs(cx - expected_x) < 1e-9
    assert abs(cy - expected_y) < 1e-9
    naive_mean_x = (20.0 + 60.0) / 2
    assert abs(cx - naive_mean_x) > 1.0  # confirms it's NOT the naive mean-of-means


def test_nearest_defenders_ranks_by_distance():
    candidates = (
        DefenderCandidate(position_name="Center Back", x=100.0, y=40.0),  # far
        DefenderCandidate(position_name="Right Back", x=110.0, y=42.0),  # nearest
        DefenderCandidate(position_name="Center Midfield", x=105.0, y=38.0),  # middle
    )
    first, second = nearest_defenders(112.0, 40.0, candidates)
    assert first.position_name == "Right Back"
    assert second.position_name == "Center Midfield"


def test_nearest_defenders_handles_fewer_than_two():
    assert nearest_defenders(100.0, 40.0, ()) == (None, None)
    one = (DefenderCandidate(position_name="Center Back", x=100.0, y=40.0),)
    first, second = nearest_defenders(105.0, 40.0, one)
    assert first is not None
    assert second is None


def test_compute_shot_features_fewer_than_two_defenders_null_not_fabricated():
    candidates = (DefenderCandidate(position_name="Center Back", x=100.0, y=40.0),)
    centroids = {"CB": (98.0, 40.0)}
    result = compute_shot_features(112.0, 40.0, candidates, centroids)
    assert result["nearest_defender_role"] == "CB"
    assert result["nearest_defender_zone_displacement"] is not None
    assert result["second_nearest_defender_role"] is None
    assert result["nearest_defender_gap"] is None
    assert result["nearest_defender_rank_null_reason"] == NULL_REASON_FEWER_THAN_2_DEFENDERS


def test_compute_shot_features_two_defenders_all_populated():
    candidates = (
        DefenderCandidate(position_name="Right Back", x=110.0, y=42.0),
        DefenderCandidate(position_name="Center Midfield", x=105.0, y=38.0),
    )
    centroids = {"Fullback_WingBack": (95.0, 42.0), "Midfield": (60.0, 38.0)}
    result = compute_shot_features(112.0, 40.0, candidates, centroids)
    assert result["nearest_defender_role"] == "Fullback_WingBack"
    assert result["second_nearest_defender_role"] == "Midfield"
    assert result["nearest_defender_gap"] is not None and result["nearest_defender_gap"] > 0
    assert result["nearest_defender_rank_null_reason"] is None


def test_flip_to_own_attacking_frame_is_180_degree_reflection():
    # A defender standing exactly at the shot-facing goal (120, 40) sits at their OWN goal
    # (0, 40) in their own team's attacking frame -- and the flip is its own inverse.
    assert _flip_to_own_attacking_frame(120.0, 40.0) == (0.0, 40.0)
    assert _flip_to_own_attacking_frame(0.0, 40.0) == (120.0, 40.0)
    x, y = 87.3, 22.1
    flipped_twice = _flip_to_own_attacking_frame(*_flip_to_own_attacking_frame(x, y))
    assert abs(flipped_twice[0] - x) < 1e-9 and abs(flipped_twice[1] - y) < 1e-9


def test_zone_displacement_uses_flipped_frame_not_raw_shot_frame():
    # A "CB" defender sitting deep near their own goal in the shot's frame (x=118, near
    # goal-line) is, once flipped into their own team's attacking frame, at x=2 -- right
    # where a CB's own centroid plausibly sits (near their own goal in their OWN frame).
    # Comparing the RAW (unflipped) shot-frame coordinate to that same centroid would
    # produce a huge, spurious ~116-unit displacement instead of a small one.
    candidates = (DefenderCandidate(position_name="Center Back", x=118.0, y=40.0),)
    centroids = {"CB": (5.0, 40.0)}
    result = compute_shot_features(60.0, 40.0, candidates, centroids)
    # flipped: (120-118, 80-40) = (2, 40); distance from (5, 40) is small (~3 native units)
    assert result["nearest_defender_zone_displacement"] < 10.0


def test_compute_shot_features_no_defenders():
    result = compute_shot_features(112.0, 40.0, (), {})
    assert result["nearest_defender_role"] is None
    assert result["nearest_defender_zone_displacement"] is None
    assert result["second_nearest_defender_role"] is None
    assert result["nearest_defender_gap"] is None
    assert result["nearest_defender_rank_null_reason"] == NULL_REASON_FEWER_THAN_2_DEFENDERS
