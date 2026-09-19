from opponent_adjusted.analysis.corrpca.features import (
    CXG_EVENT_QUALIFIED,
    CXG_PLUS_CATEGORICAL,
    CXG_PLUS_PRE_TRIM_NUMERIC,
    REDUNDANCY_THRESHOLD,
    REDUNDANT_PAIRS,
    dropped_features,
    final_cxg_event_pool,
    final_cxg_plus_numeric_pool,
)


def test_redundancy_threshold_is_085():
    assert REDUNDANCY_THRESHOLD == 0.85


def test_every_redundant_pair_meets_threshold():
    for track, a, b, r, dropped, reason in REDUNDANT_PAIRS:
        assert abs(r) >= REDUNDANCY_THRESHOLD, f"{track} {a}/{b} r={r} below threshold"
        assert dropped, f"{track} {a}/{b} has no drop action"
        assert reason


def test_cxg_event_pool_drops_previous_action_time_gap_s():
    pool = final_cxg_event_pool()
    assert "last_action_interval_s" in pool
    assert "previous_action_time_gap_s" not in pool
    assert len(pool) == len(CXG_EVENT_QUALIFIED) - 1


def test_cxg_event_pool_never_drops_for_weak_signal():
    # Every candidate not explicitly redundancy-dropped must survive, regardless of signal strength.
    pool = final_cxg_event_pool()
    dropped = dropped_features("cxg_event")
    for f in CXG_EVENT_QUALIFIED:
        if f not in dropped:
            assert f in pool


def test_cxg_plus_pool_size_and_odi_always_included():
    pool = final_cxg_plus_numeric_pool()
    # ODI trio must survive despite weak univariate signal -- non-negotiable inclusion principle.
    for odi_feature in ("nearest_defender_odi", "mean_backline_odi", "gk_odi"):
        assert odi_feature in pool
    assert len(pool) + len(CXG_PLUS_CATEGORICAL) == 18


def test_phase2_precedent_pairs_applied():
    dropped = dropped_features("cxg_plus")
    # defensive_centroid_x / defensive_line_depth precedent: keep line_depth.
    assert "defensive_centroid_x" in dropped
    assert "defensive_line_depth" not in dropped
    # defensive_compactness / defensive_hull_area precedent: drop both.
    assert "defensive_compactness" in dropped
    assert "defensive_hull_area" in dropped
    assert "defensive_width" not in dropped


def test_gk_distance_to_shooter_beats_shot_x_sb_in_plus_pool():
    # New pair surfaced by the Step 0 reverification, not in the original given baseline.
    pool = final_cxg_plus_numeric_pool()
    assert "gk_distance_to_shooter" in pool
    assert "shot_x_sb" not in pool  # dropped from CxG+ specifically, not from CxG's own pool


def test_shot_x_sb_untouched_in_cxg_event_pool():
    # The 360-cohort-specific redundancy trim must never affect CxG's own full-population pool.
    assert "shot_x_sb" in final_cxg_event_pool()


def test_no_duplicate_candidates_in_pre_trim_pool():
    assert len(CXG_PLUS_PRE_TRIM_NUMERIC) == len(set(CXG_PLUS_PRE_TRIM_NUMERIC))
