from opponent_adjusted.features.cxg.contracts import (
    OPPONENT_ADJUSTED_FAMILY_CANDIDATE_COUNTS,
    contextual_candidate_names,
    event_candidate_names,
    opponent_adjusted_candidate_names,
    opponent_adjusted_families,
    three_sixty_candidate_names,
)


def test_opponent_adjusted_family_registered():
    families = opponent_adjusted_families()
    assert len(families) == 1
    assert families[0].family_id == "opponent_adjusted"
    assert families[0].source_type == "opponent_adjusted"


def test_opponent_adjusted_candidates_match_task_spec():
    assert opponent_adjusted_candidate_names() == (
        "nearest_defender_odi",
        "mean_backline_odi",
        "gk_odi",
        "defensive_profile_cluster",
    )
    assert OPPONENT_ADJUSTED_FAMILY_CANDIDATE_COUNTS["opponent_adjusted"] == 4


def test_opponent_adjusted_disjoint_from_frozen_families():
    oa = set(opponent_adjusted_candidate_names())
    assert oa.isdisjoint(event_candidate_names())
    assert oa.isdisjoint(three_sixty_candidate_names())


def test_frozen_combiner_untouched_by_new_family():
    # contextual_candidate_names() is pinned at 150 (75 event + 75 three_sixty)
    # by frozen tests; opponent_adjusted must never be folded into it.
    assert len(contextual_candidate_names()) == 150
    assert set(opponent_adjusted_candidate_names()).isdisjoint(contextual_candidate_names())
