from opponent_adjusted.features.cxg.contracts import (
    CXG_CONTEXT_TAXONOMY_ID,
    EVENT_FAMILY_CANDIDATE_COUNTS,
    THREE_SIXTY_FAMILY_CANDIDATE_COUNTS,
    cxg_event_contextual_allowlist,
    contextual_candidate_names,
    event_candidate_names,
    event_candidate_names_for_families,
    event_families,
    three_sixty_candidate_names,
    three_sixty_families,
)


def test_taxonomy_identity_and_family_counts():
    assert CXG_CONTEXT_TAXONOMY_ID == "cxg_context_taxonomy_v3"
    assert len(event_families()) == 13
    assert len(three_sixty_families()) == 15
    assert len(event_families() + three_sixty_families()) == 28


def test_candidate_counts_and_declared_family_counts():
    event = event_families()
    three_sixty = three_sixty_families()

    assert len(event_candidate_names()) == 75
    assert len(three_sixty_candidate_names()) == 75
    assert len(contextual_candidate_names()) == 150
    assert {
        family.family_id: family.candidate_count for family in event
    } == EVENT_FAMILY_CANDIDATE_COUNTS
    assert {
        family.family_id: family.candidate_count for family in three_sixty
    } == THREE_SIXTY_FAMILY_CANDIDATE_COUNTS


def test_candidate_universes_are_unique_and_disjoint():
    event_candidates = event_candidate_names()
    three_sixty_candidates = three_sixty_candidate_names()

    assert len(set(event_candidates)) == len(event_candidates)
    assert len(set(three_sixty_candidates)) == len(three_sixty_candidates)
    assert set(event_candidates).isdisjoint(three_sixty_candidates)


def test_family_ids_are_ordered():
    assert [family.family_id for family in event_families()] == [f"E{i}" for i in range(1, 14)]
    assert [family.family_id for family in three_sixty_families()] == [
        f"F{i}" for i in range(1, 16)
    ]


def test_governed_methodology_flags_are_explicit():
    family_flags = {
        family.family_id: family.methodology_flags
        for family in event_families() + three_sixty_families()
    }

    assert "later_stage_not_first_pass_screening" in family_flags["F15"]
    assert "requires_explicit_actor_receiver_shooter_linkage" in family_flags["F10"]
    assert (
        "requires_explicit_sequence_depth_eligibility_and_coverage_metadata" in family_flags["F14"]
    )
    assert "derivation_parameters_not_yet_locked" in family_flags["E13"]


def test_event_only_cxg_allowlist_excludes_all_360_candidates():
    allowlist = cxg_event_contextual_allowlist()

    assert allowlist == event_candidate_names()
    assert set(allowlist).isdisjoint(three_sixty_candidate_names())


def test_e1_e6_membership_and_native_coordinate_renames():
    candidates = event_candidate_names_for_families(("E1", "E2", "E3", "E4", "E5", "E6"))

    assert len(candidates) == 34
    assert [len(family.candidates) for family in event_families()[:6]] == [7, 8, 4, 6, 3, 6]
    for legacy in (
        "goalward_progress_m",
        "recorded_event_path_length_m",
        "max_single_action_progress_m",
        "previous_action_progress_m",
    ):
        assert legacy not in event_candidate_names()
    for replacement in (
        "goalward_progress_sb",
        "recorded_event_path_length_sb",
        "max_single_action_progress_sb",
        "previous_action_progress_sb",
    ):
        assert event_candidate_names().count(replacement) == 1


def test_e7_e12_membership_preserves_deferred_e13_and_global_counts():
    candidates = event_candidate_names_for_families(("E7", "E8", "E9", "E10", "E11", "E12"))

    assert len(candidates) == 35
    assert [len(family.candidates) for family in event_families()[6:12]] == [6, 5, 7, 6, 5, 6]
    assert len(event_families()[12].candidates) == 6
    assert len(event_candidate_names()) == 75
    assert len(three_sixty_candidate_names()) == 75
    assert len(contextual_candidate_names()) == 150
