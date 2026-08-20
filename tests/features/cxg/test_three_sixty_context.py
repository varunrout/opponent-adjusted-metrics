from opponent_adjusted.features.cxg.contracts import (
    cxg_event_contextual_allowlist,
    contextual_candidate_names,
    event_candidate_names,
    three_sixty_candidate_names,
)
from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.event_context_e13 import derive_e13_contexts
from opponent_adjusted.features.cxg.three_sixty_context import ALL_360_FEATURES, derive_360_contexts
from opponent_adjusted.features.cxg.three_sixty_frame import Frame, FramePlayer

VISIBLE_AREA_FULL = (0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0)


def event(event_id, index, **changes):
    values = dict(
        event_id=event_id,
        match_id=1,
        event_index=index,
        period=1,
        minute=0,
        second=index,
        timestamp=f"00:00:{index:02d}.000",
        event_type_name="Pass",
        outcome_name=None,
        team_id=1,
        possession_id=1,
        possession_team_id=1,
        location_x=60.0,
        location_y=40.0,
        player_id=1,
        play_pattern_name="Regular Play",
    )
    values.update(changes)
    return EventRecord(**values)


def shot(event_id="shot", index=9, **changes):
    changes.setdefault("event_type_name", "Shot")
    changes.setdefault("location_x", 110.0)
    return event(event_id, index, **changes)


def test_overall_taxonomy_is_150_and_disjoint():
    assert len(event_candidate_names()) == 75
    assert len(three_sixty_candidate_names()) == 75
    assert len(contextual_candidate_names()) == 150
    assert set(event_candidate_names()).isdisjoint(three_sixty_candidate_names())
    assert cxg_event_contextual_allowlist() == event_candidate_names()


def test_all_75_f_candidates_are_exposed_by_orchestrator():
    assert len(ALL_360_FEATURES) == 75
    assert set(ALL_360_FEATURES) == set(three_sixty_candidate_names())


def test_derive_360_contexts_returns_all_75_keys_per_shot():
    players = (
        player := FramePlayer(0, True, True, None, 100.0, 40.0),
        FramePlayer(1, False, None, False, 105.0, 40.0),
    )
    rows = [
        event("a", 1, second=1, location_x=60, location_y=40),
        shot(index=2, second=6, location_x=108, location_y=40),
    ]
    frames = {"shot": Frame("shot", 1, VISIBLE_AREA_FULL, players)}
    contexts = derive_360_contexts(rows, frames)
    assert set(contexts["shot"]) == set(ALL_360_FEATURES)
    assert player.actor is True


def test_e13_and_f_contexts_share_governed_event_ids_and_no_leakage_fields():
    rows = [
        event("a", 1, second=1, location_x=60, location_y=40),
        shot(index=2, second=6, location_x=108, location_y=40, outcome_name="Goal"),
    ]
    frames: dict = {}
    e13 = derive_e13_contexts(rows)
    f_context = derive_360_contexts(rows, frames)
    assert set(e13) == {"shot"}
    assert set(f_context) == {"shot"}
    leakage_fields = {"is_goal", "outcome_name", "statsbomb_xg", "end_x", "end_y"}
    assert leakage_fields.isdisjoint(e13["shot"].values)
    assert leakage_fields.isdisjoint(f_context["shot"])
