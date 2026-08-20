from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.three_sixty_context import derive_360_contexts
from opponent_adjusted.features.cxg.three_sixty_frame import Frame, FramePlayer

VISIBLE_AREA_FULL = (0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0)


def event(event_id, index, match_id=1, **changes):
    values = dict(
        event_id=event_id,
        match_id=match_id,
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


def shot(event_id, index, match_id=1, **changes):
    changes.setdefault("event_type_name", "Shot")
    changes.setdefault("location_x", 110.0)
    return event(event_id, index, match_id=match_id, **changes)


def test_shared_event_uuid_across_matches_does_not_leak_state():
    # Two independent matches happen to reuse the same event_id/frame key; each match's
    # shot must only ever see prior states from its OWN match's event list.
    players_m1 = (FramePlayer(0, teammate=False, actor=None, keeper=False, x=90.0, y=30.0),)
    players_m2 = (FramePlayer(0, teammate=False, actor=None, keeper=False, x=70.0, y=30.0),)
    rows = [
        event("shared_id", 1, match_id=1, location_x=60, location_y=40),
        shot("shot1", 2, match_id=1, location_x=108, location_y=40),
        event("shared_id_m2", 1, match_id=2, location_x=60, location_y=40),
        shot("shot2", 2, match_id=2, location_x=108, location_y=40),
    ]
    frames = {
        "shared_id": Frame("shared_id", 1, VISIBLE_AREA_FULL, players_m1),
        "shot1": Frame("shot1", 1, VISIBLE_AREA_FULL, players_m1),
        "shared_id_m2": Frame("shared_id_m2", 2, VISIBLE_AREA_FULL, players_m2),
        "shot2": Frame("shot2", 2, VISIBLE_AREA_FULL, players_m2),
    }
    contexts = derive_360_contexts(rows, frames)
    assert contexts["shot1"]["defenders_in_box"] is not None
    assert contexts["shot2"]["defenders_in_box"] is not None
