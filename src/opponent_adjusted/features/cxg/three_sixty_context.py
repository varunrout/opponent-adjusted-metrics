"""Top-level CxG+ F1-F15 orchestration entry point (cxg_360_context_v1)."""

from __future__ import annotations

from typing import Iterable

from opponent_adjusted.features.cxg.event_context import (
    EventRecord,
    _ordered,
    _valid_location,
    derive_event_contexts,
)
from opponent_adjusted.features.cxg.three_sixty_composite import (
    F15_FEATURES,
    derive_composite_360_context,
)
from opponent_adjusted.features.cxg.three_sixty_frame import Frame, orient_players
from opponent_adjusted.features.cxg.three_sixty_sequence import (
    F6_F14_FEATURES,
    derive_dynamic_360_context,
)
from opponent_adjusted.features.cxg.three_sixty_static import (
    F1_F5_FEATURES,
    derive_static_360_context,
)

ALL_360_FEATURES = F1_F5_FEATURES + F6_F14_FEATURES + F15_FEATURES


def derive_360_contexts(
    events: Iterable[EventRecord], frames: dict[str, Frame]
) -> dict[str, dict[str, object | None]]:
    """Derive all F1-F15 candidates for every shot in the governed event corpus."""
    events = list(events)
    base = derive_event_contexts(events)
    dynamic_by_shot = derive_dynamic_360_context(events, frames)

    matches: dict[int, list[EventRecord]] = {}
    for event in events:
        matches.setdefault(event.match_id, []).append(event)

    contexts: dict[str, dict[str, object | None]] = {}
    for match_events in matches.values():
        for shot in _ordered(match_events):
            if shot.event_type_name != "Shot":
                continue
            e1e6 = base[shot.event_id]
            static_values = {name: None for name in F1_F5_FEATURES}
            if e1e6.possession_context_valid:
                frame = frames.get(shot.event_id)
                if frame is not None and shot.team_id is not None and _valid_location(shot):
                    oriented = orient_players(frame.players, shot.team_id, shot.team_id)
                    if oriented is not None:
                        static_values = derive_static_360_context(
                            frame, oriented, shot.location_x, shot.location_y
                        )
            dynamic_values = dynamic_by_shot.get(
                shot.event_id, {name: None for name in F6_F14_FEATURES}
            )
            composite_values = derive_composite_360_context(
                dynamic_values, static_values, e1e6.value("possession_age_s")
            )
            contexts[shot.event_id] = {**static_values, **dynamic_values, **composite_values}
    return contexts
