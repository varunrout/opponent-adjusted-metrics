"""Event-only governed CxG E7-E12 contextual derivation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from opponent_adjusted.features.cxg.contracts import event_candidate_names_for_families
from opponent_adjusted.features.cxg.event_context import (
    EventRecord,
    _ordered,
    _same_possession_team,
    _valid_location,
    derive_event_contexts,
    event_clock_s,
)
from opponent_adjusted.features.cxg.geometry import shot_geometry

CXG_EVENT_CONTEXT_E7_E12_CONTRACT_ID = "cxg_event_context_e7_e12_v1"
E7_E12_FAMILY_IDS = ("E7", "E8", "E9", "E10", "E11", "E12")
E7_E12_FEATURES = event_candidate_names_for_families(E7_E12_FAMILY_IDS)
FIRST_PHASE_MAX_SECONDS = 10.0
FIRST_PHASE_MAX_PRIOR_ACTIONS = 3
WINDOW_S = 300.0
_ATTACKING_TYPES = {"Pass", "Carry", "Dribble", "Ball Receipt*", "Shot"}


@dataclass(frozen=True)
class ExtendedEventContext:
    event_id: str
    event_context_contract_id: str
    values: dict[str, object | None]


def _inside_final(event: EventRecord) -> bool:
    return _valid_location(event) and event.location_x >= 80.0


def _inside_box(event: EventRecord) -> bool:
    return _valid_location(event) and event.location_x >= 102.0 and 18.0 <= event.location_y <= 62.0


def _goal_distance(x: float, y: float) -> float:
    return math.hypot(120.0 - x, 40.0 - y)


def _end_observation(event: EventRecord) -> tuple[float, float, float | None] | None:
    if event.event_type_name not in {"Pass", "Carry"} or event.end_x is None or event.end_y is None:
        return None
    if not shot_geometry(event.end_x, event.end_y).geometry_valid:
        return None
    clock = event_clock_s(event)
    end_clock = clock
    if (
        event.duration is not None
        and math.isfinite(event.duration)
        and event.duration >= 0
        and clock is not None
    ):
        end_clock = clock + event.duration
    return event.end_x, event.end_y, end_clock


def _observations(
    events: list[EventRecord], shot: EventRecord
) -> list[tuple[float, float, float | None]]:
    observations: list[tuple[float, float, float | None]] = []
    for event in [*events, shot]:
        if _valid_location(event):
            observations.append((event.location_x, event.location_y, event_clock_s(event)))
        if event is not shot:
            end = _end_observation(event)
            if end is not None:
                observations.append(end)
    return observations


def _entry_clock(
    observations: list[tuple[float, float, float | None]], predicate, last: bool = False
) -> float | None:
    previous_inside: bool | None = None
    entries: list[float | None] = []
    for x, y, clock in observations:
        inside = predicate(x, y)
        if inside and previous_inside is not True:
            entries.append(clock)
        previous_inside = inside
    return (entries[-1] if last else entries[0]) if entries else None


def _elapsed(shot_clock: float | None, entry_clock: float | None) -> float | None:
    if shot_clock is None or entry_clock is None:
        return None
    value = shot_clock - entry_clock
    return value if math.isfinite(value) and value >= 0 else None


def _success(event: EventRecord) -> bool:
    if event.event_type_name == "Pass":
        return event.action_outcome_name is None
    if event.event_type_name == "Carry":
        return True
    if event.event_type_name == "Dribble":
        return event.action_outcome_name == "Complete"
    if event.event_type_name == "Ball Receipt*":
        return event.action_outcome_name != "Incomplete"
    return False


def _is_opponent_pressure(event: EventRecord, shot: EventRecord) -> bool:
    return (
        event.event_type_name == "Pressure"
        and event.team_id is not None
        and event.team_id != shot.team_id
        and event.possession_id == shot.possession_id
    )


def _previous_same_possession(events: list[EventRecord], shot: EventRecord) -> EventRecord | None:
    candidates = [event for event in events if event.possession_id == shot.possession_id]
    return candidates[-1] if candidates else None


def _territory_values(
    prior: list[EventRecord], shot: EventRecord, shot_clock: float | None
) -> tuple[dict[str, object | None], float | None, float | None]:
    observations = _observations(prior, shot)
    final_clock = _entry_clock(observations, lambda x, _y: x >= 80.0)
    first_box = _entry_clock(observations, lambda x, y: x >= 102.0 and 18.0 <= y <= 62.0)
    last_box = _entry_clock(observations, lambda x, y: x >= 102.0 and 18.0 <= y <= 62.0, last=True)
    seen_inside = False
    seen_exit_after_inside = False
    reentry = False
    for x, y, _clock in observations:
        inside = x >= 102.0 and 18.0 <= y <= 62.0
        if inside and seen_exit_after_inside:
            reentry = True
            break
        if inside:
            seen_inside = True
        elif seen_inside:
            seen_exit_after_inside = True
    return (
        {
            "final_third_entry_to_shot_s": _elapsed(shot_clock, final_clock),
            "first_box_entry_to_shot_s": _elapsed(shot_clock, first_box),
            "last_box_entry_to_shot_s": _elapsed(shot_clock, last_box),
            "final_third_actions_before_shot": sum(_inside_final(event) for event in prior),
            "box_actions_before_shot": sum(_inside_box(event) for event in prior),
            "box_exit_reentry": reentry,
        },
        first_box,
        final_clock,
    )


def _momentum(
    events: list[EventRecord], shot: EventRecord, shot_clock: float | None
) -> dict[str, object | None]:
    if shot_clock is None:
        return {
            name: None
            for name in E7_E12_FEATURES
            if name.startswith(("team_", "opp_", "event_", "territorial_"))
        }
    window = [
        event
        for event in events
        if event.period == shot.period
        and event_clock_s(event) is not None
        and 0 <= shot_clock - event_clock_s(event) <= WINDOW_S
    ]
    team = [event for event in window if event.team_id == shot.team_id]
    opposition = [
        event for event in window if event.team_id is not None and event.team_id != shot.team_id
    ]
    known_possessions = [event for event in window if event.possession_team_id is not None]
    attacking_team = [event for event in team if event.event_type_name in _ATTACKING_TYPES]
    attacking_opp = [event for event in opposition if event.event_type_name in _ATTACKING_TYPES]

    def mass(items: list[EventRecord]) -> float:
        return sum(event.location_x / 120.0 for event in items if _valid_location(event))

    team_mass, opp_mass = mass(attacking_team), mass(attacking_opp)
    total_mass = team_mass + opp_mass
    return {
        "team_shots_last_5m": sum(event.event_type_name == "Shot" for event in team),
        "opp_shots_last_5m": sum(event.event_type_name == "Shot" for event in opposition),
        "team_attacking_actions_last_5m": len(attacking_team),
        "opp_attacking_actions_last_5m": len(attacking_opp),
        "event_possession_share_last_5m": (
            sum(event.possession_team_id == shot.team_id for event in known_possessions)
            / len(known_possessions)
            if known_possessions
            else None
        ),
        "territorial_dominance_last_5m": (
            (team_mass - opp_mass) / total_mass if total_mass > 0 else None
        ),
    }


def _derive_match(events: list[EventRecord]) -> dict[str, ExtendedEventContext]:
    base = derive_event_contexts(events)
    contexts: dict[str, ExtendedEventContext] = {}
    ordered = _ordered(events)
    for index, shot in enumerate(ordered):
        if shot.event_type_name != "Shot":
            continue
        e1e6 = base[shot.event_id]
        values = {name: None for name in E7_E12_FEATURES}
        if e1e6.possession_context_valid:
            prior = [
                event
                for event in ordered[:index]
                if _same_possession_team(event)
                and event.possession_id == shot.possession_id
                and event.team_id == shot.team_id
            ]
            shot_clock = event_clock_s(shot)
            territory, first_box_clock, _final_clock = _territory_values(prior, shot, shot_clock)
            values.update(territory)
            live = e1e6.value("restart_vs_live_regain") == "live_regain"
            restart = e1e6.value("restart_vs_live_regain") == "restart"
            age = e1e6.value("possession_age_s")
            start_x = e1e6.value("possession_start_x")
            pace = e1e6.value("pace_to_goal")
            values.update(
                {
                    "transition_proxy": age <= 15.0 if live and age is not None else None,
                    "high_turnover_shot": (
                        start_x >= 80.0 and age <= 15.0
                        if live and start_x is not None and age is not None
                        else None
                    ),
                    "regain_height_speed_interaction": (
                        (start_x / 120.0) * pace
                        if live and start_x is not None and pace is not None
                        else None
                    ),
                    "regain_to_box_entry_s": (
                        _elapsed(first_box_clock, e1e6.value("possession_age_s") and 0)
                        if False
                        else None
                    ),
                    "pressures_faced_possession": sum(
                        _is_opponent_pressure(event, shot) for event in ordered[:index]
                    ),
                    "under_pressure_actions_n": sum(
                        event.under_pressure is True for event in prior
                    ),
                    "under_pressure_action_share": (
                        sum(event.under_pressure is True for event in prior) / len(prior)
                        if prior
                        else None
                    ),
                }
            )
            if live:
                start_clock = (
                    shot_clock - age if shot_clock is not None and age is not None else None
                )
                values["regain_to_box_entry_s"] = (
                    _elapsed(first_box_clock, start_clock) if first_box_clock is not None else None
                )
                possession_start = prior[0] if prior else shot
                possession_start_clock = event_clock_s(possession_start)
                if possession_start_clock is None:
                    values["counterpress_regain_proxy"] = None
                else:
                    values["counterpress_regain_proxy"] = any(
                        event.event_index < possession_start.event_index
                        and event.team_id == shot.team_id
                        and event.counterpress is True
                        and event.possession_team_id != shot.team_id
                        and event_clock_s(event) is not None
                        and 0 <= possession_start_clock - event_clock_s(event) <= 5
                        for event in ordered
                    )
            if restart:
                values["set_piece_category"] = e1e6.value("possession_start_type")
                values["seconds_since_restart"] = age
                values["actions_since_restart"] = len(prior)
                if len(prior) == 0:
                    phase = "restart_shot"
                elif (
                    age is not None
                    and age <= FIRST_PHASE_MAX_SECONDS
                    and len(prior) <= FIRST_PHASE_MAX_PRIOR_ACTIONS
                ):
                    phase = "first_phase"
                else:
                    phase = "second_phase"
                values["set_piece_phase"] = phase
                values["direct_vs_second_phase"] = (
                    "direct" if phase in {"restart_shot", "first_phase"} else "second_phase"
                )
            elif live:
                values["set_piece_category"] = "none"
            pressures = [event for event in ordered[:index] if _is_opponent_pressure(event, shot)]
            if shot_clock is not None:
                gaps = [
                    shot_clock - event_clock_s(event)
                    for event in pressures
                    if event_clock_s(event) is not None and 0 <= shot_clock - event_clock_s(event)
                ]
                values["pressures_last_5s"] = sum(gap <= 5 for gap in gaps)
                values["pressures_last_10s"] = sum(gap <= 10 for gap in gaps)
                values["time_since_last_pressure_s"] = min(gaps) if gaps else None
            previous = _previous_same_possession(ordered[:index], shot)
            if previous is not None:
                previous_clock = event_clock_s(previous)
                values["previous_action_type"] = previous.event_type_name
                values["previous_action_time_gap_s"] = (
                    shot_clock - previous_clock
                    if shot_clock is not None
                    and previous_clock is not None
                    and shot_clock >= previous_clock
                    else None
                )
                anchor = (
                    _end_observation(previous)
                    if previous.event_type_name in {"Pass", "Carry"}
                    else None
                )
                if anchor is None and _valid_location(previous):
                    anchor = (previous.location_x, previous.location_y, previous_clock)
                if anchor is not None and _valid_location(shot):
                    values["previous_action_distance_to_shot"] = math.hypot(
                        shot.location_x - anchor[0], shot.location_y - anchor[1]
                    )
                if (
                    previous.event_type_name in {"Pass", "Carry"}
                    and _valid_location(previous)
                    and _end_observation(previous) is not None
                ):
                    values["previous_action_progress_sb"] = _goal_distance(
                        previous.location_x, previous.location_y
                    ) - _goal_distance(anchor[0], anchor[1])
                values["previous_action_under_pressure"] = previous.under_pressure
                values["same_player_previous_action"] = (
                    previous.player_id == shot.player_id
                    if previous.player_id is not None and shot.player_id is not None
                    else None
                )
            values["successful_under_pressure_actions_n"] = sum(
                event.under_pressure is True and _success(event) for event in prior
            )
        values.update(_momentum(ordered[:index], shot, event_clock_s(shot)))
        contexts[shot.event_id] = ExtendedEventContext(
            shot.event_id, CXG_EVENT_CONTEXT_E7_E12_CONTRACT_ID, values
        )
    return contexts


def derive_extended_event_contexts(
    events: Iterable[EventRecord],
) -> dict[str, ExtendedEventContext]:
    matches: dict[int, list[EventRecord]] = {}
    for event in events:
        matches.setdefault(event.match_id, []).append(event)
    contexts: dict[str, ExtendedEventContext] = {}
    for match_events in matches.values():
        contexts.update(_derive_match(match_events))
    return contexts
