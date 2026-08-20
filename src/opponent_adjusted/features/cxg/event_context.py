"""Event-only governed CxG E1-E6 contextual feature derivation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from opponent_adjusted.features.cxg.contracts import event_candidate_names_for_families
from opponent_adjusted.features.cxg.geometry import shot_geometry

CXG_EVENT_CONTEXT_E1_E6_CONTRACT_ID = "cxg_event_context_e1_e6_v1"
E1_E6_FAMILY_IDS = ("E1", "E2", "E3", "E4", "E5", "E6")
E1_E6_FEATURES = event_candidate_names_for_families(E1_E6_FAMILY_IDS)
_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.(\d+))?$")
_PLAY_PATTERN_MAP = {
    "From Kick Off": "kick_off",
    "From Goal Kick": "goal_kick",
    "From Free Kick": "free_kick",
    "From Corner": "corner",
    "From Throw In": "throw_in",
    "From Keeper": "keeper",
    "From Counter": "counter",
    "Regular Play": "live_other",
}
_RESTART_TYPES = {"kick_off", "goal_kick", "free_kick", "corner", "throw_in"}


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    match_id: int
    event_index: int
    period: int | None
    minute: int | None
    second: int | None
    timestamp: str | None
    event_type_name: str | None
    outcome_name: str | None
    team_id: int | None
    possession_id: int | None
    possession_team_id: int | None
    location_x: float | None
    location_y: float | None
    player_id: int | None = None
    play_pattern_name: str | None = None
    card_name: str | None = None
    end_x: float | None = None
    end_y: float | None = None
    duration: float | None = None
    counterpress: bool | None = None
    under_pressure: bool | None = None
    action_outcome_name: str | None = None


@dataclass(frozen=True)
class EventContext:
    event_id: str
    event_context_contract_id: str
    values: dict[str, object | None]
    possession_context_valid: bool
    possession_age_valid: bool
    possession_start_x_sb: float | None
    possession_start_y_sb: float | None

    def value(self, name: str) -> object | None:
        return self.values[name]


@dataclass
class _PossessionState:
    start_clock_s: float | None
    start_x: float | None
    start_y: float | None
    start_type: str
    prior: list[EventRecord]


def event_clock_s(event: EventRecord) -> float | None:
    if not isinstance(event.minute, int) or not isinstance(event.second, int):
        return None
    if event.minute < 0 or event.second < 0:
        return None
    fraction = 0.0
    if event.timestamp is not None:
        match = _TIMESTAMP_RE.fullmatch(event.timestamp)
        if match is None:
            return None
        if match.group(1):
            fraction = float(f"0.{match.group(1)}")
    return event.minute * 60.0 + event.second + fraction


def goal_beneficiary_team_id(event: EventRecord) -> int | None:
    if event.period == 5:
        return None
    if event.event_type_name == "Shot" and event.outcome_name == "Goal":
        return event.team_id
    return event.team_id if event.event_type_name == "Own Goal For" else None


def _valid_location(event: EventRecord) -> bool:
    return shot_geometry(event.location_x, event.location_y).geometry_valid


def _same_possession_team(event: EventRecord) -> bool:
    return (
        event.possession_id is not None
        and event.team_id == event.possession_team_id
        and event.team_id is not None
    )


def _start_type(name: str | None) -> str:
    return _PLAY_PATTERN_MAP.get(name, "unknown")


def _ordered(events: Iterable[EventRecord]) -> list[EventRecord]:
    return sorted(events, key=lambda event: ((event.period or 0), event.event_index))


def _goal_distance(x: float, y: float) -> float:
    return math.hypot(120.0 - x, 40.0 - y)


def _manpower_diff(dismissed: dict[int, set[object]], team_id: int | None) -> int | None:
    if team_id is None:
        return None
    own = len(dismissed.get(team_id, set()))
    other = sum(len(players) for other_id, players in dismissed.items() if other_id != team_id)
    return -own + other


def _possession_values(
    event: EventRecord, state: _PossessionState | None, clock: float | None
) -> tuple[dict[str, object | None], bool, bool, float | None, float | None]:
    empty = {
        name: None
        for name in E1_E6_FEATURES
        if name
        not in {
            "score_diff",
            "game_state",
            "match_minute",
            "regulation_time_remaining",
            "manpower_diff",
            "late_game_trailing",
            "late_game_leading",
        }
    }
    if not _same_possession_team(event):
        return empty, False, False, None, None
    state = state or _PossessionState(clock, None, None, _start_type(event.play_pattern_name), [])
    shot_valid = _valid_location(event)
    start_x, start_y = state.start_x, state.start_y
    if start_x is None and shot_valid:
        start_x, start_y = event.location_x, event.location_y
    age = None if clock is None or state.start_clock_s is None else clock - state.start_clock_s
    age_valid = age is not None and math.isfinite(age) and age >= 0
    if not age_valid:
        age = None
    start_distance = (
        _goal_distance(start_x, start_y) if start_x is not None and start_y is not None else None
    )
    shot_distance = _goal_distance(event.location_x, event.location_y) if shot_valid else None
    progression = (
        start_distance - shot_distance
        if start_distance is not None and shot_distance is not None
        else None
    )
    prior = state.prior
    valid_chain = [(item.location_x, item.location_y) for item in prior if _valid_location(item)]
    if shot_valid:
        valid_chain.append((event.location_x, event.location_y))
    path = None
    if len(valid_chain) >= 2:
        path = sum(
            math.hypot(next_x - x, next_y - y)
            for (x, y), (next_x, next_y) in zip(valid_chain, valid_chain[1:])
        )
    eligible_actions = [
        item
        for item in prior
        if item.event_type_name in {"Pass", "Carry"}
        and _valid_location(item)
        and item.end_x is not None
        and item.end_y is not None
        and shot_geometry(item.end_x, item.end_y).geometry_valid
    ]
    action_progress = [
        _goal_distance(item.location_x, item.location_y) - _goal_distance(item.end_x, item.end_y)
        for item in eligible_actions
    ]
    action_count = len(prior)
    start_kind = state.start_type
    live = start_kind in {"keeper", "counter", "live_other"}
    restart = start_kind in _RESTART_TYPES
    values = {
        "possession_start_x": start_x,
        "possession_start_y": start_y,
        "possession_start_goal_distance": start_distance,
        "possession_start_zone": (
            None
            if start_x is None
            else (
                "defensive_third"
                if start_x < 40
                else "middle_third" if start_x < 80 else "final_third"
            )
        ),
        "possession_start_type": start_kind,
        "restart_vs_live_regain": "restart" if restart else "live_regain" if live else None,
        "high_regain": (
            start_x >= 80 if live and start_x is not None else False if restart else None
        ),
        "deep_regain": (
            start_x <= 40 if live and start_x is not None else False if restart else None
        ),
        "possession_age_s": age,
        "shot_within_5s_regain": age <= 5 if live and age is not None else None,
        "shot_within_10s_regain": age <= 10 if live and age is not None else None,
        "shot_within_15s_regain": age <= 15 if live and age is not None else None,
        "possession_action_count": action_count,
        "possession_pass_count": sum(item.event_type_name == "Pass" for item in prior),
        "possession_carry_count": sum(item.event_type_name == "Carry" for item in prior),
        "possession_dribble_count": sum(item.event_type_name == "Dribble" for item in prior),
        "unique_attackers_involved": len(
            {item.player_id for item in prior if item.player_id is not None}
        ),
        "recorded_receipt_count": sum(item.event_type_name == "Ball Receipt*" for item in prior),
        "avg_action_interval_s": age / action_count if age is not None and action_count else None,
        "last_action_interval_s": (
            clock - event_clock_s(prior[-1])
            if prior
            and clock is not None
            and event_clock_s(prior[-1]) is not None
            and clock >= event_clock_s(prior[-1])
            else None
        ),
        "actions_per_second": action_count / age if age is not None and age > 0 else None,
        "goalward_progress_sb": progression,
        "recorded_event_path_length_sb": path,
        "possession_directness_proxy": (
            progression / path
            if progression is not None and path is not None and path > 0
            else None
        ),
        "pace_to_goal": (
            progression / age if progression is not None and age is not None and age > 0 else None
        ),
        "progressive_actions_n": sum(
            progress / _goal_distance(item.location_x, item.location_y) >= 0.10
            for item, progress in zip(eligible_actions, action_progress)
            if _goal_distance(item.location_x, item.location_y) > 0
        ),
        "max_single_action_progress_sb": max(action_progress) if action_progress else None,
    }
    return values, True, age_valid, start_x, start_y


def derive_event_contexts(events: Iterable[EventRecord]) -> dict[str, EventContext]:
    """Derive E1-E6 contexts independently for every governed match."""
    matches: dict[int, list[EventRecord]] = {}
    for event in events:
        matches.setdefault(event.match_id, []).append(event)
    contexts: dict[str, EventContext] = {}
    for match_events in matches.values():
        contexts.update(_derive_match_event_contexts(match_events))
    return contexts


def _derive_match_event_contexts(events: Iterable[EventRecord]) -> dict[str, EventContext]:
    scores: dict[int, int] = {}
    dismissed: dict[int, set[object]] = {}
    possessions: dict[tuple[int, int], _PossessionState] = {}
    contexts: dict[str, EventContext] = {}
    for event in _ordered(events):
        clock = event_clock_s(event)
        if event.event_type_name == "Shot":
            score = (
                None
                if event.period == 5 or event.team_id is None
                else scores.get(event.team_id, 0)
                - sum(value for team, value in scores.items() if team != event.team_id)
            )
            state = (
                possessions.get((event.possession_id, event.team_id))
                if _same_possession_team(event)
                else None
            )
            values, valid, age_valid, start_x, start_y = _possession_values(event, state, clock)
            values.update(
                {
                    "score_diff": score,
                    "game_state": (
                        None
                        if score is None
                        else "trailing" if score < 0 else "drawing" if score == 0 else "leading"
                    ),
                    "match_minute": None if clock is None else clock / 60.0,
                    "regulation_time_remaining": (
                        None if event.period == 5 or clock is None else max(0.0, 5400.0 - clock)
                    ),
                    "manpower_diff": (
                        None if event.period == 5 else _manpower_diff(dismissed, event.team_id)
                    ),
                    "late_game_trailing": (
                        None
                        if event.period == 5 or clock is None or score is None
                        else clock >= 4500 and score < 0
                    ),
                    "late_game_leading": (
                        None
                        if event.period == 5 or clock is None or score is None
                        else clock >= 4500 and score > 0
                    ),
                }
            )
            contexts[event.event_id] = EventContext(
                event.event_id,
                CXG_EVENT_CONTEXT_E1_E6_CONTRACT_ID,
                values,
                valid,
                age_valid,
                start_x,
                start_y,
            )
        if _same_possession_team(event):
            key = (event.possession_id, event.team_id)
            state = possessions.setdefault(
                key, _PossessionState(clock, None, None, _start_type(event.play_pattern_name), [])
            )
            if state.start_x is None and _valid_location(event):
                state.start_x, state.start_y = event.location_x, event.location_y
            state.prior.append(event)
        if event.card_name in {"Red Card", "Second Yellow"} and event.team_id is not None:
            identity = event.player_id if event.player_id is not None else event.event_id
            dismissed.setdefault(event.team_id, set()).add(identity)
        beneficiary = goal_beneficiary_team_id(event)
        if beneficiary is not None:
            scores[beneficiary] = scores.get(beneficiary, 0) + 1
    return contexts
