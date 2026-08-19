"""Event-only CxG E1-E6 contextual feature sub-contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from opponent_adjusted.features.cxg.geometry import shot_geometry

CXG_EVENT_CONTEXT_E1_E6_CONTRACT_ID = "cxg_event_context_e1_e6_v1"
E1_E6_FEATURES = (
    "score_difference_pre_shot",
    "match_elapsed_s",
    "possession_age_s",
    "possession_team_events_pre_shot",
    "possession_net_progression_sb",
    "possession_progression_speed_sb_s",
)

_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.(\d+))?$")


@dataclass(frozen=True)
class EventRecord:
    """Minimal governed Silver event facts for E1-E6 derivation."""

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


@dataclass(frozen=True)
class EventContext:
    """Derived E1-E6 values and explicit audit helpers for one shot."""

    event_id: str
    event_context_contract_id: str
    score_difference_pre_shot: int | None
    match_elapsed_s: float | None
    possession_age_s: float | None
    possession_team_events_pre_shot: int | None
    possession_net_progression_sb: float | None
    possession_progression_speed_sb_s: float | None
    possession_context_valid: bool
    possession_age_valid: bool
    possession_start_x_sb: float | None
    possession_start_y_sb: float | None


@dataclass(frozen=True)
class ScoreValidation:
    """Audit-only comparison of reconstructed periods 1-4 score to match metadata."""

    reconstructed_home_score: int
    reconstructed_away_score: int
    metadata_home_score: int | None
    metadata_away_score: int | None

    @property
    def matches_metadata(self) -> bool | None:
        if self.metadata_home_score is None or self.metadata_away_score is None:
            return None
        return (
            self.reconstructed_home_score == self.metadata_home_score
            and self.reconstructed_away_score == self.metadata_away_score
        )


@dataclass
class _PossessionState:
    start_clock_s: float | None
    start_x_sb: float | None
    start_y_sb: float | None
    has_location: bool
    prior_team_events: int


def event_clock_s(event: EventRecord) -> float | None:
    """Return absolute governed match clock seconds, retaining timestamp precision."""
    if not isinstance(event.minute, int) or not isinstance(event.second, int):
        return None
    if event.minute < 0 or event.second < 0:
        return None

    fractional_second = 0.0
    if event.timestamp is not None:
        match = _TIMESTAMP_RE.fullmatch(event.timestamp)
        if match is None:
            return None
        fraction = match.group(1)
        if fraction:
            fractional_second = float(f"0.{fraction}")
    return event.minute * 60.0 + event.second + fractional_second


def goal_beneficiary_team_id(event: EventRecord) -> int | None:
    """Resolve one scoring beneficiary from governed StatsBomb event semantics."""
    if event.period == 5:
        return None
    if event.event_type_name == "Shot" and event.outcome_name == "Goal":
        return event.team_id
    if event.event_type_name == "Own Goal For":
        return event.team_id
    return None


def _is_possession_team_event(event: EventRecord) -> bool:
    return (
        event.possession_id is not None
        and event.team_id is not None
        and event.possession_team_id is not None
        and event.team_id == event.possession_team_id
    )


def _ordered(events: Iterable[EventRecord]) -> list[EventRecord]:
    return sorted(events, key=lambda event: ((event.period or 0), event.event_index))


def derive_event_contexts(events: Iterable[EventRecord]) -> dict[str, EventContext]:
    """Derive E1-E6 for shots using only strictly preceding governed events."""
    scores: dict[int, int] = {}
    possession_states: dict[tuple[int, int, int], _PossessionState] = {}
    contexts: dict[str, EventContext] = {}

    for event in _ordered(events):
        clock_s = event_clock_s(event)
        is_shot = event.event_type_name == "Shot"
        possession_context_valid = _is_possession_team_event(event)

        if is_shot:
            score_difference = None
            if event.period != 5 and event.team_id is not None:
                score_difference = scores.get(event.team_id, 0) - sum(
                    goals for team_id, goals in scores.items() if team_id != event.team_id
                )

            start_x_sb = start_y_sb = None
            possession_age_s = None
            possession_team_events_pre_shot = None
            possession_net_progression_sb = None
            possession_progression_speed_sb_s = None
            possession_age_valid = False
            if possession_context_valid:
                key = (event.match_id, event.possession_id, event.team_id)
                state = possession_states.get(key)
                shot_location = shot_geometry(event.location_x, event.location_y)
                start_clock_s = state.start_clock_s if state is not None else clock_s
                start_x_sb = state.start_x_sb if state is not None and state.has_location else None
                start_y_sb = state.start_y_sb if state is not None and state.has_location else None
                if start_x_sb is None and shot_location.geometry_valid:
                    start_x_sb, start_y_sb = event.location_x, event.location_y
                possession_team_events_pre_shot = state.prior_team_events if state else 0
                if (
                    clock_s is not None
                    and start_clock_s is not None
                    and math.isfinite(clock_s)
                    and math.isfinite(start_clock_s)
                ):
                    age = clock_s - start_clock_s
                    if age >= 0 and math.isfinite(age):
                        possession_age_s = age
                        possession_age_valid = True
                if shot_location.geometry_valid and start_x_sb is not None:
                    possession_net_progression_sb = event.location_x - start_x_sb
                    if possession_age_s is not None and possession_age_s > 0:
                        possession_progression_speed_sb_s = (
                            possession_net_progression_sb / possession_age_s
                        )

            contexts[event.event_id] = EventContext(
                event_id=event.event_id,
                event_context_contract_id=CXG_EVENT_CONTEXT_E1_E6_CONTRACT_ID,
                score_difference_pre_shot=score_difference,
                match_elapsed_s=clock_s,
                possession_age_s=possession_age_s,
                possession_team_events_pre_shot=possession_team_events_pre_shot,
                possession_net_progression_sb=possession_net_progression_sb,
                possession_progression_speed_sb_s=possession_progression_speed_sb_s,
                possession_context_valid=possession_context_valid,
                possession_age_valid=possession_age_valid,
                possession_start_x_sb=start_x_sb,
                possession_start_y_sb=start_y_sb,
            )

        if _is_possession_team_event(event):
            key = (event.match_id, event.possession_id, event.team_id)
            state = possession_states.get(key)
            if state is None:
                state = _PossessionState(clock_s, None, None, False, 0)
                possession_states[key] = state
            location = shot_geometry(event.location_x, event.location_y)
            if not state.has_location and location.geometry_valid:
                state.start_x_sb = event.location_x
                state.start_y_sb = event.location_y
                state.has_location = True
            state.prior_team_events += 1

        beneficiary = goal_beneficiary_team_id(event)
        if beneficiary is not None:
            scores[beneficiary] = scores.get(beneficiary, 0) + 1

    return contexts


def validate_final_score(
    events: Iterable[EventRecord],
    *,
    home_team_id: int,
    away_team_id: int,
    home_score: int | None,
    away_score: int | None,
) -> ScoreValidation:
    """Audit-only periods 1-4 score reconstruction against match metadata."""
    scores: dict[int, int] = {}
    for event in _ordered(events):
        beneficiary = goal_beneficiary_team_id(event)
        if beneficiary is not None:
            scores[beneficiary] = scores.get(beneficiary, 0) + 1
    return ScoreValidation(
        reconstructed_home_score=scores.get(home_team_id, 0),
        reconstructed_away_score=scores.get(away_team_id, 0),
        metadata_home_score=home_score,
        metadata_away_score=away_score,
    )
