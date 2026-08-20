"""Event-only governed CxG E13 phase-of-play contextual derivation.

E13 methodology (cxg_event_context_e13_v1), locked parameters below. All six
candidates are computed strictly from events at or before the shot within the
shot's own possession chain (same eligibility gate as E1-E6/E7-E12: the shot
must belong to a valid, same-team possession). No 360 input, no shot outcome,
no StatsBomb xG, no post-shot event is read.

Feature summary (see docs/plan closure report for the full adversarial review):

- phase_location: zone of the MEAN valid location of prior possession actions
  (excludes the shot itself), distinguishing "where the buildup happened" from
  E2's possession_start_zone (origin only) and from the shot's own S location.
- possession_directness_score: mean, over eligible prior Pass/Carry actions, of
  each action's bounded net-progress-to-path-length ratio. This is a *mean of
  per-action ratios* (consistency of directness), distinct from E6's
  possession_directness_proxy which is a *ratio of sums* (global net
  efficiency: progression / total path length).
- phase_directness_bucket: fixed-threshold categorical bucket of the score.
- phase_control_score: fraction of prior possession actions classified as
  "successful" via the same _success predicate used by E8/E9 (reused, not
  duplicated). Normalized by action count, so it does not reward possession
  length on its own (see adversarial review).
- phase_control_state: fixed-threshold categorical bucket of the score, with
  an explicit null branch so "unknown" is never conflated with "low control".
- time_to_control: elapsed seconds from possession start to the first prior
  action classified as successful. Uses only events already known to precede
  the shot, so it is causal at shot-evaluation time.
"""

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
from opponent_adjusted.features.cxg.event_context_extended import _success
from opponent_adjusted.features.cxg.geometry import shot_geometry

CXG_EVENT_CONTEXT_E13_CONTRACT_ID = "cxg_event_context_e13_v1"
E13_FAMILY_IDS = ("E13",)
E13_FEATURES = event_candidate_names_for_families(E13_FAMILY_IDS)

# Fixed, non-target-tuned bucket thresholds (versioned parameters of e13_v1).
DIRECTNESS_DIRECT_THRESHOLD = 0.5
DIRECTNESS_REGRESSIVE_THRESHOLD = -0.5
CONTROL_HIGH_THRESHOLD = 0.75
CONTROL_MODERATE_THRESHOLD = 0.4


@dataclass(frozen=True)
class E13Context:
    event_id: str
    event_context_contract_id: str
    values: dict[str, object | None]

    def value(self, name: str) -> object | None:
        return self.values[name]


def _goal_distance(x: float, y: float) -> float:
    return math.hypot(120.0 - x, 40.0 - y)


def _eligible_progress_action(event: EventRecord) -> bool:
    return (
        event.event_type_name in {"Pass", "Carry"}
        and _valid_location(event)
        and event.end_x is not None
        and event.end_y is not None
        and shot_geometry(event.end_x, event.end_y).geometry_valid
    )


def _derive_match(events: list[EventRecord]) -> dict[str, E13Context]:
    base = derive_event_contexts(events)
    contexts: dict[str, E13Context] = {}
    ordered = _ordered(events)
    for index, shot in enumerate(ordered):
        if shot.event_type_name != "Shot":
            continue
        e1e6 = base[shot.event_id]
        values: dict[str, object | None] = {name: None for name in E13_FEATURES}
        if e1e6.possession_context_valid:
            prior = [
                event
                for event in ordered[:index]
                if _same_possession_team(event)
                and event.possession_id == shot.possession_id
                and event.team_id == shot.team_id
            ]

            valid_prior_locations = [
                (event.location_x, event.location_y) for event in prior if _valid_location(event)
            ]
            if valid_prior_locations:
                mean_x = sum(x for x, _ in valid_prior_locations) / len(valid_prior_locations)
                values["phase_location"] = (
                    "defensive_third"
                    if mean_x < 40.0
                    else "middle_third" if mean_x < 80.0 else "final_third"
                )

            ratios = []
            for event in prior:
                if not _eligible_progress_action(event):
                    continue
                distance = math.hypot(
                    event.end_x - event.location_x, event.end_y - event.location_y
                )
                if distance > 0:
                    progress = _goal_distance(event.location_x, event.location_y) - _goal_distance(
                        event.end_x, event.end_y
                    )
                    ratios.append(progress / distance)
            directness = sum(ratios) / len(ratios) if ratios else None
            values["possession_directness_score"] = directness
            if directness is not None:
                values["phase_directness_bucket"] = (
                    "direct"
                    if directness >= DIRECTNESS_DIRECT_THRESHOLD
                    else (
                        "regressive" if directness <= DIRECTNESS_REGRESSIVE_THRESHOLD else "mixed"
                    )
                )

            control_score = sum(_success(event) for event in prior) / len(prior) if prior else None
            values["phase_control_score"] = control_score
            if control_score is not None:
                values["phase_control_state"] = (
                    "high_control"
                    if control_score >= CONTROL_HIGH_THRESHOLD
                    else (
                        "moderate_control"
                        if control_score >= CONTROL_MODERATE_THRESHOLD
                        else "low_control"
                    )
                )

            age = e1e6.value("possession_age_s")
            shot_clock = event_clock_s(shot)
            start_clock = shot_clock - age if shot_clock is not None and age is not None else None
            first_control = next((event for event in prior if _success(event)), None)
            if first_control is not None and start_clock is not None:
                control_clock = event_clock_s(first_control)
                if control_clock is not None and control_clock >= start_clock:
                    values["time_to_control"] = control_clock - start_clock
        contexts[shot.event_id] = E13Context(
            shot.event_id, CXG_EVENT_CONTEXT_E13_CONTRACT_ID, values
        )
    return contexts


def derive_e13_contexts(events: Iterable[EventRecord]) -> dict[str, E13Context]:
    """Derive E13 contexts independently for every governed match."""
    matches: dict[int, list[EventRecord]] = {}
    for event in events:
        matches.setdefault(event.match_id, []).append(event)
    contexts: dict[str, E13Context] = {}
    for match_events in matches.values():
        contexts.update(_derive_match(match_events))
    return contexts
