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
- phase_control_score / phase_control_state / time_to_control (REDESIGNED
  2026-08-20, cxg_event_context_e13_v1 parameters below): these represent
  progression from an unstable/transition possession toward a SETTLED state,
  not a raw prior-action success rate (the prior design let a single
  successful first action register as "high control" at time_to_control=0,
  which is not football-semantically a settled possession). A prefix of the
  possession is `_settled` only once it has BOTH sufficient bulk (action
  count and elapsed age) AND is outside the live-regain transition window
  AND has a majority-successful, majority-comfortable (low-pressure) action
  history. `phase_control_score` is a continuous [0,1] mean of four
  normalized settledness components (age, action count, success rate,
  pressure comfort) at the shot-time prefix -- never just the success
  fraction alone. `phase_control_state` buckets the shot-time prefix as
  settled / developing / transition. `time_to_control` causally REPLAYS the
  possession prefix action-by-action and returns the elapsed time at the
  FIRST prefix where `_settled` genuinely holds (not the shot-time state
  backfilled to an earlier moment); null if the possession never reaches
  settledness before the shot. All three are null when there are zero prior
  actions (no possession history to assess).
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

# Settledness criterion parameters (versioned parameters of e13_v1). A prefix must clear
# ALL of these to count as "settled"; TRANSITION_WINDOW_S matches E8's transition_proxy
# window (15.0s) for consistency with the already-frozen transition definition.
MIN_ACTIONS_FOR_SETTLED = 3
MIN_AGE_FOR_SETTLED_S = 6.0
TRANSITION_WINDOW_S = 15.0
MIN_SUCCESS_RATE_FOR_SETTLED = 0.6
MAX_PRESSURE_SHARE_FOR_SETTLED = 0.5
# Normalizers for the continuous phase_control_score components (fixed, not target-tuned).
SETTLED_AGE_NORM_S = 12.0
SETTLED_ACTION_NORM = 5.0
CONTROL_DEVELOPING_THRESHOLD = 0.5


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


def _is_settled(
    action_count: int,
    age_s: float | None,
    live_regain: bool,
    success_rate: float,
    pressure_share: float,
) -> bool:
    """Fixed, versioned settledness criterion (see module docstring)."""
    if action_count < MIN_ACTIONS_FOR_SETTLED:
        return False
    if age_s is None or age_s < MIN_AGE_FOR_SETTLED_S:
        return False
    if live_regain and age_s <= TRANSITION_WINDOW_S:
        return False
    if success_rate < MIN_SUCCESS_RATE_FOR_SETTLED:
        return False
    if pressure_share > MAX_PRESSURE_SHARE_FOR_SETTLED:
        return False
    return True


def _control_trio(
    prior: list[EventRecord], live_regain: bool, start_clock: float | None
) -> dict[str, object | None]:
    """Compute phase_control_score/state/time_to_control via causal prefix replay."""
    out: dict[str, object | None] = {
        "phase_control_score": None,
        "phase_control_state": None,
        "time_to_control": None,
    }
    action_count = len(prior)
    if action_count == 0:
        return out

    success_count = 0
    pressure_count = 0
    time_to_control: float | None = None
    final_age: float | None = None
    for i, event in enumerate(prior, start=1):
        success_count += int(_success(event))
        pressure_count += int(event.under_pressure is True)
        clock = event_clock_s(event)
        age_s = clock - start_clock if clock is not None and start_clock is not None else None
        if i == action_count:
            final_age = age_s
        if time_to_control is None and age_s is not None and age_s >= 0:
            success_rate_i = success_count / i
            pressure_share_i = pressure_count / i
            if _is_settled(i, age_s, live_regain, success_rate_i, pressure_share_i):
                time_to_control = age_s
    out["time_to_control"] = time_to_control

    success_rate = success_count / action_count
    pressure_share = pressure_count / action_count
    age_component = min(1.0, final_age / SETTLED_AGE_NORM_S) if final_age is not None else 0.0
    action_component = min(1.0, action_count / SETTLED_ACTION_NORM)
    pressure_component = 1.0 - pressure_share
    score = (age_component + action_component + success_rate + pressure_component) / 4.0
    out["phase_control_score"] = score

    settled = _is_settled(action_count, final_age, live_regain, success_rate, pressure_share)
    still_in_transition_window = (
        live_regain and final_age is not None and final_age <= TRANSITION_WINDOW_S
    )
    out["phase_control_state"] = (
        "settled"
        if settled
        else (
            "transition"
            if still_in_transition_window
            else "developing" if score >= CONTROL_DEVELOPING_THRESHOLD else "transition"
        )
    )
    return out


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

            age = e1e6.value("possession_age_s")
            shot_clock = event_clock_s(shot)
            start_clock = shot_clock - age if shot_clock is not None and age is not None else None
            live_regain = e1e6.value("restart_vs_live_regain") == "live_regain"
            values.update(_control_trio(prior, live_regain, start_clock))
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
