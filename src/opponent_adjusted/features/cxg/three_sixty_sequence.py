"""CxG+ F6-F14: linked-frame sequence 360 dynamics (cxg_360_context_v1).

Governed linked-frame sequence contract (see section 24 of the E13/F1-F15 job
spec): for a shot, eligible prior states are strictly-prior events (same
match, same governed possession, same team as the shot) that (a) have a
linkable 360 frame, (b) have a resolvable frame-event team identity (for
orientation), and (c) have a strictly positive, finite elapsed time back to
the shot. Frames are never compared across matches or possessions, and a
"delta"/"rate" is only ever computed between two states whose underlying
family definition was itself eligible at both endpoints (never zero-filled).

Every rate below is `state_delta / elapsed_seconds` between two discrete,
linked freeze-frame snapshots -- an observed STATE-CHANGE PROXY, never a
claim of continuous velocity/acceleration (see three_sixty_frame.py docstring
and the closure report "scientific language" section).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from opponent_adjusted.features.cxg.contracts import three_sixty_candidate_names_for_families
from opponent_adjusted.features.cxg.event_context import (
    EventRecord,
    _ordered,
    _same_possession_team,
    _valid_location,
    derive_event_contexts,
    event_clock_s,
)
from opponent_adjusted.features.cxg.three_sixty_frame import Frame, orient_players
from opponent_adjusted.features.cxg.three_sixty_static import derive_static_360_context
from opponent_adjusted.features.cxg import three_sixty_geometry as geo

F6_F14_FAMILY_IDS = ("F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14")
F6_F14_FEATURES = three_sixty_candidate_names_for_families(F6_F14_FAMILY_IDS)

# --- Versioned fixed parameters (cxg_360_context_v1) ---------------------------------------
LINE_BREAK_BAND_NATIVE = 8.0  # fixed lateral band half-width around a ball-to-ball path proxy
LAST_N_ACTIONS_FOR_BYPASS = 3
LAST_N_FRAMES_FOR_COMPACTNESS = 3
MIN_SEQUENCE_SLOPE_POINTS = 3


@dataclass(frozen=True)
class _State:
    event: EventRecord
    elapsed_s: float
    static: dict[str, object | None]
    ball_x: float | None
    ball_y: float | None
    defenders: tuple


def _prior_same_possession(
    ordered: list[EventRecord], index: int, shot: EventRecord
) -> list[EventRecord]:
    return [
        event
        for event in ordered[:index]
        if _same_possession_team(event)
        and event.possession_id == shot.possession_id
        and event.team_id == shot.team_id
    ]


def _build_state(
    event: EventRecord, shot: EventRecord, shot_clock: float | None, frames: dict[str, Frame]
) -> _State | None:
    frame = frames.get(event.event_id)
    if frame is None or event.team_id is None or shot_clock is None:
        return None
    event_clock = event_clock_s(event)
    if event_clock is None:
        return None
    elapsed = shot_clock - event_clock
    if not math.isfinite(elapsed) or elapsed <= 0:
        return None
    if not _valid_location(event):
        return None
    oriented = orient_players(frame.players, event.team_id, shot.team_id)
    if oriented is None:
        return None
    static = derive_static_360_context(frame, oriented, event.location_x, event.location_y)
    defenders = geo.outfield_defenders(oriented)
    return _State(event, elapsed, static, event.location_x, event.location_y, defenders)


def _delta(current: object | None, prior: object | None) -> float | None:
    return (
        current - prior
        if isinstance(current, (int, float)) and isinstance(prior, (int, float))
        else None
    )


def _rate(delta: float | None, elapsed: float | None) -> float | None:
    if delta is None or elapsed is None or elapsed <= 0 or not math.isfinite(elapsed):
        return None
    return delta / elapsed


def _goal_exposure(static: dict[str, object | None]) -> float | None:
    occlusion = static.get("estimated_goalface_occlusion")
    return None if occlusion is None else 1.0 - occlusion


def _band_count(defenders: tuple, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> int:
    return sum(
        1
        for d in defenders
        if x_lo <= d.x <= x_hi
        and y_lo - LINE_BREAK_BAND_NATIVE <= d.y <= y_hi + LINE_BREAK_BAND_NATIVE
    )


def _derive_shot(
    shot: EventRecord,
    e1e6_restart_kind: str | None,
    ordered: list[EventRecord],
    index: int,
    frames: dict[str, Frame],
    shot_static: dict[str, object | None],
) -> dict[str, object | None]:
    values: dict[str, object | None] = {name: None for name in F6_F14_FEATURES}
    shot_clock = event_clock_s(shot)
    prior = _prior_same_possession(ordered, index, shot)

    states = [
        s for s in (_build_state(e, shot, shot_clock, frames) for e in prior) if s is not None
    ]
    states.sort(key=lambda s: s.elapsed_s, reverse=True)  # farthest-in-past first, shot last

    latest_prior = states[-1] if states else None  # smallest elapsed = most recent prior state

    # F6, F8, F9, F13: current shot minus latest eligible prior linked frame -----------------
    if latest_prior is not None:
        elapsed = latest_prior.elapsed_s
        for current_name, prior_name, out_name in (
            ("defensive_line_depth", "defensive_line_depth", "defensive_line_depth_delta"),
            ("defensive_width", "defensive_width", "defensive_width_delta"),
            ("defensive_length", "defensive_length", "defensive_length_delta"),
            ("defensive_compactness", "defensive_compactness", "defensive_compactness_delta"),
            ("defensive_hull_area", "defensive_hull_area", "defensive_hull_area_delta"),
        ):
            values[out_name] = _delta(
                shot_static.get(current_name), latest_prior.static.get(prior_name)
            )
        values["defensive_line_state_change_rate"] = _rate(
            values["defensive_line_depth_delta"], elapsed
        )

        cx_cur, cy_cur = shot_static.get("defensive_centroid_x"), shot_static.get(
            "defensive_centroid_y"
        )
        cx_prev, cy_prev = latest_prior.static.get("defensive_centroid_x"), latest_prior.static.get(
            "defensive_centroid_y"
        )
        if None not in (cx_cur, cy_cur, cx_prev, cy_prev):
            values["defensive_centroid_delta"] = math.hypot(cx_cur - cx_prev, cy_cur - cy_prev)

        for current_name, out_name in (
            ("defenders_goal_side", "defenders_goal_side_delta"),
            ("defenders_in_box", "defenders_in_box_delta"),
            ("attackers_in_box", "attackers_in_box_delta"),
            ("box_numerical_balance", "box_numerical_balance_delta"),
        ):
            values[out_name] = _delta(
                shot_static.get(current_name), latest_prior.static.get(current_name)
            )
        values["local_density_delta"] = _delta(
            shot_static.get("local_defensive_density"),
            latest_prior.static.get("local_defensive_density"),
        )
        values["local_density_state_change_rate"] = _rate(values["local_density_delta"], elapsed)

        values["nearest_defender_distance_delta"] = _delta(
            shot_static.get("nearest_defender_distance"),
            latest_prior.static.get("nearest_defender_distance"),
        )
        values["nearest_defender_distance_state_change_rate"] = _rate(
            values["nearest_defender_distance_delta"], elapsed
        )

        gk_x_cur, gk_y_cur = shot_static.get("gk_x"), shot_static.get("gk_y")
        gk_x_prev, gk_y_prev = latest_prior.static.get("gk_x"), latest_prior.static.get("gk_y")
        if None not in (gk_x_cur, gk_y_cur, gk_x_prev, gk_y_prev):
            values["gk_lateral_displacement"] = gk_y_cur - gk_y_prev
            gk_depth_cur = shot_static.get("gk_depth")
            gk_depth_prev = latest_prior.static.get("gk_depth")
            values["gk_depth_displacement"] = _delta(gk_depth_cur, gk_depth_prev)
            values["gk_total_displacement"] = math.hypot(gk_x_cur - gk_x_prev, gk_y_cur - gk_y_prev)
            values["gk_lateral_state_change_rate"] = _rate(
                values["gk_lateral_displacement"], elapsed
            )
            values["gk_depth_state_change_rate"] = _rate(values["gk_depth_displacement"], elapsed)

        for current_name, prior_name, out_name in (
            (
                "estimated_goalface_occlusion",
                "estimated_goalface_occlusion",
                "estimated_goalface_occlusion_delta",
            ),
            ("shot_corridor_occlusion", "shot_corridor_occlusion", "shot_corridor_occlusion_delta"),
            ("visible_goal_angle_proxy", "visible_goal_angle_proxy", "visible_goal_angle_delta"),
        ):
            values[out_name] = _delta(
                shot_static.get(current_name), latest_prior.static.get(prior_name)
            )

    # F10: shooter/receiver space evolution, explicit-linkage only ----------------------------
    if shot.player_id is not None:
        shooter_states = [s for s in states if s.event.player_id == shot.player_id]
        receipt_states = [s for s in shooter_states if s.event.event_type_name == "Ball Receipt*"]
        pre_shot_receiver_space = None
        if receipt_states:
            most_recent_receipt = min(receipt_states, key=lambda s: s.elapsed_s)
            pre_shot_receiver_space = most_recent_receipt.static.get("actor_space")
        values["pre_shot_receiver_space"] = pre_shot_receiver_space

        previous_linked = None
        if shooter_states:
            most_recent = min(shooter_states, key=lambda s: s.elapsed_s)
            previous_linked = most_recent.static.get("actor_space")
        values["shooter_space_previous_linked_event"] = previous_linked

        if pre_shot_receiver_space is not None and previous_linked is not None:
            values["shooter_space_change"] = previous_linked - pre_shot_receiver_space

        space_states = [
            (s.static.get("actor_space"), s.elapsed_s)
            for s in shooter_states
            if s.static.get("actor_space") is not None
        ]
        if space_states:
            max_space, max_elapsed = max(space_states, key=lambda item: (item[0], -item[1]))
            values["time_since_shooter_max_linkable_space"] = max_elapsed

    # F11: defensive bypass / line-break proxies -------------------------------------------
    # Each action's proxy uses the defender snapshot from ITS OWN eligible linked frame (via
    # `states`), never the shot's own (later) frame -- using a later state as if it were the
    # defensive layer an earlier action faced would be a future-state leakage. An action with
    # no eligible linked frame is simply excluded (null / not counted), per an explicit
    # sparse-but-honest contract; it is never backfilled from the shot's snapshot.
    state_by_event_id = {s.event.event_id: s for s in states}

    def _defenders_for(event: EventRecord) -> tuple | None:
        state = state_by_event_id.get(event.event_id)
        return state.defenders if state is not None and state.defenders else None

    if prior:
        last_action = prior[-1]
        last_action_defenders = _defenders_for(last_action)
        if last_action_defenders and _valid_location(last_action):
            x_lo, x_hi = sorted((last_action.location_x, shot.location_x))
            y_lo, y_hi = sorted((last_action.location_y, shot.location_y))
            values["defensive_layer_bypass_proxy_last_action"] = _band_count(
                last_action_defenders, x_lo, x_hi, y_lo, y_hi
            )
            deepest_x = min(d.x for d in last_action_defenders)
            deepest = next(d for d in last_action_defenders if d.x == deepest_x)
            values["line_break_proxy_last_action"] = (
                x_lo <= deepest.x <= x_hi
                and y_lo - LINE_BREAK_BAND_NATIVE <= deepest.y <= y_hi + LINE_BREAK_BAND_NATIVE
            )

        last_n_actions = prior[-LAST_N_ACTIONS_FOR_BYPASS:]
        last_n_defenders = None
        for candidate in reversed(last_n_actions):
            last_n_defenders = _defenders_for(candidate)
            if last_n_defenders:
                break
        last_n_valid = [e for e in last_n_actions if _valid_location(e)]
        if last_n_defenders and last_n_valid:
            xs = [e.location_x for e in last_n_valid] + [shot.location_x]
            ys = [e.location_y for e in last_n_valid] + [shot.location_y]
            values["defensive_layer_bypass_proxy_last_3_actions"] = _band_count(
                last_n_defenders, min(xs), max(xs), min(ys), max(ys)
            )

        chain = [e for e in [*prior, shot] if _valid_location(e)]
        count = 0
        eligible_transitions = 0
        for a, b in zip(chain, chain[1:]):
            a_defenders = _defenders_for(a)
            if not a_defenders:
                continue
            eligible_transitions += 1
            deepest_x = min(d.x for d in a_defenders)
            deepest = next(d for d in a_defenders if d.x == deepest_x)
            x_lo, x_hi = sorted((a.location_x, b.location_x))
            y_lo, y_hi = sorted((a.location_y, b.location_y))
            if (
                x_lo <= deepest.x <= x_hi
                and y_lo - LINE_BREAK_BAND_NATIVE <= deepest.y <= y_hi + LINE_BREAK_BAND_NATIVE
            ):
                count += 1
        values["line_break_proxy_possession_count"] = count if eligible_transitions > 0 else None

    # F12: rest-defence / transition structure ------------------------------------------------
    values.update(_rest_defence(shot, prior, e1e6_restart_kind, frames))

    # F14: sequence-level spatial dynamics -----------------------------------------------------
    values.update(_sequence_features(states, shot_static, shot_clock))

    return values


def _rest_defence(
    shot: EventRecord,
    prior: list[EventRecord],
    restart_kind: str | None,
    frames: dict[str, Frame],
) -> dict[str, object | None]:
    out: dict[str, object | None] = {
        "defenders_behind_ball_at_regain": None,
        "rest_defence_count_at_regain": None,
        "rest_defence_count_at_shot": None,
        "rest_defence_recovery_delta": None,
        "rest_defence_reset_fraction": None,
    }
    shot_frame = frames.get(shot.event_id)
    if shot_frame is not None and shot.team_id is not None and _valid_location(shot):
        oriented = orient_players(shot_frame.players, shot.team_id, shot.team_id)
        if oriented is not None:
            attackers = geo.outfield_attackers(oriented)
            out["rest_defence_count_at_shot"] = sum(1 for a in attackers if a.x < shot.location_x)
            if attackers:
                out["rest_defence_reset_fraction"] = out["rest_defence_count_at_shot"] / len(
                    attackers
                )

    if restart_kind == "live_regain":
        regain_event = prior[0] if prior else shot
        regain_frame = frames.get(regain_event.event_id)
        if (
            regain_frame is not None
            and regain_event.team_id is not None
            and _valid_location(regain_event)
        ):
            oriented = orient_players(regain_frame.players, regain_event.team_id, shot.team_id)
            if oriented is not None:
                attackers = geo.outfield_attackers(oriented)
                opposition = geo.all_opposition(oriented)
                out["defenders_behind_ball_at_regain"] = sum(
                    1 for d in opposition if d.x < regain_event.location_x
                )
                out["rest_defence_count_at_regain"] = sum(
                    1 for a in attackers if a.x < regain_event.location_x
                )
                if (
                    out["rest_defence_count_at_shot"] is not None
                    and out["rest_defence_count_at_regain"] is not None
                ):
                    out["rest_defence_recovery_delta"] = (
                        out["rest_defence_count_at_shot"] - out["rest_defence_count_at_regain"]
                    )
    return out


def _sequence_features(
    states: list[_State], shot_static: dict[str, object | None], shot_clock: float | None
) -> dict[str, object | None]:
    out: dict[str, object | None] = {
        "mean_defensive_compactness_last_3": None,
        "min_defensive_compactness_sequence": None,
        "max_box_numerical_advantage": None,
        "time_since_max_box_advantage": None,
        "max_goal_exposure": None,
        "goal_exposure_decay": None,
        "defensive_recovery_slope": None,
    }
    prior_compactness = [
        s.static.get("defensive_compactness")
        for s in states[-LAST_N_FRAMES_FOR_COMPACTNESS:]
        if s.static.get("defensive_compactness") is not None
    ]
    if prior_compactness:
        out["mean_defensive_compactness_last_3"] = sum(prior_compactness) / len(prior_compactness)

    full_compactness = [s.static.get("defensive_compactness") for s in states] + [
        shot_static.get("defensive_compactness")
    ]
    valid_compactness = [c for c in full_compactness if c is not None]
    if valid_compactness:
        out["min_defensive_compactness_sequence"] = min(valid_compactness)

    balances = [(s.static.get("box_numerical_balance"), s.elapsed_s) for s in states]
    balances.append((shot_static.get("box_numerical_balance"), 0.0))
    valid_balances = [(b, e) for b, e in balances if b is not None]
    if valid_balances:
        max_balance = max(b for b, _ in valid_balances)
        out["max_box_numerical_advantage"] = max_balance
        earliest_at_max = max(e for b, e in valid_balances if b == max_balance)
        out["time_since_max_box_advantage"] = earliest_at_max

    exposures = [(_goal_exposure(s.static), s.elapsed_s) for s in states]
    exposures.append((_goal_exposure(shot_static), 0.0))
    valid_exposures = [(x, e) for x, e in exposures if x is not None]
    if valid_exposures:
        max_exposure = max(x for x, _ in valid_exposures)
        out["max_goal_exposure"] = max_exposure
        shot_exposure = _goal_exposure(shot_static)
        if shot_exposure is not None:
            out["goal_exposure_decay"] = max_exposure - shot_exposure

    points = [(event_clock_s(s.event), s.static.get("defensive_line_depth")) for s in states]
    points = [(t, v) for t, v in points if t is not None and v is not None]
    if shot_static.get("defensive_line_depth") is not None and shot_clock is not None:
        points.append((shot_clock, shot_static.get("defensive_line_depth")))
    if len(points) >= MIN_SEQUENCE_SLOPE_POINTS:
        xs = [t for t, _ in points]
        ys = [v for _, v in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 0:
            cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            out["defensive_recovery_slope"] = cov_xy / var_x

    return out


def _derive_match(
    events: list[EventRecord], frames: dict[str, Frame]
) -> dict[str, dict[str, object | None]]:
    base = derive_event_contexts(events)
    ordered = _ordered(events)
    contexts: dict[str, dict[str, object | None]] = {}
    for index, shot in enumerate(ordered):
        if shot.event_type_name != "Shot":
            continue
        e1e6 = base[shot.event_id]
        if not e1e6.possession_context_valid:
            contexts[shot.event_id] = {name: None for name in F6_F14_FEATURES}
            continue
        shot_frame = frames.get(shot.event_id)
        shot_static: dict[str, object | None] = {}
        if shot_frame is not None and shot.team_id is not None and _valid_location(shot):
            oriented = orient_players(shot_frame.players, shot.team_id, shot.team_id)
            if oriented is not None:
                shot_static = derive_static_360_context(
                    shot_frame, oriented, shot.location_x, shot.location_y
                )
        contexts[shot.event_id] = _derive_shot(
            shot,
            e1e6.value("restart_vs_live_regain"),
            ordered,
            index,
            frames,
            shot_static,
        )
    return contexts


def derive_dynamic_360_context(
    events: Iterable[EventRecord], frames: dict[str, Frame]
) -> dict[str, dict[str, object | None]]:
    """Derive F6-F14 contexts independently for every governed match."""
    matches: dict[int, list[EventRecord]] = {}
    for event in events:
        matches.setdefault(event.match_id, []).append(event)
    contexts: dict[str, dict[str, object | None]] = {}
    for match_events in matches.values():
        contexts.update(_derive_match(match_events, frames))
    return contexts
