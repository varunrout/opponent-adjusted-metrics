"""Phase C rolling-window defensive features (v3, BOTH tracks -- CxG event-wide AND CxG+).

Unlike Phase A/B (CxG+-only, 360-freeze-frame-dependent), these three candidate features
are event-log-based and apply to every shot in the governed 610-match universe, not just
the 360-eligible subset. They therefore land on `cxg_event_context_features` (the
event-wide Gold family table), not a `*_360_features` table.

  - defensive_action_rate_{15,30,45,60}m  within-match rolling rate (per minute) of the
                                           DEFENDING team's defensive actions in a trailing
                                           window before the shot.
  - territorial_dominance_last_15m        `_momentum()`'s exact field-tilt math
                                           (`event_context_extended.py`), extended from its
                                           frozen 5-minute window to 15 minutes. That module
                                           is frozen (do not modify), so the mass/ratio
                                           computation is duplicated here parameterized by
                                           window size rather than imported.
  - cross_match_defensive_rate            recency-weighted (exponential decay) average of
                                           the defending team's per-minute defensive-action
                                           rate across their PRIOR matches, pooled globally
                                           by team_id + chronological match date (not scoped
                                           to competition/season -- competitions 43/55 share
                                           real team_ids, so scoping would double-count/
                                           fragment history for those teams; isolation for
                                           genuinely single-competition teams like
                                           competition 2/season 27 emerges naturally from
                                           the data rather than being policy-enforced).

DEFENSIVE ACTION TYPE SET
--------------------------
Reused unchanged from Phase B (`analysis/defstyle/features.py:ACTION_TYPES`): Pressure,
Duel, Interception, Clearance, Block, Foul Committed, 50/50. Not re-derived here.

HALF-LIFE PARAMETER (cross-match decay)
----------------------------------------
`CROSS_MATCH_HALF_LIFE_MATCHES = 3.0`, justified against the live matches-per-team
distribution (74 teams, 610 matches; min=3, median=8, max=38; bimodal -- 13 teams sit at
the floor of exactly 3 matches, 20 teams sit at the ceiling of 38):

  - 3 matches matches the common football-analytics "recent form" window (3-5 games).
  - For the 13 floor teams (only 3 total matches each), a half-life of 3 keeps all of a
    team's history contributing non-trivially to any later match (gap-1/2/3 weights of
    ~0.79/0.63/0.50) rather than collapsing to "essentially just the last match" the way a
    much shorter half-life (e.g. 1) would.
  - For the 20 ceiling teams (38 matches), matches more than ~15 games back decay below
    ~3% weight (0.5^(15/3) = 0.031), appropriately favouring recent defensive form over
    ancient history for teams with a long observed run.

WINDOW-DENOMINATOR / COLD-START CONVENTION (2a)
-------------------------------------------------
The rate denominator is the ACTUAL elapsed time since the period started (capped at the
nominal window size), not the nominal window size itself -- `elapsed_minutes =
min(window_s, shot_clock - period_start_clock) / 60`. This is the mathematically unbiased
way to report a partial-window rate (dividing by the time actually observed, not a fixed
nominal size that would silently deflate the rate near kickoff) -- explicitly NOT the
"truncated rate silently presented as a full window" anti-pattern. The only genuine
null case is `elapsed_minutes <= 0` (a shot at the literal start of a period, before any
observation window exists) -- reason `"zero_elapsed_time_in_period"`.

`period_start_clock` is derived empirically per (match, period) as the minimum observed
`event_clock_s` across that period's events, rather than assuming a fixed StatsBomb
period-boundary convention, matching this project's "verify, don't assume" discipline.
"""

from __future__ import annotations

from opponent_adjusted.features.cxg.event_context import EventRecord, _valid_location, event_clock_s

CROSS_MATCH_HALF_LIFE_MATCHES = 3.0

WINDOW_MINUTES: tuple[int, ...] = (15, 30, 45, 60)
FIELD_TILT_WINDOW_S = 900.0  # 15 minutes, extending _momentum()'s frozen 300s (5m) window.

_ATTACKING_TYPES = {"Pass", "Carry", "Dribble", "Ball Receipt*", "Shot"}

NULL_REASON_ZERO_ELAPSED = "zero_elapsed_time_in_period"
NULL_REASON_NO_SHOT_CLOCK = "shot_clock_unavailable"
NULL_REASON_FIRST_MATCH = "team_first_match_in_dataset"
NULL_REASON_DEFENDING_TEAM_UNRESOLVED = "defending_team_unresolved"


def period_start_clock(events: list[EventRecord], period: int | None) -> float | None:
    """Empirically-derived start-of-period clock: the earliest observed `event_clock_s`
    among that period's events, not an assumed fixed boundary."""
    clocks = [
        event_clock_s(event)
        for event in events
        if event.period == period and event_clock_s(event) is not None
    ]
    return min(clocks) if clocks else None


def defensive_action_rates(
    events: list[EventRecord],
    defending_team_id: int | None,
    action_types: frozenset[str],
    shot_period: int | None,
    shot_clock: float | None,
) -> dict[str, object | None]:
    """Per-minute rate of the defending team's defensive actions, trailing windows of
    `WINDOW_MINUTES`. `events` should be every event strictly BEFORE the shot in the same
    match (backward-looking only, no future leakage)."""
    columns: dict[str, object | None] = {
        f"defensive_action_rate_{m}m": None for m in WINDOW_MINUTES
    }
    columns["defensive_action_rate_null_reason"] = None

    if shot_clock is None:
        columns["defensive_action_rate_null_reason"] = NULL_REASON_NO_SHOT_CLOCK
        return columns
    if defending_team_id is None:
        columns["defensive_action_rate_null_reason"] = NULL_REASON_DEFENDING_TEAM_UNRESOLVED
        return columns

    period_events = [event for event in events if event.period == shot_period]
    start_clock = period_start_clock(period_events, shot_period)
    elapsed_actual = None if start_clock is None else shot_clock - start_clock
    if elapsed_actual is None or elapsed_actual <= 0:
        columns["defensive_action_rate_null_reason"] = NULL_REASON_ZERO_ELAPSED
        return columns

    defending_events = [
        event
        for event in period_events
        if event.team_id == defending_team_id
        and event.event_type_name in action_types
        and event_clock_s(event) is not None
    ]

    for window_min in WINDOW_MINUTES:
        window_s = window_min * 60.0
        count = sum(
            0 <= shot_clock - event_clock_s(event) <= window_s for event in defending_events
        )
        elapsed_minutes = min(window_s, elapsed_actual) / 60.0
        columns[f"defensive_action_rate_{window_min}m"] = count / elapsed_minutes

    return columns


def territorial_dominance_extended(
    events: list[EventRecord],
    shot_team_id: int | None,
    shot_period: int | None,
    shot_clock: float | None,
    window_s: float = FIELD_TILT_WINDOW_S,
) -> float | None:
    """`_momentum()`'s exact `territorial_dominance_last_5m` math
    (`event_context_extended.py`), parameterized to `window_s` instead of the frozen 300s.
    `events` should be every event strictly before the shot in the same match."""
    if shot_clock is None:
        return None
    window = [
        event
        for event in events
        if event.period == shot_period
        and event_clock_s(event) is not None
        and 0 <= shot_clock - event_clock_s(event) <= window_s
    ]
    team = [event for event in window if event.team_id == shot_team_id]
    opposition = [
        event for event in window if event.team_id is not None and event.team_id != shot_team_id
    ]
    attacking_team = [event for event in team if event.event_type_name in _ATTACKING_TYPES]
    attacking_opp = [event for event in opposition if event.event_type_name in _ATTACKING_TYPES]

    def mass(items: list[EventRecord]) -> float:
        return sum(event.location_x / 120.0 for event in items if _valid_location(event))

    team_mass, opp_mass = mass(attacking_team), mass(attacking_opp)
    total_mass = team_mass + opp_mass
    return (team_mass - opp_mass) / total_mass if total_mass > 0 else None


def decay_weight(gap_matches: int, half_life: float = CROSS_MATCH_HALF_LIFE_MATCHES) -> float:
    """Exponential-decay weight for a prior match `gap_matches` matches before the current
    one (gap=1 is the immediately preceding match)."""
    if gap_matches < 1:
        raise ValueError(f"gap_matches must be >= 1, got {gap_matches}")
    return 0.5 ** (gap_matches / half_life)


def cross_match_rolling_rate(
    prior_match_rates: list[float],
    half_life: float = CROSS_MATCH_HALF_LIFE_MATCHES,
) -> tuple[float | None, str | None]:
    """Recency-weighted average of a team's per-minute defensive-action rate across their
    prior matches, ordered chronologically NEAREST-FIRST (`prior_match_rates[0]` = the
    immediately preceding match, gap=1). Returns `(value, null_reason)`; a team's literal
    first match in the dataset (empty list) is an explicit null, never a silent zero."""
    if not prior_match_rates:
        return None, NULL_REASON_FIRST_MATCH
    weights = [decay_weight(gap, half_life) for gap in range(1, len(prior_match_rates) + 1)]
    weighted_sum = sum(w * r for w, r in zip(weights, prior_match_rates))
    return weighted_sum / sum(weights), None
