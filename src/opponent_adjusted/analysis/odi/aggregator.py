"""Trailing 15-minute On-pitch Defensive Index (ODI) rolling aggregator.

ODI(player, t) = sum(statsbomb_xg) - goals_conceded, over that player's
`cxg_defensive_involvement_v1` rows (as nearest defender) strictly inside
(t - WINDOW_S, t), same period as the shot being scored.

Convention matches `_momentum()` in `event_context_extended.py`: the window
is period-scoped (never bridges a period boundary) and window membership
uses `0 <= shot_clock - event_clock_s(event) <= WINDOW_S`. This module uses
the strict-exclusion variant `0 < diff <= WINDOW_S` so the shot currently
being scored (diff == 0) can never contribute to its own ODI (no
self-leakage) -- an involvement row from the *same* shot_event_id is also
explicitly excluded by id as a second, redundant guard.
"""

from __future__ import annotations

from dataclasses import dataclass

WINDOW_S = 900.0  # 15 minutes


@dataclass(frozen=True)
class InvolvementRow:
    player_id: int
    match_id: int
    shot_event_id: str
    period: int
    clock_s: float
    statsbomb_xg: float | None
    is_goal: bool | None  # None (unknown outcome) is treated as not-a-goal for ODI math


@dataclass(frozen=True)
class OdiResult:
    odi: float | None
    eligible: bool
    involvement_count: int
    null_reason: str | None  # None when eligible


def compute_odi(
    player_id: int,
    match_id: int,
    period: int,
    clock_s: float,
    current_shot_event_id: str,
    involvement_rows: list[InvolvementRow],
    period_tenure_s: float | None,
) -> OdiResult:
    """Compute one player's ODI at (match_id, period, clock_s).

    `period_tenure_s` is the player's continuous on-pitch tenure in the
    current period (from `roster.MatchRoster.period_tenure_s`); cold-start
    (< WINDOW_S of tenure) yields a null ODI with an explicit reason rather
    than a zero or a partial-window value.
    """
    if period_tenure_s is None:
        return OdiResult(None, False, 0, "not_on_pitch")
    if period_tenure_s < WINDOW_S:
        return OdiResult(None, False, 0, "cold_start_lt_15min")

    window_rows = [
        row
        for row in involvement_rows
        if row.player_id == player_id
        and row.match_id == match_id
        and row.period == period
        and row.shot_event_id != current_shot_event_id
        and 0 < clock_s - row.clock_s <= WINDOW_S
    ]
    xg_faced = sum(row.statsbomb_xg for row in window_rows if row.statsbomb_xg is not None)
    goals_conceded = sum(1 for row in window_rows if row.is_goal is True)
    return OdiResult(xg_faced - goals_conceded, True, len(window_rows), None)
