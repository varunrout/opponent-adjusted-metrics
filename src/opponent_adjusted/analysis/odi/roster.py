"""On-pitch roster resolution at an arbitrary match timestamp.

Given a team's `starting_xi_players` lineup and `substitutions` for a match,
resolves who is on the pitch (and their nominal position) as of any
(period, clock_s) point. Used by the ODI aggregator for two purposes:

1. Eligibility / cold-start: how long has a defender been continuously on
   the pitch *in the current period* as of a shot, to gate the trailing
   15-minute involvement window.
2. Position-slot aggregation (e.g. mean back-line ODI, GK ODI).

Clock convention: `clock_s` values are the same match-cumulative
`event_clock_s` used by `event_context.py` / `event_context_extended.py`
(`minute * 60 + second [+ fractional seconds]`; StatsBomb's `minute` does not
reset at half-time). Ordering across periods is therefore `(period, clock_s)`
lexicographic, matching the convention already used by `_momentum()` in
`event_context_extended.py`.

Period-tenure convention (deliberate, matches the repository's existing
non-bridging rule for rolling windows -- see `_momentum()`): a player's
*continuous on-pitch tenure*, for cold-start purposes, is measured from the
later of (a) the start of the *current* period, or (b) their substitution-on
timestamp if that happened during the current period. A player who has
played the entire first half is therefore treated as having tenure 0 at the
instant the second period starts -- this is intentional: the ODI rolling
window itself never looks back across a period boundary (see
`odi/aggregator.py`), so a player's *usable* trailing-window history really
does restart at each period boundary regardless of on-pitch continuity. This
is documented here so it is not mistaken for a bug.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StarterRecord:
    """One `starting_xi_players` row, position already resolved from the lineup."""

    team_id: int
    player_id: int
    position_name: str | None


@dataclass(frozen=True)
class SubstitutionRecord:
    """One `substitutions` row with its match clock already resolved to `clock_s`.

    `clock_s` must be computed the same way as `event_context.event_clock_s`
    (this module does not recompute it, to keep the roster resolver decoupled
    from the frozen E-family timestamp-parsing code).
    """

    team_id: int
    period: int
    clock_s: float
    player_off_id: int | None
    player_on_id: int | None


@dataclass(frozen=True)
class OnPitchPlayer:
    player_id: int
    team_id: int
    position_name: str | None
    period_tenure_start_s: float | None
    source: str  # "starting_xi" | "substitute"


class MatchRoster:
    """Resolves on-pitch personnel for one match, team-scoped, at any timestamp."""

    def __init__(
        self,
        starters: list[StarterRecord],
        subs: list[SubstitutionRecord],
        period_start_clock_s: dict[int, float],
        substitute_position_lookup: dict[int, str | None] | None = None,
    ) -> None:
        self._starters = starters
        self._subs_sorted = sorted(subs, key=lambda s: (s.period, s.clock_s))
        self._period_start_clock_s = period_start_clock_s
        self._substitute_position_lookup = substitute_position_lookup or {}

    def on_pitch(self, team_id: int, period: int, clock_s: float) -> list[OnPitchPlayer]:
        """Return the on-pitch roster for `team_id` as of (period, clock_s), inclusive."""
        subbed_off: set[int] = set()
        subbed_on: dict[int, tuple[int, float]] = {}
        for sub in self._subs_sorted:
            if (sub.period, sub.clock_s) > (period, clock_s):
                break
            if sub.team_id != team_id:
                continue
            if sub.player_off_id is not None:
                subbed_off.add(sub.player_off_id)
            if sub.player_on_id is not None:
                subbed_on[sub.player_on_id] = (sub.period, sub.clock_s)

        period_start = self._period_start_clock_s.get(period)
        result: list[OnPitchPlayer] = []
        for starter in self._starters:
            if starter.team_id != team_id or starter.player_id in subbed_off:
                continue
            if starter.player_id in subbed_on:
                continue
            result.append(
                OnPitchPlayer(starter.player_id, team_id, starter.position_name, period_start, "starting_xi")
            )
        for player_id, (sub_period, sub_clock) in subbed_on.items():
            if player_id in subbed_off:
                continue
            tenure_start = sub_clock if sub_period == period else period_start
            result.append(
                OnPitchPlayer(
                    player_id,
                    team_id,
                    self._substitute_position_lookup.get(player_id),
                    tenure_start,
                    "substitute",
                )
            )
        return result

    def period_tenure_s(self, player_id: int, team_id: int, period: int, clock_s: float) -> float | None:
        """Seconds of continuous on-pitch tenure in the current period, or None if not on pitch."""
        for p in self.on_pitch(team_id, period, clock_s):
            if p.player_id == player_id:
                if p.period_tenure_start_s is None:
                    return None
                return clock_s - p.period_tenure_start_s
        return None
