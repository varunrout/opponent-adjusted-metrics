"""Shared StatsBomb 360 freeze-frame primitives for CxG+ (F1-F15).

StatsBomb 360 is EVENT-LINKED FREEZE-FRAME data, not continuous tracking. A
`Frame` below is one governed snapshot attached to exactly one source event
(`event_uuid`). Nothing in this module infers positions between frames or
claims velocity/acceleration/trajectory; see three_sixty_sequence.py for the
explicit discrete state-delta semantics used by F6-F14.

Player orientation: the raw `teammate` flag on a 360 frame player is relative
to the event actor/team that frame is attached to (StatsBomb 360 spec), NOT
necessarily the shot-taking team. `orient_players` below re-expresses
`teammate` from the shot team's perspective so every downstream F-family
helper can assume `teammate is True` means "same team as the shot".

Distance-unit contract (OPTION A, locked 2026-08-20 scientific closure
correction): the governed taxonomy names `defenders_within_3m` / `_5m` / `_8m`
require a physical-distance interpretation, but StatsBomb's native 120x80
coordinate frame has no first-party metre definition and is not measured per
stadium. These candidates are therefore explicitly defined as APPROXIMATE
METRES under an ASSUMED STANDARD PITCH of 105m x 68m (`NATIVE_TO_METRE_X/Y`
below), matching common open StatsBomb tooling conventions (e.g.
mplsoccer/kloppy). This is NOT a measured true distance for the specific
stadium/pitch of any given match; it is a fixed, versioned, documented
approximation applied ONLY to the *_within_Nm radius-band and other
metre-labelled candidates (`nearest_defender_distance`, `actor_space`,
`gk_distance_to_shooter`/`gk_distance_to_goal_centre` stay in NATIVE units,
undocumented as metres -- see three_sixty_geometry.py). No other CxG/CxG+
geometry is converted; box boundaries, goal geometry, and all "_sb"-style
native distances stay in native StatsBomb units. No universal 105x68
reprojection of the pitch itself is performed anywhere in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from opponent_adjusted.features.cxg.geometry import (
    PITCH_X_MAX,
    PITCH_X_MIN,
    PITCH_Y_MAX,
    PITCH_Y_MIN,
)

CXG_360_CONTEXT_CONTRACT_ID = "cxg_360_context_v1"

# Distance-unit contract metadata (machine-readable; see module docstring above).
DISTANCE_UNIT_CONTRACT = (
    "approximate_metres_assumed_105x68_standard_pitch_not_measured_stadium_distance"
)
NATIVE_TO_METRE_X = 105.0 / 120.0
NATIVE_TO_METRE_Y = 68.0 / 80.0

# F1: conservative visibility gate. A radius-band/local-pressure feature requires the
# ball/actor probe point itself to fall inside the frame's visible_area polygon; this
# does not prove the *entire* neighbourhood was captured, but avoids converting "not
# visible" into a fabricated zero for the common case of a wholly uncaptured frame.
MIN_VISIBLE_AREA_POINTS = 6  # >= 3 (x, y) pairs


@dataclass(frozen=True)
class FramePlayer:
    """One freeze-frame player, oriented or raw depending on caller."""

    ordinal: int
    teammate: bool | None
    actor: bool | None
    keeper: bool | None
    x: float | None
    y: float | None

    @property
    def coordinate_valid(self) -> bool:
        return (
            self.x is not None
            and self.y is not None
            and math.isfinite(self.x)
            and math.isfinite(self.y)
            and PITCH_X_MIN <= self.x <= PITCH_X_MAX
            and PITCH_Y_MIN <= self.y <= PITCH_Y_MAX
        )


@dataclass(frozen=True)
class Frame:
    """One StatsBomb 360 freeze-frame, keyed by its linked source event_uuid."""

    event_uuid: str
    match_id: int
    visible_area: tuple[float, ...]
    players: tuple[FramePlayer, ...]

    @property
    def has_visible_area(self) -> bool:
        return len(self.visible_area) >= MIN_VISIBLE_AREA_POINTS and len(self.visible_area) % 2 == 0


def metre_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Approximate metres under an assumed 105x68 standard pitch (DISTANCE_UNIT_CONTRACT);
    NOT a measured true distance for the specific match's actual stadium/pitch dimensions."""
    return math.hypot((x2 - x1) * NATIVE_TO_METRE_X, (y2 - y1) * NATIVE_TO_METRE_Y)


def orient_players(
    players: tuple[FramePlayer, ...], frame_event_team_id: int | None, shot_team_id: int | None
) -> tuple[FramePlayer, ...] | None:
    """Re-express `teammate` relative to the shot-taking team.

    Returns None when either team identity is unknown: orientation must never be guessed.
    """
    if frame_event_team_id is None or shot_team_id is None:
        return None
    if frame_event_team_id == shot_team_id:
        return players
    return tuple(
        FramePlayer(
            p.ordinal,
            None if p.teammate is None else not p.teammate,
            p.actor,
            p.keeper,
            p.x,
            p.y,
        )
        for p in players
    )


def defending_keeper(players: tuple[FramePlayer, ...]) -> tuple[FramePlayer | None, bool]:
    """Return the unambiguous defending keeper (oriented teammate=False, keeper=True).

    Second element is True when multiple candidates make the keeper ambiguous (audit flag).
    """
    candidates = [
        p for p in players if p.keeper is True and p.teammate is False and p.coordinate_valid
    ]
    if len(candidates) == 1:
        return candidates[0], False
    return None, len(candidates) > 1


def find_actor(players: tuple[FramePlayer, ...]) -> FramePlayer | None:
    candidates = [p for p in players if p.actor is True and p.coordinate_valid]
    return candidates[0] if len(candidates) == 1 else None


def _point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    """Ray-casting point-in-polygon test used for visible_area eligibility."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            if x < x_intersect:
                inside = not inside
    return inside


def visible_area_polygon(frame: Frame) -> tuple[tuple[float, float], ...] | None:
    if not frame.has_visible_area:
        return None
    coords = frame.visible_area
    return tuple((coords[i], coords[i + 1]) for i in range(0, len(coords), 2))


def point_observable(frame: Frame | None, x: float | None, y: float | None) -> bool:
    """A single probe point is observable only if it demonstrably falls in the visible area."""
    if frame is None or x is None or y is None:
        return False
    polygon = visible_area_polygon(frame)
    if polygon is None:
        return False
    return _point_in_polygon(x, y, polygon)


def region_observable(frame: Frame | None, corners: tuple[tuple[float, float], ...]) -> bool:
    """A region is observable only if ALL probe corners fall inside the visible-area polygon."""
    if frame is None:
        return False
    polygon = visible_area_polygon(frame)
    if polygon is None:
        return False
    return all(_point_in_polygon(x, y, polygon) for x, y in corners)
