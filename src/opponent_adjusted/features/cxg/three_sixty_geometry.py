"""Pure spatial geometry helpers shared by CxG+ static (F1-F5) and dynamic (F6-F14) code.

All functions operate on already-ORIENTED FramePlayer tuples (see
three_sixty_frame.orient_players): `teammate is True` means "same team as the
shot", `teammate is False` means "opposing the shot", both relative to the
CURRENT shot's attacking direction (goal at x=120, matching the frozen S/E1-E12
convention already used across this repository for shot geometry).

Every player-set argument below is expected to already be filtered to
`coordinate_valid` players by the caller; helpers here do not re-validate.

Distance units: only `nearest_defender_distance_m`/`count_within_radius_m` (and the
F5 proxies that call them) use the approximate-metres bridge documented in
`three_sixty_frame.DISTANCE_UNIT_CONTRACT`. `gk_distance_to_point`/
`gk_distance_to_goal_centre` are deliberately NATIVE-unit distances (plain
`math.hypot`, no metre conversion) and are not metre-labelled candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from opponent_adjusted.features.cxg.geometry import (
    BOX_X_MIN,
    BOX_Y_MAX,
    BOX_Y_MIN,
    GOAL_CENTRE_Y,
    GOAL_POST_HIGH_Y,
    GOAL_POST_LOW_Y,
    GOAL_X,
)
from opponent_adjusted.features.cxg.three_sixty_frame import FramePlayer, metre_distance

# --- Versioned fixed parameters (cxg_360_context_v1) --------------------------------------
MIN_DEFENDERS_FOR_BLOCK = 1
MIN_DEFENDERS_FOR_COMPACTNESS = 2
MIN_DEFENDERS_FOR_HULL = 3
LOCAL_DENSITY_RADIUS_M = 8.0
BLOCKER_BODY_RADIUS_M = 0.5
GOAL_MOUTH_LANE_HALF_WIDTH = (GOAL_POST_HIGH_Y - GOAL_POST_LOW_Y) / 2.0
N_GOAL_SAMPLE_POINTS = 11


def outfield_attackers(players: tuple[FramePlayer, ...]) -> tuple[FramePlayer, ...]:
    return tuple(p for p in players if p.teammate is True and p.coordinate_valid)


def outfield_defenders(players: tuple[FramePlayer, ...]) -> tuple[FramePlayer, ...]:
    return tuple(
        p for p in players if p.teammate is False and p.keeper is not True and p.coordinate_valid
    )


def all_opposition(players: tuple[FramePlayer, ...]) -> tuple[FramePlayer, ...]:
    return tuple(p for p in players if p.teammate is False and p.coordinate_valid)


@dataclass(frozen=True)
class Cone:
    """Linear projection of the goal mouth from a ball location ("shot cone")."""

    ball_x: float
    ball_y: float

    def lane_bounds(self, x: float) -> tuple[float, float] | None:
        if x <= self.ball_x or x > GOAL_X:
            return None
        fraction = (x - self.ball_x) / (GOAL_X - self.ball_x) if GOAL_X != self.ball_x else 1.0
        low = self.ball_y + (GOAL_POST_LOW_Y - self.ball_y) * fraction
        high = self.ball_y + (GOAL_POST_HIGH_Y - self.ball_y) * fraction
        return low, high

    def contains(self, x: float, y: float) -> bool:
        bounds = self.lane_bounds(x)
        return bounds is not None and bounds[0] <= y <= bounds[1]

    def half_width_at(self, x: float) -> float | None:
        bounds = self.lane_bounds(x)
        return None if bounds is None else (bounds[1] - bounds[0]) / 2.0


def nearest_defender_distance_m(
    ref_x: float, ref_y: float, defenders: tuple[FramePlayer, ...]
) -> float | None:
    if not defenders:
        return None
    return min(metre_distance(ref_x, ref_y, d.x, d.y) for d in defenders)


def count_within_radius_m(
    ref_x: float, ref_y: float, defenders: tuple[FramePlayer, ...], radius_m: float
) -> int:
    return sum(1 for d in defenders if metre_distance(ref_x, ref_y, d.x, d.y) <= radius_m)


def defenders_in_cone(defenders: tuple[FramePlayer, ...], cone: Cone) -> tuple[FramePlayer, ...]:
    return tuple(d for d in defenders if cone.contains(d.x, d.y))


def central_lane_bounds(x_min: float, x_max: float) -> tuple[float, float, float, float]:
    """Fixed-width central lane between two x-anchors (used by F3/F5, distinct from the Cone)."""
    return (
        x_min,
        x_max,
        GOAL_CENTRE_Y - GOAL_MOUTH_LANE_HALF_WIDTH,
        GOAL_CENTRE_Y + GOAL_MOUTH_LANE_HALF_WIDTH,
    )


def count_in_central_lane(defenders: tuple[FramePlayer, ...], x_min: float, x_max: float) -> int:
    lo_x, hi_x, lo_y, hi_y = central_lane_bounds(x_min, x_max)
    return sum(1 for d in defenders if lo_x <= d.x <= hi_x and lo_y <= d.y <= hi_y)


# --- F2 defensive block geometry ------------------------------------------------------------


def defensive_line_depth(defenders: tuple[FramePlayer, ...]) -> float | None:
    return min(d.x for d in defenders) if len(defenders) >= MIN_DEFENDERS_FOR_BLOCK else None


def defensive_centroid(defenders: tuple[FramePlayer, ...]) -> tuple[float, float] | None:
    if len(defenders) < MIN_DEFENDERS_FOR_BLOCK:
        return None
    return (
        sum(d.x for d in defenders) / len(defenders),
        sum(d.y for d in defenders) / len(defenders),
    )


def defensive_width(defenders: tuple[FramePlayer, ...]) -> float | None:
    if len(defenders) < MIN_DEFENDERS_FOR_BLOCK:
        return None
    ys = [d.y for d in defenders]
    return max(ys) - min(ys)


def defensive_length(defenders: tuple[FramePlayer, ...]) -> float | None:
    if len(defenders) < MIN_DEFENDERS_FOR_BLOCK:
        return None
    xs = [d.x for d in defenders]
    return max(xs) - min(xs)


def defensive_compactness(defenders: tuple[FramePlayer, ...]) -> float | None:
    """Bounding-box area proxy (width * length); smaller means more compact."""
    if len(defenders) < MIN_DEFENDERS_FOR_COMPACTNESS:
        return None
    width = defensive_width(defenders)
    length = defensive_length(defenders)
    return None if width is None or length is None else width * length


def _hull_points(defenders: tuple[FramePlayer, ...]) -> list[tuple[float, float]]:
    points = sorted({(d.x, d.y) for d in defenders})

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def defensive_hull_area(defenders: tuple[FramePlayer, ...]) -> float | None:
    if len(defenders) < MIN_DEFENDERS_FOR_HULL:
        return None
    hull = _hull_points(defenders)
    if len(hull) < 3:
        return None
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


# --- F3 box occupation -----------------------------------------------------------------------


def count_in_box(players: tuple[FramePlayer, ...]) -> int:
    return sum(1 for p in players if p.x >= BOX_X_MIN and BOX_Y_MIN <= p.y <= BOX_Y_MAX)


# --- F4 goalkeeper spatial state --------------------------------------------------------------


def gk_depth(gk: FramePlayer) -> float:
    return GOAL_X - gk.x


def gk_lateral_offset(gk: FramePlayer) -> float:
    return gk.y - GOAL_CENTRE_Y


def gk_distance_to_point(gk: FramePlayer, x: float, y: float) -> float:
    return math.hypot(gk.x - x, gk.y - y)


def gk_distance_to_goal_centre(gk: FramePlayer) -> float:
    return math.hypot(gk.x - GOAL_X, gk.y - GOAL_CENTRE_Y)


# --- F5 goal exposure proxies ------------------------------------------------------------------


def _goal_sample_points(n: int = N_GOAL_SAMPLE_POINTS) -> tuple[float, ...]:
    step = (GOAL_POST_HIGH_Y - GOAL_POST_LOW_Y) / (n - 1)
    return tuple(GOAL_POST_LOW_Y + step * i for i in range(n))


def estimated_goalface_occlusion(
    ball_x: float, ball_y: float, blockers: tuple[FramePlayer, ...]
) -> float:
    """Fraction of fixed goal-mouth sample points shadowed by a blocker within a fixed body radius."""
    sample_ys = _goal_sample_points()
    if not blockers:
        return 0.0
    blocked = 0
    for sample_y in sample_ys:
        for b in blockers:
            if not (ball_x < b.x <= GOAL_X):
                continue
            fraction = (b.x - ball_x) / (GOAL_X - ball_x) if GOAL_X != ball_x else 1.0
            lane_y = ball_y + (sample_y - ball_y) * fraction
            if metre_distance(b.x, b.y, b.x, lane_y) <= BLOCKER_BODY_RADIUS_M:
                blocked += 1
                break
    return blocked / len(sample_ys)


def shot_corridor_occlusion(
    ball_x: float, ball_y: float, blockers: tuple[FramePlayer, ...], cone: Cone
) -> float:
    """Sum of each in-cone blocker's body-radius shadow fraction of the cone's local half-width, capped at 1.0."""
    total = 0.0
    for b in blockers:
        if not cone.contains(b.x, b.y):
            continue
        half_width = cone.half_width_at(b.x)
        if not half_width:
            continue
        total += min(1.0, (2.0 * BLOCKER_BODY_RADIUS_M) / (2.0 * half_width))
    return min(1.0, total)


def visible_goal_angle_proxy(
    ball_x: float,
    ball_y: float,
    raw_angle_rad: float,
    blockers: tuple[FramePlayer, ...],
    cone: Cone,
) -> float:
    """Raw unobstructed shot angle minus the angular shadow of the nearest in-cone blocker."""
    in_cone = defenders_in_cone(blockers, cone)
    if not in_cone:
        return raw_angle_rad
    nearest = min(in_cone, key=lambda b: metre_distance(ball_x, ball_y, b.x, b.y))
    distance = metre_distance(ball_x, ball_y, nearest.x, nearest.y)
    if distance <= 0:
        return 0.0
    occluded = 2.0 * math.atan(BLOCKER_BODY_RADIUS_M / distance)
    return max(0.0, raw_angle_rad - occluded)
