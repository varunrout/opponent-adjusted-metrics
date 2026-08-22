"""Phase A geometric/categorical defender features (CxG+ only, new candidate features --
not yet qualified by the governed univariate/correlation/bivariate chain):

  - nearest_defender_role                  categorical (GK/CB/Fullback_WingBack/Midfield/Attack)
  - nearest_defender_zone_displacement     continuous, metres, nearest defender's actual
                                            location vs. their role's dataset-wide centroid
  - second_nearest_defender_role           same taxonomy, applied to the 2nd-nearest defender
  - nearest_defender_gap                   continuous, metres, distance between 1st and 2nd
                                            nearest defenders

Reuses `metre_distance` (three_sixty_frame.py, the DISTANCE_UNIT_CONTRACT-governed metres
bridge already used by `nearest_defender_distance`) for the distance metric, and
`bucket_defender_role` (analysis/odi/contracts.py) for role bucketing -- no new distance
formula or position-name taxonomy invented here. Ranking defenders by distance (1st/2nd
nearest) has no existing precedent in this repo (confirmed before writing this module) --
that part is genuinely new logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from opponent_adjusted.analysis.odi.contracts import bucket_defender_role
from opponent_adjusted.features.cxg.geometry import GOAL_CENTRE_Y, GOAL_X
from opponent_adjusted.features.cxg.three_sixty_frame import metre_distance

NULL_REASON_FEWER_THAN_2_DEFENDERS = "fewer_than_2_defenders_identified"


def _flip_to_own_attacking_frame(x: float, y: float) -> tuple[float, float]:
    """`shot_freeze_frame_players.x/y` is expressed in the SHOOTING team's attacking
    direction (goal at x=120, the same convention `three_sixty_frame.py` documents for
    every F-family feature). `oam_core.events.location_x/y` -- and therefore the role
    centroids computed from it -- is expressed in each event's OWN acting team's attacking
    direction (StatsBomb's standard per-team normalization, confirmed in Step 1). For a
    defender facing a shot, those are opposite-facing frames: their own goal sits at x=120
    in the shot's frame, but at x=0 in their own team's frame. A rigid 180-degree
    reflection about the pitch centre reconciles them -- verified against `gk_depth`
    (`GOAL_X - gk.x`, shot-frame): GK's own-events centroid here is x=9.55, and
    120 - 9.55 = 110.45, matching a plausible average shot-frame GK depth almost exactly.
    Without this flip, `nearest_defender_zone_displacement` came out with a ~44m mean
    (nearly half the pitch) -- the exact "systematically huge" symptom the task's Step 4
    sanity check calls out, caught before this table was finalized."""
    return GOAL_X - x, 2 * GOAL_CENTRE_Y - y


@dataclass(frozen=True)
class DefenderCandidate:
    position_name: str | None
    x: float
    y: float


def role_centroids(position_stats: dict[str, tuple[int, float, float]]) -> dict[str, tuple[float, float]]:
    """`position_stats`: {position_name: (n, sum_location_x, sum_location_y)}, aggregated
    dataset-wide from `oam_core.events` (every event with a non-null position_name and
    coordinates, not just defenders-in-freeze-frames). Buckets each `position_name` via
    `bucket_defender_role`, then computes a true weighted mean per bucket -- NOT a naive
    mean-of-position-name-means, which would overweight rare position names relative to
    common ones within the same bucket."""
    totals: dict[str, list[float]] = {}
    for position_name, (n, sum_x, sum_y) in position_stats.items():
        bucket = bucket_defender_role(position_name)
        if bucket is None or n <= 0:
            continue
        acc = totals.setdefault(bucket, [0.0, 0.0, 0.0])
        acc[0] += n
        acc[1] += sum_x
        acc[2] += sum_y
    return {bucket: (sx / n, sy / n) for bucket, (n, sx, sy) in totals.items() if n > 0}


def nearest_defenders(
    shot_x: float, shot_y: float, candidates: tuple[DefenderCandidate, ...]
) -> tuple[DefenderCandidate | None, DefenderCandidate | None]:
    """Candidates ranked ascending by `metre_distance` to the shot location. Returns
    (1st, 2nd), either or both `None` if fewer candidates are available -- never fabricated."""
    ranked = sorted(candidates, key=lambda d: metre_distance(shot_x, shot_y, d.x, d.y))
    first = ranked[0] if len(ranked) >= 1 else None
    second = ranked[1] if len(ranked) >= 2 else None
    return first, second


def compute_shot_features(
    shot_x: float,
    shot_y: float,
    candidates: tuple[DefenderCandidate, ...],
    centroids: dict[str, tuple[float, float]],
) -> dict[str, object]:
    first, second = nearest_defenders(shot_x, shot_y, candidates)

    nearest_role = bucket_defender_role(first.position_name) if first is not None else None
    zone_displacement = None
    if first is not None and nearest_role is not None and nearest_role in centroids:
        centroid_x, centroid_y = centroids[nearest_role]
        own_frame_x, own_frame_y = _flip_to_own_attacking_frame(first.x, first.y)
        zone_displacement = metre_distance(own_frame_x, own_frame_y, centroid_x, centroid_y)

    second_role = bucket_defender_role(second.position_name) if second is not None else None
    # No flip needed here: both defenders are already in the SAME (shot) frame, and a rigid
    # reflection is a distance-preserving isometry -- flipping both would leave their
    # pairwise distance unchanged, so it's simply skipped.
    gap = metre_distance(first.x, first.y, second.x, second.y) if (first is not None and second is not None) else None

    return {
        "nearest_defender_role": nearest_role,
        "nearest_defender_zone_displacement": zone_displacement,
        "second_nearest_defender_role": second_role,
        "nearest_defender_gap": gap,
        "nearest_defender_rank_null_reason": None if second is not None else NULL_REASON_FEWER_THAN_2_DEFENDERS,
    }
