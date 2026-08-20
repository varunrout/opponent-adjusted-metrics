"""CxG+ F1-F5: current-shot 360 snapshot families (cxg_360_context_v1).

All features below are computed from the SHOT's OWN linked 360 frame only
(current-shot snapshot; no sequence/history). Eligibility gates are documented
per family; a null here always means "not observable/not eligible", never a
fabricated zero. See three_sixty_frame.py for the visibility/orientation
primitives and three_sixty_geometry.py for the underlying spatial formulas.

VISIBLE_AREA PROVENANCE (2026-08-20 scientific closure correction): a prior
audit found `three_sixty_frames.visible_area` empty for 100% of the published
BigQuery corpus and concluded (INCORRECTLY) that this was a permanent source
limitation, removing visibility gating below. Root-cause investigation now
shows this was a reproducible ADAPTER_BUG in the BigQuery publish step
(`publish_core.py`'s Parquet load job was missing `enable_list_inference`),
NOT a defect in the raw StatsBomb source or the Bronze->Silver Parquet build:
- RAW 360 source (sampled 17,689 frames / 5 matches): 100% non-empty.
- Staged Silver Parquet (full corpus, 535,283 frames): 100% non-empty.
- Published BigQuery `three_sixty_frames` (pre-fix): 0% non-empty (bug,
  confirmed reproducible and fixed; see publish_core.py).
The eligibility rules below have been REVERTED to the scientifically correct
design: ball/actor-referenced and region-based candidates require the query
point/region to be demonstrably inside the frame's `visible_area` polygon
(`point_observable`/`region_observable`). Until BigQuery is republished from
the already-fixed pipeline, these gates will correctly return null for the
whole current corpus rather than silently treating an unobserved region as
zero. See CxG E13/F1-F15 closure report for the BLOCKED_ON_SILVER_REPUBLICATION
status this implies for production freeze.
"""

from __future__ import annotations

from opponent_adjusted.features.cxg.contracts import three_sixty_candidate_names_for_families
from opponent_adjusted.features.cxg.geometry import (
    BOX_X_MIN,
    BOX_Y_MAX,
    BOX_Y_MIN,
    GOAL_X,
    shot_geometry,
)
from opponent_adjusted.features.cxg.three_sixty_frame import (
    Frame,
    FramePlayer,
    defending_keeper,
    find_actor,
    point_observable,
    region_observable,
)
from opponent_adjusted.features.cxg import three_sixty_geometry as geo

F1_F5_FAMILY_IDS = ("F1", "F2", "F3", "F4", "F5")
F1_F5_FEATURES = three_sixty_candidate_names_for_families(F1_F5_FAMILY_IDS)

# Probe corners inset by a small numerical-robustness margin so exact box-boundary
# points do not trigger ray-casting edge ambiguity in point_in_polygon.
_PROBE_INSET = 0.5
_BOX_PROBE_CORNERS = (
    (BOX_X_MIN + _PROBE_INSET, BOX_Y_MIN + _PROBE_INSET),
    (GOAL_X - _PROBE_INSET, BOX_Y_MIN + _PROBE_INSET),
    (GOAL_X - _PROBE_INSET, BOX_Y_MAX - _PROBE_INSET),
    (BOX_X_MIN + _PROBE_INSET, BOX_Y_MAX - _PROBE_INSET),
)


def derive_static_360_context(
    frame: Frame | None,
    oriented_players: tuple[FramePlayer, ...] | None,
    ball_x: float | None,
    ball_y: float | None,
) -> dict[str, object | None]:
    """Derive F1-F5 for one shot's own 360 frame. `oriented_players` must already
    be re-expressed relative to the shot team (see three_sixty_frame.orient_players).
    """
    values: dict[str, object | None] = {name: None for name in F1_F5_FEATURES}
    if frame is None or oriented_players is None:
        return values

    ball_geometry = shot_geometry(ball_x, ball_y)
    ball_valid = ball_geometry.geometry_valid
    ball_observable = ball_valid and point_observable(frame, ball_x, ball_y)

    attackers = geo.outfield_attackers(oriented_players)
    defenders = geo.outfield_defenders(oriented_players)
    opposition_all = geo.all_opposition(oriented_players)

    # F1 -----------------------------------------------------------------
    if ball_observable:
        values["nearest_defender_distance"] = geo.nearest_defender_distance_m(
            ball_x, ball_y, opposition_all
        )
        values["defenders_within_3m"] = geo.count_within_radius_m(ball_x, ball_y, defenders, 3.0)
        values["defenders_within_5m"] = geo.count_within_radius_m(ball_x, ball_y, defenders, 5.0)
        values["defenders_within_8m"] = geo.count_within_radius_m(ball_x, ball_y, defenders, 8.0)
        values["local_defensive_density"] = values["defenders_within_8m"] / (
            3.14159265358979 * geo.LOCAL_DENSITY_RADIUS_M**2
        )
        cone = geo.Cone(ball_x, ball_y)
        values["defenders_between_ball_and_goal"] = len(geo.defenders_in_cone(defenders, cone))

    actor = find_actor(oriented_players)
    if actor is not None and point_observable(frame, actor.x, actor.y):
        values["actor_space"] = geo.nearest_defender_distance_m(actor.x, actor.y, opposition_all)

    # F2 -----------------------------------------------------------------
    if frame.has_visible_area and len(defenders) >= geo.MIN_DEFENDERS_FOR_BLOCK:
        values["defensive_line_depth"] = geo.defensive_line_depth(defenders)
        centroid = geo.defensive_centroid(defenders)
        if centroid is not None:
            values["defensive_centroid_x"], values["defensive_centroid_y"] = centroid
        values["defensive_width"] = geo.defensive_width(defenders)
        values["defensive_length"] = geo.defensive_length(defenders)
        values["defensive_compactness"] = geo.defensive_compactness(defenders)
        values["defensive_hull_area"] = geo.defensive_hull_area(defenders)

    # F3 -----------------------------------------------------------------
    if region_observable(frame, _BOX_PROBE_CORNERS):
        values["defenders_in_box"] = geo.count_in_box(defenders)
        values["attackers_in_box"] = geo.count_in_box(attackers)
        values["box_numerical_balance"] = values["attackers_in_box"] - values["defenders_in_box"]
        if ball_valid:
            values["defenders_goal_side"] = sum(1 for d in defenders if d.x < ball_x)
            values["central_defenders_between_ball_and_goal"] = geo.count_in_central_lane(
                defenders, ball_x, GOAL_X
            )

    # F4 -----------------------------------------------------------------
    gk, ambiguous = defending_keeper(oriented_players)
    if gk is not None and not ambiguous:
        values["gk_x"] = gk.x
        values["gk_y"] = gk.y
        values["gk_depth"] = geo.gk_depth(gk)
        values["gk_lateral_offset"] = geo.gk_lateral_offset(gk)
        values["gk_distance_to_goal_centre"] = geo.gk_distance_to_goal_centre(gk)
        if ball_valid:
            values["gk_distance_to_shooter"] = geo.gk_distance_to_point(gk, ball_x, ball_y)

    # F5 -----------------------------------------------------------------
    if ball_observable:
        blockers = opposition_all
        values["estimated_goalface_occlusion"] = geo.estimated_goalface_occlusion(
            ball_x, ball_y, blockers
        )
        cone = geo.Cone(ball_x, ball_y)
        values["shot_corridor_occlusion"] = geo.shot_corridor_occlusion(
            ball_x, ball_y, blockers, cone
        )
        values["visible_goal_angle_proxy"] = geo.visible_goal_angle_proxy(
            ball_x, ball_y, ball_geometry.shot_angle_rad, blockers, cone
        )
        values["goal_mouth_defender_count"] = geo.count_in_central_lane(blockers, ball_x, GOAL_X)

    return values
