"""CxG+ F1-F5: current-shot 360 snapshot families (cxg_360_context_v1).

All features below are computed from the SHOT's OWN linked 360 frame only
(current-shot snapshot; no sequence/history). Eligibility gates are documented
per family; a null here always means "not observable/not eligible", never a
fabricated zero. See three_sixty_frame.py for the visibility/orientation
primitives and three_sixty_geometry.py for the underlying spatial formulas.

METHODOLOGY_BUG correction (locked before production acceptance, verified via
direct BigQuery inspection on 2026-08-20): the governed `three_sixty_frames`
table's `visible_area` column is empty for 100% of rows in the currently
published corpus (data_version b0bc9f22dd77c206ddedc1d742893b3bbe64baec,
silver_schema_version statsbomb_silver_v1_2). Gating F1/F2/F3/F5 on
`point_observable`/`region_observable` (which require a populated
visible_area polygon) therefore made those families universally null against
real data -- not a genuine source limitation, but an over-conservative
eligibility rule combined with an unusable column. The eligibility rules
below instead rely only on what the frame actually, directly observed: the
ball/actor point is treated as observable because an event-linked 360 frame
is by construction captured around its own source event (the ball/actor was
the reason the frame exists); defensive-block/box-occupancy features require
only that the frame exists and (for F2) that at least one defender was
actually recorded in it. `point_observable`/`region_observable` remain
available and unit-tested in three_sixty_frame.py so a stricter, evidence-
based gate can be reinstated if/when visible_area is populated upstream.
"""

from __future__ import annotations

from opponent_adjusted.features.cxg.contracts import three_sixty_candidate_names_for_families
from opponent_adjusted.features.cxg.geometry import GOAL_X, shot_geometry
from opponent_adjusted.features.cxg.three_sixty_frame import (
    Frame,
    FramePlayer,
    defending_keeper,
    find_actor,
)
from opponent_adjusted.features.cxg import three_sixty_geometry as geo

F1_F5_FAMILY_IDS = ("F1", "F2", "F3", "F4", "F5")
F1_F5_FEATURES = three_sixty_candidate_names_for_families(F1_F5_FAMILY_IDS)


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
    # The frame's own ball/actor point is presumed observable: an event-linked 360 frame is
    # captured around the event that generated it (see module docstring METHODOLOGY_BUG note).
    ball_observable = ball_valid

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
    if actor is not None:
        values["actor_space"] = geo.nearest_defender_distance_m(actor.x, actor.y, opposition_all)

    # F2 -----------------------------------------------------------------
    if len(defenders) >= geo.MIN_DEFENDERS_FOR_BLOCK:
        values["defensive_line_depth"] = geo.defensive_line_depth(defenders)
        centroid = geo.defensive_centroid(defenders)
        if centroid is not None:
            values["defensive_centroid_x"], values["defensive_centroid_y"] = centroid
        values["defensive_width"] = geo.defensive_width(defenders)
        values["defensive_length"] = geo.defensive_length(defenders)
        values["defensive_compactness"] = geo.defensive_compactness(defenders)
        values["defensive_hull_area"] = geo.defensive_hull_area(defenders)

    # F3 -----------------------------------------------------------------
    # Frame existence alone is the eligibility gate; see module docstring METHODOLOGY_BUG note.
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
