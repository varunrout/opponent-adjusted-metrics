"""Feature selection for defensive profile clustering (CxG+ Phase 2).

Source tables: `oam_features.cxg_defensive_360_features` (F1/F2/F3 governed
candidates) and `oam_features.cxg_line_shape_360_features` (F2/F5 governed
candidates) -- see `opponent_adjusted.features.cxg.contracts` for the
governed F-family taxonomy this selection is scoped against. No geometry is
recomputed; only existing Gold columns are read.

Selection principle: keep only *static pre-shot shape* scalars (F1 local
pressure state, F2 defensive block geometry, F5 goal exposure /
shooting-corridor state). Exclude by family:
  - F3 (box occupation): >95% null on the train split (see below) -- not a
    usable clustering input, not something to paper over with imputation.
  - F6-F9, F13 (shape/pressure/GK *evolution*, i.e. `_delta` /
    `_state_change_rate` columns): these describe *change*, not the
    pre-shot state itself.
  - F10 (shooter/receiver space): describes the attacking player, not
    defensive shape.
  - F11 (bypass/line-break proxies), F12 (rest-defence/transition
    structure), F14 (sequence-level dynamics), F15 (composite proxies):
    transition/possession-history proxies, not the shot-instant defensive
    shape, and several carry their own extra eligibility gates
    (`methodology_flags` in `features/cxg/contracts.py`) layered on top of
    the base geometry eligibility.
  - F4/F9 (goalkeeper spatial state / movement): not present in either
    source table named for this task (`cxg_defensive_360_features`,
    `cxg_line_shape_360_features`); they live in `cxg_goalkeeper_360_features`,
    out of scope here.

Within the F1/F2/F3/F5 candidates, near-duplicate columns (Pearson |r| >=
0.85 on the TRAIN split only, per `docs/cxg_split_policy_and_parallel_plan.md`
step 4's "redundancy/correlation on train only" convention) are pruned to one
representative each:
  - `actor_space` <-> `nearest_defender_distance`: r = 1.000 (identical for
    this shot-only actor==shooter case) -> keep `nearest_defender_distance`
    (the governed F1 primary name).
  - `defenders_within_8m` <-> `local_defensive_density`: r = 1.000
    (`local_defensive_density` is `defenders_within_8m` normalised by area,
    per `three_sixty_static.py`) -> keep `local_defensive_density` (already
    scale-normalised).
  - `defensive_centroid_x` <-> `defensive_line_depth`: r = 0.852 -> keep
    `defensive_line_depth` (governed F2 primary name for block depth),
    drop `defensive_centroid_x`; `defensive_centroid_y` is kept as the
    independent lateral-position signal.
  - `defensive_compactness` <-> `defensive_length`: r = 0.866, and
    `defensive_compactness` <-> `defensive_hull_area`: r = 0.917.
    `defensive_compactness` is literally `defensive_width * defensive_length`
    (see `three_sixty_geometry.py::defensive_compactness`), i.e. a product
    of two primitives already retained below -> drop both
    `defensive_compactness` and `defensive_hull_area` (a correlated
    composite of the same footprint), keep the primitive `defensive_width`
    and `defensive_length` dimensions instead (more interpretable, no
    information loss vs. keeping their product).

  Remaining moderate correlations (0.7 <= |r| < 0.85) are retained
  deliberately -- e.g. `defenders_within_3m`/`defenders_within_5m` (r=0.74),
  `defensive_line_depth`/`defensive_length` (r=-0.76): these are
  conceptually distinct measurements that happen to covary in real defensive
  shapes (a deep low block is naturally more vertically compressed), not
  near-duplicate columns.

All 13 selected columns share an identical 2.41% null rate on the train
split -- a single shared eligibility gate (visible_area-gated geometry
eligibility, per the "Revert F1/F2/F3/F5 to honest visible_area-gated
eligibility" governance decision), not independent per-column missingness.
"""

from __future__ import annotations

DEFENSIVE_360_TABLE = "cxg_defensive_360_features"
LINE_SHAPE_360_TABLE = "cxg_line_shape_360_features"

# Columns read from cxg_defensive_360_features.
DEFENSIVE_360_CANDIDATES: tuple[str, ...] = (
    "nearest_defender_distance",
    "defenders_within_3m",
    "defenders_within_5m",
    "defenders_within_8m",
    "local_defensive_density",
    "defenders_between_ball_and_goal",
    "actor_space",
)

# Columns read from cxg_line_shape_360_features.
LINE_SHAPE_360_CANDIDATES: tuple[str, ...] = (
    "defensive_line_depth",
    "defensive_centroid_x",
    "defensive_centroid_y",
    "defensive_width",
    "defensive_length",
    "defensive_compactness",
    "defensive_hull_area",
    "estimated_goalface_occlusion",
    "shot_corridor_occlusion",
    "visible_goal_angle_proxy",
    "goal_mouth_defender_count",
)

# Dropped as near-duplicates (|r| >= 0.85 on train), see module docstring.
DROPPED_REDUNDANT: tuple[str, ...] = (
    "actor_space",
    "defenders_within_8m",
    "defensive_centroid_x",
    "defensive_compactness",
    "defensive_hull_area",
)

# Final clustering input: 13 static pre-shot defensive-shape scalars.
CLUSTER_FEATURES: tuple[str, ...] = (
    "nearest_defender_distance",
    "defenders_within_3m",
    "defenders_within_5m",
    "local_defensive_density",
    "defenders_between_ball_and_goal",
    "defensive_line_depth",
    "defensive_centroid_y",
    "defensive_width",
    "defensive_length",
    "estimated_goalface_occlusion",
    "shot_corridor_occlusion",
    "visible_goal_angle_proxy",
    "goal_mouth_defender_count",
)

assert set(CLUSTER_FEATURES) == (
    set(DEFENSIVE_360_CANDIDATES) | set(LINE_SHAPE_360_CANDIDATES)
) - set(DROPPED_REDUNDANT)
