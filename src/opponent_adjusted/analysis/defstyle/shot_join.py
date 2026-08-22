"""Shot -> nearest-defender -> archetype resolution (CxG+ Phase B, step 5).

The "nearest defender" definition is NOT redefined here. It is the one this
codebase already uses for CxG+ freeze-frame work, materialised in
`oam_analysis.cxg_defensive_involvement_v1` by
`scripts/materialize_cxg_defensive_involvement.py`: from
`oam_core.shot_freeze_frame_players`, take rows with `teammate = FALSE` and
a non-null `player_id`, and pick the minimum Euclidean distance to the shot
location, tie-broken by `freeze_frame_player_ordinal`; penalty-shootout
(period 5) shots are excluded. This module consumes that table's resolved
`player_id` rather than recomputing any geometry.

Only the archetype STRING crosses into shot level. `player_id` is used to
perform the lookup and is then dropped -- see `contracts.py`.

Every non-assignment is a NULL with a recorded reason. There is no default
archetype and no nearest-fallback: a shot whose defender we cannot
characterise is honestly unknown, not quietly assigned to the biggest
cluster.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from opponent_adjusted.analysis.defstyle.features import (
    NULL_REASON_BELOW_THRESHOLD,
    NULL_REASON_NO_DEFENDER,
    NULL_REASON_NOT_CLUSTERED,
)


@dataclass(frozen=True)
class ShotArchetype:
    """Resolution outcome for one shot.

    Exactly one of `archetype` / `null_reason` is populated. `null_reason` is
    audit-only: the Gold column downstream carries `archetype` alone, so a
    null there is simply NULL.
    """

    archetype: str | None
    null_reason: str | None

    def __post_init__(self) -> None:
        if (self.archetype is None) == (self.null_reason is None):
            raise ValueError("exactly one of archetype / null_reason must be set")


def resolve_shot_archetype(
    nearest_defender_player_id: int | None,
    archetype_by_player: Mapping[int, str | None],
) -> ShotArchetype:
    """Resolve one shot's `nearest_defender_style_archetype`.

    `archetype_by_player` maps every player present in
    `cxg_defender_style_clusters_v1` to their archetype, or to None when the
    player is in that table but was not clustered (below threshold).
    """
    if nearest_defender_player_id is None:
        return ShotArchetype(None, NULL_REASON_NO_DEFENDER)
    if nearest_defender_player_id not in archetype_by_player:
        return ShotArchetype(None, NULL_REASON_NOT_CLUSTERED)
    archetype = archetype_by_player[nearest_defender_player_id]
    if archetype is None:
        return ShotArchetype(None, NULL_REASON_BELOW_THRESHOLD)
    return ShotArchetype(archetype, None)


def coverage_summary(resolutions: list[ShotArchetype]) -> dict[str, int]:
    """Row-count reconciliation: assigned vs each distinct null reason."""
    summary: dict[str, int] = {
        "total": len(resolutions),
        "assigned": sum(1 for r in resolutions if r.archetype is not None),
        NULL_REASON_NO_DEFENDER: 0,
        NULL_REASON_NOT_CLUSTERED: 0,
        NULL_REASON_BELOW_THRESHOLD: 0,
    }
    for resolution in resolutions:
        if resolution.null_reason is not None:
            summary[resolution.null_reason] += 1
    return summary
