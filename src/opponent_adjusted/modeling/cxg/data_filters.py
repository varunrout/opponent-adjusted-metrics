"""Filtering utilities for CxG modeling datasets.

Implements the governance and inclusion rules outlined in the modelling
roadmap so that downstream models only see high-integrity shots.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import select

from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Competition, Match
from opponent_adjusted.db.session import SessionLocal


@dataclass
class FilterSummary:
    total_rows: int
    removed_missing_geometry: int = 0
    removed_missing_location: int = 0
    removed_missing_xg: int = 0
    removed_penalties: int = 0
    removed_own_goals: int = 0
    removed_out_of_scope: int = 0

    def to_dict(self) -> dict:
        summary = asdict(self)
        summary["total_after"] = summary["total_rows"] - sum(
            summary[key]
            for key in [
                "removed_missing_geometry",
                "removed_missing_location",
                "removed_missing_xg",
                "removed_penalties",
                "removed_own_goals",
                "removed_out_of_scope",
            ]
        )
        return summary


def _load_allowed_match_ids() -> Optional[set[int]]:
    """Return match IDs that fall inside the configured competition scope."""

    allowed_comp_ids = {
        comp.get("competition_id") for comp in settings.competitions if comp.get("competition_id")
    }
    if not allowed_comp_ids:
        return None

    with SessionLocal() as session:
        stmt = (
            select(Match.id)
            .join(Competition, Competition.id == Match.competition_id)
            .where(Competition.statsbomb_competition_id.in_(allowed_comp_ids))
        )
        rows = session.execute(stmt).scalars().all()
    return set(rows)


def apply_modeling_filters(
    df: pd.DataFrame,
    *,
    allowed_match_ids: Optional[Iterable[int]] = None,
    drop_penalties: bool = True,
    require_geometry: bool = True,
    require_location: bool = True,
    require_xg: bool = True,
) -> tuple[pd.DataFrame, FilterSummary]:
    """Apply high-level modelling filters and return the filtered frame + summary."""

    filtered = df.copy()
    summary = FilterSummary(total_rows=len(filtered))

    if require_geometry:
        geom_mask = filtered["shot_distance"].notna() & filtered["shot_angle"].notna()
        summary.removed_missing_geometry = int((~geom_mask).sum())
        filtered = filtered[geom_mask]

    if require_location:
        loc_mask = filtered["location_x"].notna() & filtered["location_y"].notna()
        summary.removed_missing_location = int((~loc_mask).sum())
        filtered = filtered[loc_mask]

    if require_xg and "statsbomb_xg" in filtered.columns:
        xg_mask = filtered["statsbomb_xg"].notna()
        summary.removed_missing_xg = int((~xg_mask).sum())
        filtered = filtered[xg_mask]

    if drop_penalties and "set_piece_category" in filtered.columns:
        pen_mask = filtered["set_piece_category"].astype(str) == "Penalty"
        summary.removed_penalties = int(pen_mask.sum())
        filtered = filtered[~pen_mask]

    if "shot_outcome" in filtered.columns:
        own_goal_mask = filtered["shot_outcome"].str.lower().eq("own goal")
        summary.removed_own_goals = int(own_goal_mask.sum())
        filtered = filtered[~own_goal_mask]

    if allowed_match_ids:
        allowed_set = set(allowed_match_ids)
        scope_mask = filtered["match_id"].isin(allowed_set)
        summary.removed_out_of_scope = int((~scope_mask).sum())
        filtered = filtered[scope_mask]

    filtered = filtered.drop_duplicates("shot_id")
    return filtered.reset_index(drop=True), summary


def get_filter_scope() -> Optional[set[int]]:
    """Expose the scoped match ids using the project settings."""

    try:
        return _load_allowed_match_ids()
    except Exception:
        # Swallow DB errors gracefully; caller can proceed without competition filter.
        return None
