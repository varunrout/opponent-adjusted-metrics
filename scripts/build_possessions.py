"""Build possession rows from normalized events.

This is an incremental database step: it reads the existing normalized `events`
table and materializes one row per StatsBomb match/possession into
`possessions`. It does not fetch, ingest, or normalize raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import sqlalchemy as sa

from opponent_adjusted.db.models import Event, Possession, RawEvent, Team
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PossessionBuildSummary:
    """Summary of a possession table rebuild."""

    events_loaded: int
    possessions_built: int
    rows_deleted: int
    rows_inserted: int
    final_possession_count: int


def _chunked(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_team_lookup(session: Any) -> dict[int, int]:
    return dict(session.query(Team.statsbomb_team_id, Team.id).all())


def _load_start_event_context(
    session: Any,
    event_ids: list[int],
    chunk_size: int,
) -> dict[int, tuple[int, dict[str, Any]]]:
    context: dict[int, tuple[int, dict[str, Any]]] = {}
    for chunk in _chunked(event_ids, chunk_size):
        rows = (
            session.query(Event.id, Event.team_id, RawEvent.raw_json)
            .join(RawEvent, Event.raw_event_id == RawEvent.id)
            .filter(Event.id.in_(chunk))
            .all()
        )
        for event_id, event_team_id, raw_json in rows:
            context[int(event_id)] = (int(event_team_id), raw_json or {})
    return context


def _resolve_possession_team_id(
    raw_json: dict[str, Any],
    event_team_id: int,
    team_lookup: dict[int, int],
) -> int:
    possession_team = raw_json.get("possession_team") or {}
    statsbomb_team_id = possession_team.get("id")
    if statsbomb_team_id is not None and statsbomb_team_id in team_lookup:
        return team_lookup[statsbomb_team_id]
    return event_team_id


def build_possessions(session: Any, chunk_size: int = 5000) -> PossessionBuildSummary:
    """Rebuild `possessions` from normalized events.

    The normalized `Event.possession` column comes from StatsBomb's possession
    number. StatsBomb also carries a possession-team field in raw JSON; that is
    preferred over event acting team because defensive actions can occur inside
    the attacking team's possession.
    """

    events_loaded = int(
        session.query(sa.func.count(Event.id)).filter(Event.possession.is_not(None)).scalar() or 0
    )
    logger.info(
        "Possession build: loaded %d normalized events with possession numbers", events_loaded
    )

    clock_seconds = (Event.minute * 60) + Event.second
    grouped_rows = (
        session.query(
            Event.match_id.label("match_id"),
            Event.possession.label("possession_number"),
            sa.func.min(Event.id).label("start_event_id"),
            sa.func.max(Event.id).label("end_event_id"),
            sa.func.min(Event.minute).label("start_minute"),
            sa.func.max(Event.minute).label("end_minute"),
            sa.func.min(clock_seconds).label("start_seconds"),
            sa.func.max(clock_seconds).label("end_seconds"),
            sa.func.count(Event.id).label("event_count"),
        )
        .filter(Event.possession.is_not(None))
        .group_by(Event.match_id, Event.possession)
        .order_by(Event.match_id, Event.possession)
        .all()
    )

    logger.info("Possession build: built %d possession groups", len(grouped_rows))

    start_event_ids = [int(row.start_event_id) for row in grouped_rows]
    start_context = _load_start_event_context(session, start_event_ids, chunk_size)
    team_lookup = _load_team_lookup(session)

    rows_deleted = int(session.query(Possession).delete(synchronize_session=False))
    if rows_deleted:
        logger.info("Possession build: deleted %d existing possession rows", rows_deleted)

    rows_to_insert: list[Possession] = []
    for row in grouped_rows:
        event_team_id, raw_json = start_context.get(int(row.start_event_id), (0, {}))
        team_id = _resolve_possession_team_id(raw_json, event_team_id, team_lookup)
        rows_to_insert.append(
            Possession(
                match_id=int(row.match_id),
                possession_number=int(row.possession_number),
                team_id=team_id,
                start_event_id=int(row.start_event_id),
                end_event_id=int(row.end_event_id),
                start_minute=int(row.start_minute) if row.start_minute is not None else None,
                end_minute=int(row.end_minute) if row.end_minute is not None else None,
                duration_seconds=float((row.end_seconds or 0) - (row.start_seconds or 0)),
                event_count=int(row.event_count),
            )
        )

    rows_inserted = 0
    for chunk in _chunked(list(range(len(rows_to_insert))), chunk_size):
        session.bulk_save_objects([rows_to_insert[index] for index in chunk])
        rows_inserted += len(chunk)
        session.flush()

    final_count = int(session.query(sa.func.count(Possession.id)).scalar() or 0)
    logger.info(
        "Possession build complete: inserted=%d final_count=%d",
        rows_inserted,
        final_count,
    )

    return PossessionBuildSummary(
        events_loaded=events_loaded,
        possessions_built=len(grouped_rows),
        rows_deleted=rows_deleted,
        rows_inserted=rows_inserted,
        final_possession_count=final_count,
    )


def main() -> None:
    """CLI entrypoint."""

    with session_scope() as session:
        summary = build_possessions(session)

    logger.info("Possession build summary: %s", summary)


if __name__ == "__main__":
    main()
