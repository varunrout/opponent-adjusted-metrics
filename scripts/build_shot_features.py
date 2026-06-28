"""Build shot-level geometry and basic context features.

This script scans raw StatsBomb events, materializes minimal Event and Shot rows
for shots (if missing), and writes geometry features to `shot_features` with a
version tag.
"""

import argparse
from typing import Dict

from opponent_adjusted.db.session import session_scope
from opponent_adjusted.db.models import (
    RawEvent,
    Event,
    Possession,
    Shot,
    ShotFeature,
    Team,
    Match,
)
from opponent_adjusted.features.geometry import calculate_all_geometry_features
from opponent_adjusted.features.context import (
    calculate_game_state,
    calculate_minute_bucket_label,
    calculate_pressure_proxy_score,
)
from opponent_adjusted.ingestion.statsbomb_io import (
    extract_shot_info,
    extract_event_location,
)
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)

DEFENSIVE_ACTION_TYPES = {
    "Pressure",
    "Block",
    "Interception",
    "Clearance",
    "Duel",
    "Tackle",
}


def _event_seconds(event: Event) -> float:
    return float((event.minute or 0) * 60 + (event.second or 0))


def populate_possession_features(session, version: str) -> dict[str, int]:
    """Backfill possession context fields on shot features.

    Possessions are built incrementally from normalized events, so this pass can
    enrich existing shot feature rows after `make build-possessions` without
    re-fetching or re-normalizing source data.
    """

    rows = (
        session.query(ShotFeature, Shot, Event)
        .join(Shot, ShotFeature.shot_id == Shot.id)
        .join(Event, Shot.event_id == Event.id)
        .filter(ShotFeature.version_tag == version)
        .order_by(Shot.match_id, Event.period, Event.minute, Event.second, Event.id)
        .all()
    )
    if not rows:
        stats = {
            "evaluated": 0,
            "updated": 0,
            "missing_possession_number": 0,
            "missing_possession_row": 0,
        }
        logger.info("Shot possession context: %s", stats)
        return stats

    possession_keys = {
        (int(event.match_id), int(event.possession))
        for _, _, event in rows
        if event.possession is not None
    }
    possession_rows = []
    if possession_keys:
        possession_rows = (
            session.query(Possession)
            .filter(Possession.match_id.in_({match_id for match_id, _ in possession_keys}))
            .all()
        )
    possessions = {(int(row.match_id), int(row.possession_number)): row for row in possession_rows}

    updates = 0
    missing_possession_number = 0
    missing_possession_row = 0
    for feature, _shot, event in rows:
        possession = None
        if event.possession is not None:
            possession = possessions.get((int(event.match_id), int(event.possession)))
            if possession is None:
                missing_possession_row += 1
        else:
            missing_possession_number += 1

        previous_events = []
        if event.possession is not None:
            previous_events = (
                session.query(Event)
                .filter(
                    Event.match_id == event.match_id,
                    Event.possession == event.possession,
                    Event.id < event.id,
                )
                .order_by(
                    Event.period.desc(),
                    Event.minute.desc(),
                    Event.second.desc(),
                    Event.id.desc(),
                )
                .limit(5)
                .all()
            )

        previous_event = previous_events[0] if previous_events else None
        recent_def_actions_count = sum(
            1 for previous in previous_events if previous.type in DEFENSIVE_ACTION_TYPES
        )
        sequence_length = possession.event_count if possession else None
        duration = possession.duration_seconds if possession else None
        previous_action_gap = (
            max(0.0, _event_seconds(event) - _event_seconds(previous_event))
            if previous_event is not None
            else (0.0 if event.possession is not None else None)
        )
        pressure_proxy_score = (
            calculate_pressure_proxy_score(
                bool(event.under_pressure),
                recent_def_actions_count,
                float(duration or 0.0),
            )
            if event.possession is not None
            else None
        )

        changed = False
        if feature.possession_sequence_length != sequence_length:
            feature.possession_sequence_length = sequence_length
            changed = True
        if feature.possession_duration != duration:
            feature.possession_duration = duration
            changed = True
        if feature.previous_action_gap != previous_action_gap:
            feature.previous_action_gap = previous_action_gap
            changed = True
        if feature.recent_def_actions_count != recent_def_actions_count:
            feature.recent_def_actions_count = recent_def_actions_count
            changed = True
        if feature.pressure_proxy_score != pressure_proxy_score:
            feature.pressure_proxy_score = pressure_proxy_score
            changed = True
        if changed:
            updates += 1

    stats = {
        "evaluated": len(rows),
        "updated": updates,
        "missing_possession_number": missing_possession_number,
        "missing_possession_row": missing_possession_row,
    }
    logger.info(
        "Shot possession context: evaluated=%d updated=%d missing_possession_number=%d "
        "missing_possession_row=%d",
        stats["evaluated"],
        stats["updated"],
        stats["missing_possession_number"],
        stats["missing_possession_row"],
    )
    return stats


def populate_game_state_features(session, version: str) -> int:
    """Backfill score differential, game state flags, and minute buckets."""

    rows = (
        session.query(ShotFeature, Shot, Event)
        .join(Shot, ShotFeature.shot_id == Shot.id)
        .join(Event, Shot.event_id == Event.id)
        .filter(ShotFeature.version_tag == version)
        .order_by(Shot.match_id, Event.period, Event.minute, Event.second, Event.id)
        .all()
    )

    current_match_id = None
    score_by_team: Dict[int, int] = {}
    updates = 0

    for feature, shot, event in rows:
        if shot.match_id != current_match_id:
            current_match_id = shot.match_id
            score_by_team = {}

        team_score = score_by_team.get(shot.team_id, 0)
        opp_score = score_by_team.get(shot.opponent_team_id, 0)
        score_diff = team_score - opp_score
        game_state = calculate_game_state(score_diff)
        minute_bucket = (
            calculate_minute_bucket_label(int(event.minute)) if event.minute is not None else None
        )

        changed = False
        if feature.score_diff_at_shot != score_diff:
            feature.score_diff_at_shot = score_diff
            changed = True
        if feature.is_leading != game_state["is_leading"]:
            feature.is_leading = game_state["is_leading"]
            changed = True
        if feature.is_trailing != game_state["is_trailing"]:
            feature.is_trailing = game_state["is_trailing"]
            changed = True
        if feature.is_drawing != game_state["is_drawing"]:
            feature.is_drawing = game_state["is_drawing"]
            changed = True
        if minute_bucket and feature.minute_bucket != minute_bucket:
            feature.minute_bucket = minute_bucket
            changed = True

        if changed:
            updates += 1

        outcome = (shot.outcome or "").strip().lower()
        if outcome == "goal":
            score_by_team[shot.team_id] = team_score + 1
        else:
            score_by_team.setdefault(shot.team_id, team_score)
        score_by_team.setdefault(shot.opponent_team_id, opp_score)

    logger.info("Updated %d shot features with game state context", updates)
    return updates


def get_or_create_event(session, raw_ev: RawEvent, team_id: int | None) -> Event:
    ev = session.query(Event).filter_by(raw_event_id=raw_ev.id).first()
    if ev:
        return ev

    # Minimal event fields
    x, y = extract_event_location(raw_ev.raw_json)
    ts = raw_ev.raw_json.get("timestamp")
    possession = raw_ev.raw_json.get("possession")
    player_id = None  # optional; can be filled by a separate normalization step

    ev = Event(
        raw_event_id=raw_ev.id,
        match_id=raw_ev.match_id,
        team_id=team_id if team_id is not None else 0,
        player_id=player_id,
        type="Shot",
        period=raw_ev.period,
        minute=raw_ev.minute,
        second=raw_ev.second,
        timestamp=ts,
        possession=possession,
        location_x=x,
        location_y=y,
        under_pressure=bool(raw_ev.raw_json.get("under_pressure", False)),
        outcome=None,
    )
    session.add(ev)
    session.flush()
    return ev


def get_team_ids(session, raw_ev: RawEvent) -> tuple[int | None, int | None]:
    team_dict = raw_ev.raw_json.get("team", {})
    sb_team_id = team_dict.get("id")
    team = None
    if sb_team_id is not None:
        team = session.query(Team).filter_by(statsbomb_team_id=sb_team_id).first()

    match = session.query(Match).filter_by(id=raw_ev.match_id).first()
    if not match:
        return (team.id if team else None, None)

    if team and team.id == match.home_team_id:
        return (team.id, match.away_team_id)
    if team and team.id == match.away_team_id:
        return (team.id, match.home_team_id)
    # Fallback: unknown mapping
    return (team.id if team else None, None)


def upsert_shot_and_features(session, raw_ev: RawEvent, version: str) -> str:
    team_id, opponent_team_id = get_team_ids(session, raw_ev)
    ev = get_or_create_event(session, raw_ev, team_id)

    shot_info = extract_shot_info(raw_ev.raw_json) or {}
    outcome = shot_info.get("outcome") or "Unknown"
    is_blocked = outcome == "Blocked"

    shot = session.query(Shot).filter_by(event_id=ev.id).first()
    if not shot:
        shot = Shot(
            event_id=ev.id,
            match_id=ev.match_id,
            team_id=team_id if team_id is not None else 0,
            player_id=None,
            opponent_team_id=opponent_team_id if opponent_team_id is not None else 0,
            statsbomb_xg=shot_info.get("statsbomb_xg"),
            body_part=shot_info.get("body_part"),
            technique=shot_info.get("technique"),
            shot_type=shot_info.get("shot_type"),
            outcome=outcome,
            first_time=bool(shot_info.get("first_time", False)),
            is_blocked=is_blocked,
        )
        session.add(shot)
        session.flush()

    # Compute geometry features from location
    x = shot_info.get("location_x")
    y = shot_info.get("location_y")
    if x is None or y is None:
        x, y = extract_event_location(raw_ev.raw_json)

    if x is None or y is None:
        # Cannot compute geometry without location
        return "skipped"

    geom = calculate_all_geometry_features(float(x), float(y))

    existing_feat = (
        session.query(ShotFeature).filter_by(shot_id=shot.id, version_tag=version).first()
    )
    if not existing_feat:
        feat = ShotFeature(
            shot_id=shot.id,
            version_tag=version,
            shot_distance=geom.get("shot_distance"),
            shot_angle=geom.get("shot_angle"),
            centrality=geom.get("centrality"),
            distance_to_goal_line=geom.get("distance_to_goal_line"),
        )
        session.add(feat)
        session.flush()
        return "inserted"

    changed = False
    for column, value in (
        ("shot_distance", geom.get("shot_distance")),
        ("shot_angle", geom.get("shot_angle")),
        ("centrality", geom.get("centrality")),
        ("distance_to_goal_line", geom.get("distance_to_goal_line")),
    ):
        if getattr(existing_feat, column) != value:
            setattr(existing_feat, column, value)
            changed = True
    if changed:
        session.flush()
        return "updated"
    return "unchanged"


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for building shot features.

    Accepts an optional ``argv`` list so that tests or other callers
    can invoke it without `argparse` attempting to parse their
    command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Build shot features")
    parser.add_argument("--version", default="v1", help="Feature version tag (default: v1)")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing shot features for the version before rebuilding",
    )
    args = parser.parse_args(argv)

    inserted = 0
    geometry_updates = 0
    unchanged = 0
    skipped = 0
    processed = 0
    context_updates = 0
    possession_stats = {
        "evaluated": 0,
        "updated": 0,
        "missing_possession_number": 0,
        "missing_possession_row": 0,
    }

    with session_scope() as session:
        if args.rebuild:
            deleted = (
                session.query(ShotFeature)
                .filter_by(version_tag=args.version)
                .delete(synchronize_session=False)
            )
            logger.info("Deleted %d existing shot features for version %s", deleted, args.version)

        q = session.query(RawEvent).filter_by(type="Shot")
        total = q.count()
        logger.info("Building features for %d shot events...", total)

        batch_size = 5000
        offset = 0
        while True:
            batch = q.order_by(RawEvent.id).offset(offset).limit(batch_size).all()
            if not batch:
                break
            for raw_ev in batch:
                result = upsert_shot_and_features(session, raw_ev, args.version)
                if result == "inserted":
                    inserted += 1
                elif result == "updated":
                    geometry_updates += 1
                elif result == "unchanged":
                    unchanged += 1
                else:
                    skipped += 1
                processed += 1
                if processed % 5000 == 0:
                    logger.info(
                        "Processed %d/%d shots; inserted=%d geometry_updated=%d unchanged=%d "
                        "skipped=%d",
                        processed,
                        total,
                        inserted,
                        geometry_updates,
                        unchanged,
                        skipped,
                    )
            offset += batch_size

        context_updates = populate_game_state_features(session, args.version)
        possession_stats = populate_possession_features(session, args.version)

    logger.info(
        "Shot feature build complete. inserted=%d geometry_updated=%d unchanged=%d skipped=%d "
        "game_state_updated=%d possession_context_updated=%d possession_missing_number=%d "
        "possession_missing_row=%d",
        inserted,
        geometry_updates,
        unchanged,
        skipped,
        context_updates,
        possession_stats["updated"],
        possession_stats["missing_possession_number"],
        possession_stats["missing_possession_row"],
    )


if __name__ == "__main__":
    main()
