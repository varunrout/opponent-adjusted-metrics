"""Build shot-level geometry and basic context features.

This script scans raw StatsBomb events, materializes minimal Event and Shot rows
for shots (if missing), and writes geometry features to `shot_features` with a
version tag.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.db.session import session_scope
from opponent_adjusted.db.models import (
    RawEvent,
    Event,
    Shot,
    ShotFeature,
    Team,
    Match,
)
from opponent_adjusted.features.geometry import calculate_all_geometry_features
from opponent_adjusted.ingestion.statsbomb_io import (
    extract_shot_info,
    extract_event_location,
)
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


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


def upsert_shot_and_features(session, raw_ev: RawEvent, version: str) -> bool:
    # Skip if we already have a shot feature for this raw event and version
    existing_ev = session.query(Event).filter_by(raw_event_id=raw_ev.id).first()
    if existing_ev:
        existing_shot = session.query(Shot).filter_by(event_id=existing_ev.id).first()
        if existing_shot:
            existing_feat = (
                session.query(ShotFeature)
                .filter_by(shot_id=existing_shot.id, version_tag=version)
                .first()
            )
            if existing_feat:
                return False

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
        return False

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
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Build shot features")
    parser.add_argument("--version", default="v1", help="Feature version tag (default: v1)")
    args = parser.parse_args()

    inserted = 0
    processed = 0

    with session_scope() as session:
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
                if upsert_shot_and_features(session, raw_ev, args.version):
                    inserted += 1
                processed += 1
                if processed % 5000 == 0:
                    logger.info("Processed %d/%d shots; features inserted so far: %d", processed, total, inserted)
            offset += batch_size

    logger.info("Shot feature build complete. Inserted: %d", inserted)


if __name__ == "__main__":
    main()
