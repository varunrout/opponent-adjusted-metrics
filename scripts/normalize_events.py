"""Normalize all raw events into the `events` table and populate
specialized event-type detail tables (passes, dribbles, carries, clearances,
duels, blocks, interceptions, pressures, ball_receipts).

This script is idempotent: it skips creating detail rows if they already exist.
It will also upsert players referenced in events if missing.

Usage:
    poetry run python scripts/normalize_events.py [--only-missing] [--from-id N] [--limit N] [--batch-size N] [--fill-missing-detail]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import sqlalchemy as sa

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.db.session import session_scope
from opponent_adjusted.db.models import (
    RawEvent,
    Event,
    Player,
    Team,
    PassEvent,
    DribbleEvent,
    CarryEvent,
    ClearanceEvent,
    DuelEvent,
    BlockEvent,
    InterceptionEvent,
    PressureEvent,
    BallReceiptEvent,
)
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 10000
SUPPORTED_DETAIL = {
    "Pass": "pass",
    "Dribble": "dribble",
    "Carry": "carry",
    "Clearance": "clearance",
    "Duel": "duel",
    "Block": "block",
    "Interception": "interception",
    "Pressure": "pressure",
    "Ball Receipt": "ball_receipt",
    "Ball Receipt*": "ball_receipt",
}


def _extract_location(raw_json):
    loc = raw_json.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None, None


def _get_or_create_player(session, raw_json) -> Optional[int]:
    player_dict = raw_json.get("player") or {}
    sb_player_id = player_dict.get("id")
    name = player_dict.get("name")
    if sb_player_id is None:
        return None
    player = session.query(Player).filter_by(statsbomb_player_id=sb_player_id).first()
    if not player:
        player = Player(statsbomb_player_id=sb_player_id, name=name or f"Player {sb_player_id}")
        session.add(player)
        session.flush()
    return player.id


def _get_team_id(session, raw_ev: RawEvent) -> Optional[int]:
    team_dict = raw_ev.raw_json.get("team", {})
    sb_team_id = team_dict.get("id")
    if sb_team_id is None:
        return None
    team = session.query(Team).filter_by(statsbomb_team_id=sb_team_id).first()
    return team.id if team else None


def _derive_outcome(raw_ev: RawEvent) -> Optional[str]:
    # Attempt to derive a consistent outcome string from type-specific subobject
    t = raw_ev.type
    r = raw_ev.raw_json
    if t == "Pass":
        return (r.get("pass", {}).get("outcome") or {}).get("name")
    if t == "Shot":
        return (r.get("shot", {}).get("outcome") or {}).get("name")
    if t == "Clearance":
        return (r.get("clearance", {}).get("outcome") or {}).get("name")
    if t == "Duel":
        return (r.get("duel", {}).get("outcome") or {}).get("name")
    if t == "Interception":
        return (r.get("interception", {}).get("outcome") or {}).get("name")
    # Others rarely have explicit outcome nomenclature
    return None


def _ensure_event(session, raw_ev: RawEvent) -> Event:
    ev = session.query(Event).filter_by(raw_event_id=raw_ev.id).first()
    if ev:
        return ev
    x, y = _extract_location(raw_ev.raw_json)
    player_id = _get_or_create_player(session, raw_ev.raw_json)
    team_id = _get_team_id(session, raw_ev)
    outcome = _derive_outcome(raw_ev)
    ev = Event(
        raw_event_id=raw_ev.id,
        match_id=raw_ev.match_id,
        team_id=team_id if team_id is not None else 0,
        player_id=player_id,
        type=raw_ev.type,
        period=raw_ev.period,
        minute=raw_ev.minute,
        second=raw_ev.second,
        timestamp=raw_ev.raw_json.get("timestamp"),
        possession=raw_ev.raw_json.get("possession"),
        location_x=x,
        location_y=y,
        under_pressure=bool(raw_ev.raw_json.get("under_pressure", False)),
        outcome=outcome,
    )
    session.add(ev)
    session.flush()
    return ev


def _populate_pass(session, ev: Event, raw_json: dict):
    if session.query(PassEvent).filter_by(event_id=ev.id).first():
        return
    p = raw_json.get("pass", {})
    end_loc = p.get("end_location") or [None, None]
    end_x = end_loc[0] if isinstance(end_loc, list) and len(end_loc) >= 2 else None
    end_y = end_loc[1] if isinstance(end_loc, list) and len(end_loc) >= 2 else None
    recipient_id = None
    rec = p.get("recipient") or {}
    if rec.get("id") is not None:
        # ensure player exists
        recipient_id = _get_or_create_player(session, {"player": rec})
    obj = PassEvent(
        event_id=ev.id,
        length=p.get("length"),
        angle=p.get("angle"),
        end_x=end_x,
        end_y=end_y,
        pass_height=(p.get("height") or {}).get("name"),
        pass_type=(p.get("type") or {}).get("name"),
        body_part=(p.get("body_part") or {}).get("name"),
        outcome=(p.get("outcome") or {}).get("name"),
        recipient_player_id=recipient_id,
        is_cross=bool(p.get("cross", False)),
        is_through_ball=bool(p.get("through_ball", False)),
    )
    session.add(obj)


def _fill_missing_pass_end_locations(session, batch_size: int, limit: Optional[int]) -> int:
    """Backfill PassEvent end_x/end_y for existing rows.

    This is needed when end_x/end_y were added after passes were already normalized.
    """

    filled = 0

    q = (
        session.query(Event.id, RawEvent.raw_json)
        .join(RawEvent, Event.raw_event_id == RawEvent.id)
        .join(PassEvent, PassEvent.event_id == Event.id)
        .filter(RawEvent.type == "Pass")
        .filter(sa.or_(PassEvent.end_x.is_(None), PassEvent.end_y.is_(None)))
        .order_by(Event.id)
    )

    while True:
        take = batch_size if limit is None else max(0, min(batch_size, limit - filled))
        if take == 0:
            break
        batch = q.limit(take).all()
        if not batch:
            break

        for ev_id, raw_json in batch:
            p = (raw_json or {}).get("pass", {})
            end_loc = p.get("end_location") or [None, None]
            end_x = end_loc[0] if isinstance(end_loc, list) and len(end_loc) >= 2 else None
            end_y = end_loc[1] if isinstance(end_loc, list) and len(end_loc) >= 2 else None

            row = session.query(PassEvent).filter_by(event_id=ev_id).first()
            if row is None:
                continue
            if row.end_x is None:
                row.end_x = end_x
            if row.end_y is None:
                row.end_y = end_y
            filled += 1
            if limit is not None and filled >= limit:
                break

        session.commit()

        last_id = batch[-1][0]
        q = q.filter(Event.id > last_id)
        if limit is not None and filled >= limit:
            break

    if filled:
        logger.info("Backfilled end_x/end_y for %d pass rows", filled)
    return filled


def _populate_dribble(session, ev: Event, raw_json: dict):
    if session.query(DribbleEvent).filter_by(event_id=ev.id).first():
        return
    d = raw_json.get("dribble", {})
    obj = DribbleEvent(
        event_id=ev.id,
        outcome=(d.get("outcome") or {}).get("name"),
        overrun=bool(d.get("overrun", False)),
        nutmeg=bool(d.get("nutmeg", False)),
    )
    session.add(obj)


def _populate_carry(session, ev: Event, raw_json: dict):
    if session.query(CarryEvent).filter_by(event_id=ev.id).first():
        return
    c = raw_json.get("carry", {})
    end_loc = c.get("end_location") or [None, None]
    start_x, start_y = ev.location_x, ev.location_y
    end_x = end_loc[0] if isinstance(end_loc, list) and len(end_loc) > 1 else None
    end_y = end_loc[1] if isinstance(end_loc, list) and len(end_loc) > 1 else None
    length = None
    if None not in (start_x, start_y, end_x, end_y):
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx * dx + dy * dy) ** 0.5
    obj = CarryEvent(
        event_id=ev.id,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        length=length,
    )
    session.add(obj)


def _populate_clearance(session, ev: Event, raw_json: dict):
    if session.query(ClearanceEvent).filter_by(event_id=ev.id).first():
        return
    c = raw_json.get("clearance", {})
    obj = ClearanceEvent(
        event_id=ev.id,
        body_part=(c.get("body_part") or {}).get("name"),
        outcome=(c.get("outcome") or {}).get("name"),
        under_pressure=bool(raw_json.get("under_pressure", False)),
    )
    session.add(obj)


def _populate_duel(session, ev: Event, raw_json: dict):
    if session.query(DuelEvent).filter_by(event_id=ev.id).first():
        return
    d = raw_json.get("duel", {})
    obj = DuelEvent(
        event_id=ev.id,
        duel_type=(d.get("type") or {}).get("name"),
        outcome=(d.get("outcome") or {}).get("name"),
        is_success=(d.get("outcome") or {}).get("name") == "Won",
    )
    session.add(obj)


def _populate_block(session, ev: Event, raw_json: dict):
    if session.query(BlockEvent).filter_by(event_id=ev.id).first():
        return
    b = raw_json.get("block", {})
    obj = BlockEvent(
        event_id=ev.id,
        block_type=(b.get("deflection") and "Deflection") or None,
        is_deflection=bool(b.get("deflection", False)),
        is_save_block=bool(b.get("save_block", False)),
    )
    session.add(obj)


def _populate_interception(session, ev: Event, raw_json: dict):
    if session.query(InterceptionEvent).filter_by(event_id=ev.id).first():
        return
    i = raw_json.get("interception", {})
    obj = InterceptionEvent(
        event_id=ev.id,
        outcome=(i.get("outcome") or {}).get("name"),
        is_success=(i.get("outcome") or {}).get("name") == "Won",
    )
    session.add(obj)


def _populate_pressure(session, ev: Event, raw_json: dict):
    if session.query(PressureEvent).filter_by(event_id=ev.id).first():
        return
    p = raw_json.get("pressure", {})
    obj = PressureEvent(
        event_id=ev.id,
        duration_seconds=p.get("duration"),
    )
    session.add(obj)


def _populate_ball_receipt(session, ev: Event, raw_json: dict):
    if session.query(BallReceiptEvent).filter_by(event_id=ev.id).first():
        return
    b = raw_json.get("ball_receipt", {})
    obj = BallReceiptEvent(
        event_id=ev.id,
        clean_receipt=bool(b.get("clean_receipt", False)),
    )
    session.add(obj)


POPULATORS = {
    "Pass": _populate_pass,
    "Dribble": _populate_dribble,
    "Carry": _populate_carry,
    "Clearance": _populate_clearance,
    "Duel": _populate_duel,
    "Block": _populate_block,
    "Interception": _populate_interception,
    "Pressure": _populate_pressure,
    "Ball Receipt": _populate_ball_receipt,
    "Ball Receipt*": _populate_ball_receipt,
}


def _normalize_missing_events(session, batch_size: int, start_id: Optional[int], limit: Optional[int]) -> int:
    processed = 0
    # Build backlog query: only raw events without normalized Event
    q = (
        session.query(RawEvent)
        .outerjoin(Event, Event.raw_event_id == RawEvent.id)
        .filter(Event.id.is_(None))
        .order_by(RawEvent.id)
    )
    if start_id is not None:
        q = q.filter(RawEvent.id >= start_id)

    remaining = q.count()
    logger.info("Backlog raw events to normalize: %d", remaining)

    fetched = 0
    while True:
        batch_q = q
        if limit is not None:
            # Do not exceed overall limit across batches
            take = max(0, min(batch_size, limit - fetched))
            if take == 0:
                break
            batch_q = batch_q.limit(take)
        else:
            batch_q = batch_q.limit(batch_size)

        batch = batch_q.all()
        if not batch:
            break

        for raw_ev in batch:
            ev = _ensure_event(session, raw_ev)
            # Populate detail table (skip Shot which has its own table)
            if raw_ev.type in POPULATORS and raw_ev.type != "Shot":
                POPULATORS[raw_ev.type](session, ev, raw_ev.raw_json)
            processed += 1
        session.commit()

        fetched += len(batch)
        if limit is not None and fetched >= limit:
            break

        # Rebuild q for next page starting after last id processed
        last_id = batch[-1].id
        q = q.filter(RawEvent.id > last_id)

        if processed % (batch_size * 2) == 0:
            logger.info("Processed %d new events...", processed)

    return processed


def _fill_missing_details(session, batch_size: int, limit: Optional[int]) -> int:
    """Populate missing detail rows for events that already exist."""
    filled = 0

    def fill_for_type(evt_type: str, detail_model, populator):
        nonlocal filled
        count_filled = 0
        q = (
            session.query(Event.id, RawEvent.raw_json)
            .join(RawEvent, Event.raw_event_id == RawEvent.id)
            .outerjoin(detail_model, detail_model.event_id == Event.id)
            .filter(RawEvent.type == evt_type, detail_model.event_id.is_(None))
            .order_by(Event.id)
        )
        fetched = 0
        while True:
            bq = q.limit(batch_size)
            batch = bq.all()
            if not batch:
                break
            for ev_id, raw_json in batch:
                ev = session.get(Event, ev_id)
                populator(session, ev, raw_json)
                count_filled += 1
                filled += 1
                if limit is not None and filled >= limit:
                    break
            session.commit()
            if limit is not None and filled >= limit:
                break
            last_id = batch[-1][0]
            q = q.filter(Event.id > last_id)
        if count_filled:
            logger.info("Filled %d missing rows for %s", count_filled, evt_type)

    fill_for_type("Pass", PassEvent, _populate_pass)
    fill_for_type("Dribble", DribbleEvent, _populate_dribble)
    fill_for_type("Carry", CarryEvent, _populate_carry)
    fill_for_type("Clearance", ClearanceEvent, _populate_clearance)
    fill_for_type("Duel", DuelEvent, _populate_duel)
    fill_for_type("Block", BlockEvent, _populate_block)
    fill_for_type("Interception", InterceptionEvent, _populate_interception)
    fill_for_type("Pressure", PressureEvent, _populate_pressure)
    fill_for_type("Ball Receipt", BallReceiptEvent, _populate_ball_receipt)
    fill_for_type("Ball Receipt*", BallReceiptEvent, _populate_ball_receipt)
    return filled


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for normalizing events.

    Accepts an optional ``argv`` list so that callers (like tests) can
    invoke this function without `argparse` trying to parse pytest's
    command line arguments.
    """

    parser = argparse.ArgumentParser(description="Normalize raw events to normalized tables")
    parser.add_argument("--only-missing", dest="only_missing", action="store_true", default=True, help="Process only raw events missing in events table")
    parser.add_argument("--from-id", type=int, default=None, help="Start from raw_event id >= N")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N raw events")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--fill-missing-detail", action="store_true", help="Also fill missing specialized detail rows for already-normalized events")
    parser.add_argument(
        "--fill-missing-pass-end-location",
        action="store_true",
        help="Backfill end_x/end_y for existing pass detail rows",
    )
    args = parser.parse_args(argv)

    with session_scope() as session:
        total_raw = session.query(RawEvent).count()
        total_events = session.query(Event).count()
        logger.info("Totals before: raw_events=%d, events=%d", total_raw, total_events)

        processed = 0

        if args.only_missing:
            processed = _normalize_missing_events(session, args.batch_size, args.from_id, args.limit)
        else:
            # Fallback: iterate all raw events but _ensure_event makes it idempotent
            logger.info("Scanning all raw events (idempotent mode)...")
            q = session.query(RawEvent).order_by(RawEvent.id)
            if args.from_id is not None:
                q = q.filter(RawEvent.id >= args.from_id)
            fetched = 0
            while True:
                take = args.batch_size if args.limit is None else max(0, min(args.batch_size, args.limit - fetched))
                if take == 0:
                    break
                batch = q.limit(take).all()
                if not batch:
                    break
                for raw_ev in batch:
                    ev = _ensure_event(session, raw_ev)
                    if raw_ev.type in POPULATORS and raw_ev.type != "Shot":
                        POPULATORS[raw_ev.type](session, ev, raw_ev.raw_json)
                    processed += 1
                session.commit()
                fetched += len(batch)
                if args.limit is not None and fetched >= args.limit:
                    break
                last_id = batch[-1].id
                q = q.filter(RawEvent.id > last_id)

        logger.info("Normalized %d new base events", processed)

        filled = 0
        if args.fill_missing_pass_end_location:
            filled += _fill_missing_pass_end_locations(session, args.batch_size, args.limit)

        if args.fill_missing_detail:
            filled += _fill_missing_details(session, args.batch_size, args.limit)
            logger.info("Filled %d missing detail rows", filled)

        after_events = session.query(Event).count()
        logger.info("Totals after: events=%d (+%d)", after_events, after_events - total_events)


if __name__ == "__main__":
    main()
