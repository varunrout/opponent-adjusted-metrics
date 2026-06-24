"""Generate a database ingestion status report.

This script gives a lightweight, repeatable way to verify what has been loaded
into the database after fetch, ingestion, and normalisation steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import func

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.db.models import (  # noqa: E402
    AggregatesPlayer,
    AggregatesTeam,
    BallReceiptEvent,
    BlockEvent,
    CarryEvent,
    ClearanceEvent,
    Competition,
    DribbleEvent,
    DuelEvent,
    Event,
    EvaluationMetric,
    InterceptionEvent,
    Match,
    ModelRegistry,
    OpponentDefProfile,
    PassEvent,
    Player,
    Possession,
    PressureEvent,
    RawEvent,
    Shot,
    ShotFeature,
    ShotPrediction,
    Team,
)
from opponent_adjusted.db.session import session_scope  # noqa: E402
from opponent_adjusted.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

TABLES = {
    "competitions": Competition,
    "teams": Team,
    "players": Player,
    "matches": Match,
    "raw_events": RawEvent,
    "events": Event,
    "possessions": Possession,
    "passes": PassEvent,
    "dribbles": DribbleEvent,
    "carries": CarryEvent,
    "clearances": ClearanceEvent,
    "duels": DuelEvent,
    "blocks": BlockEvent,
    "interceptions": InterceptionEvent,
    "pressures": PressureEvent,
    "ball_receipts": BallReceiptEvent,
    "shots": Shot,
    "shot_features": ShotFeature,
    "opponent_def_profile": OpponentDefProfile,
    "model_registry": ModelRegistry,
    "shot_predictions": ShotPrediction,
    "aggregates_player": AggregatesPlayer,
    "aggregates_team": AggregatesTeam,
    "evaluation_metrics": EvaluationMetric,
}


def _count_rows(session: Any) -> dict[str, int]:
    return {name: int(session.query(model).count()) for name, model in TABLES.items()}


def _event_type_counts(session: Any, limit: int = 25) -> list[dict[str, Any]]:
    rows = (
        session.query(RawEvent.type, func.count(RawEvent.id))
        .group_by(RawEvent.type)
        .order_by(func.count(RawEvent.id).desc())
        .limit(limit)
        .all()
    )
    return [{"event_type": event_type, "count": int(count)} for event_type, count in rows]


def build_report() -> dict[str, Any]:
    with session_scope() as session:
        table_counts = _count_rows(session)
        event_type_counts = _event_type_counts(session)

    report = {
        "table_counts": table_counts,
        "event_type_counts": event_type_counts,
        "readiness": {
            "has_competitions": table_counts["competitions"] > 0,
            "has_matches": table_counts["matches"] > 0,
            "has_raw_events": table_counts["raw_events"] > 0,
            "has_normalized_events": table_counts["events"] > 0,
            "has_shots": table_counts["shots"] > 0,
            "has_shot_features": table_counts["shot_features"] > 0,
            "has_model_registry": table_counts["model_registry"] > 0,
            "has_predictions": table_counts["shot_predictions"] > 0,
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report database ingestion status")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "ingestion" / "db_status.json",
        help="Path to write JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Ingestion status report written to %s", args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
