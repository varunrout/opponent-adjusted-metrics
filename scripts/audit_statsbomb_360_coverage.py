"""Generate empirical StatsBomb 360 coverage report from versioned raw GCS data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from google.cloud import storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.ingestion.three_sixty_audit import (  # noqa: E402
    TARGET_THREE_SIXTY_COMPETITIONS,
    build_three_sixty_coverage_report,
)


def _blob_payload_or_raise(blob: storage.Blob) -> list | dict:
    raw = blob.download_as_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, (list, dict)):
        raise ValueError(f"Unsupported JSON root type for {blob.name}: {type(payload).__name__}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit StatsBomb 360 coverage from GCS raw landing"
    )
    parser.add_argument("--gcs-bucket", required=True, help="Raw data bucket")
    parser.add_argument("--data-version", required=True, help="Pinned immutable data version SHA")
    parser.add_argument("--source-ref", required=True, help="Pinned source ref SHA")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "ingestion" / "three_sixty_coverage.json",
        help="Output coverage JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = storage.Client()
    bucket = client.bucket(args.gcs_bucket)
    prefix = f"raw/statsbomb/{args.data_version}"

    matches_by_competition: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for competition_id, season_id in sorted(TARGET_THREE_SIXTY_COMPETITIONS):
        matches_name = f"{prefix}/matches/{competition_id}/{season_id}.json"
        matches_blob = bucket.blob(matches_name)
        if not matches_blob.exists(client=client):
            raise FileNotFoundError(
                f"Required matches object missing: gs://{args.gcs_bucket}/{matches_name}"
            )
        matches_payload = _blob_payload_or_raise(matches_blob)
        if not isinstance(matches_payload, list):
            raise ValueError(f"Matches payload must be list: gs://{args.gcs_bucket}/{matches_name}")
        matches_by_competition[(competition_id, season_id)] = [
            item for item in matches_payload if isinstance(item, dict)
        ]

    def load_events(match_id: int) -> list | dict:
        name = f"{prefix}/events/{match_id}.json"
        blob = bucket.blob(name)
        if not blob.exists(client=client):
            raise FileNotFoundError(
                f"Required events object missing: gs://{args.gcs_bucket}/{name}"
            )
        return _blob_payload_or_raise(blob)

    def load_three_sixty(match_id: int) -> list | dict:
        name = f"{prefix}/three-sixty/{match_id}.json"
        blob = bucket.blob(name)
        if not blob.exists(client=client):
            raise FileNotFoundError(name)
        return _blob_payload_or_raise(blob)

    report = build_three_sixty_coverage_report(
        source_ref=args.source_ref,
        matches_by_competition=matches_by_competition,
        load_events=load_events,
        load_three_sixty=load_three_sixty,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Coverage report written: {args.output}")


if __name__ == "__main__":
    main()
