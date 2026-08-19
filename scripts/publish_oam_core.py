"""Publish StatsBomb Silver v1 outputs to BigQuery oam_core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.pipelines.silver.contracts import SILVER_SCHEMA_VERSION  # noqa: E402
from opponent_adjusted.pipelines.silver.publish_core import (  # noqa: E402
    PublishConfig,
    publish_oam_core,
)

DEFAULT_SOURCE_REF = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Silver v1 to BigQuery oam_core")
    parser.add_argument("--project-id", default="oam-varun-260819")
    parser.add_argument("--dataset", default="oam_core")
    parser.add_argument("--location", default="europe-west2")
    parser.add_argument("--bucket", default="oam-varun-260819-data")
    parser.add_argument("--data-version", default=DEFAULT_SOURCE_REF)
    parser.add_argument(
        "--output-prefix",
        default=f"staged/statsbomb/{DEFAULT_SOURCE_REF}/{SILVER_SCHEMA_VERSION}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = publish_oam_core(
        PublishConfig(
            project_id=args.project_id,
            dataset=args.dataset,
            location=args.location,
            bucket_name=args.bucket,
            output_prefix=args.output_prefix,
            data_version=args.data_version,
            silver_schema_version=SILVER_SCHEMA_VERSION,
        )
    )
    print(
        json.dumps(
            {
                "table_row_counts": result.table_row_counts,
                "load_actions": result.load_actions,
                "join_checks": result.join_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
