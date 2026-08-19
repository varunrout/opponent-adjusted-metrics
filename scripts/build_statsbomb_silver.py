"""Build and publish StatsBomb Silver v1 Parquet outputs to GCS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.pipelines.silver.builder import (  # noqa: E402
    SilverBuildConfig,
    build_statsbomb_silver,
)
from opponent_adjusted.pipelines.silver.contracts import SILVER_SCHEMA_VERSION  # noqa: E402


DEFAULT_SOURCE_REF = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build StatsBomb Silver v1")
    parser.add_argument("--bucket", default="oam-varun-260819-data")
    parser.add_argument("--artifacts-bucket", default="oam-varun-260819-artifacts")
    parser.add_argument("--project-id", default="oam-varun-260819")
    parser.add_argument("--data-version", default=DEFAULT_SOURCE_REF)
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    parser.add_argument("--subset-config", default="configs/statsbomb_subset.json")
    parser.add_argument(
        "--output-prefix",
        default=(f"staged/statsbomb/{DEFAULT_SOURCE_REF}/{SILVER_SCHEMA_VERSION}"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["GIT_COMMIT_SHA"] = _git_sha()

    result = build_statsbomb_silver(
        SilverBuildConfig(
            bucket_name=args.bucket,
            artifacts_bucket=args.artifacts_bucket,
            data_version=args.data_version,
            source_ref=args.source_ref,
            silver_schema_version=SILVER_SCHEMA_VERSION,
            competitions_config_path=PROJECT_ROOT / args.subset_config,
            output_prefix=args.output_prefix,
            project_id=args.project_id,
        )
    )

    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "output_prefix": result.output_prefix,
                "manifest_uri": result.manifest_uri,
                "artifacts_manifest_uri": result.artifacts_manifest_uri,
                "row_counts": result.row_counts,
                "object_counts": result.object_counts,
                "bytes_by_table": result.bytes_by_table,
                "bronze_summary": result.bronze_summary,
                "qa_summary": result.qa_summary,
                "warnings_count": len(result.warnings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
