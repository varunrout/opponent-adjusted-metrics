"""Publish StatsBomb Silver v1 Parquet to BigQuery oam_core with idempotent logic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from google.api_core.exceptions import NotFound
from google.cloud import bigquery  # type: ignore[import-untyped]
from google.cloud import storage  # type: ignore[import-untyped]

from opponent_adjusted.pipelines.silver.contracts import (
    CONTRACTS,
    table_bq_schema,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishConfig:
    project_id: str
    dataset: str
    location: str
    bucket_name: str
    output_prefix: str
    data_version: str
    silver_schema_version: str


@dataclass(frozen=True)
class PublishResult:
    table_row_counts: dict[str, int]
    load_actions: dict[str, str]
    join_checks: dict[str, int]


def _count_rows_for_version(
    client: bigquery.Client,
    table_ref: str,
    data_version: str,
    silver_schema_version: str,
) -> int:
    query = f"""
        SELECT COUNT(1) AS c
        FROM `{table_ref}`
        WHERE data_version = @data_version
          AND silver_schema_version = @silver_schema_version
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("data_version", "STRING", data_version),
                bigquery.ScalarQueryParameter(
                    "silver_schema_version", "STRING", silver_schema_version
                ),
            ]
        ),
    )
    return int(next(iter(job.result()))["c"])


def _table_uris(storage_client: storage.Client, bucket_name: str, table_prefix: str) -> list[str]:
    blobs = list(storage_client.list_blobs(bucket_name, prefix=table_prefix))
    return [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".parquet")]


def publish_oam_core(config: PublishConfig) -> PublishResult:
    bq = bigquery.Client(project=config.project_id)
    st = storage.Client(project=config.project_id)

    dataset_ref = f"{config.project_id}.{config.dataset}"
    dataset = bq.get_dataset(dataset_ref)
    if dataset.location != config.location:
        raise RuntimeError(
            f"Dataset location mismatch: expected {config.location}, got {dataset.location}"
        )

    manifest_blob = st.bucket(config.bucket_name).blob(f"{config.output_prefix}/manifest.json")
    manifest = json.loads(manifest_blob.download_as_bytes().decode("utf-8"))

    actions: dict[str, str] = {}
    table_counts: dict[str, int] = {}

    for table_name in CONTRACTS:
        expected = int(manifest["tables"][table_name]["row_count"])
        table_ref = f"{dataset_ref}.{table_name}"
        uris = _table_uris(st, config.bucket_name, f"{config.output_prefix}/{table_name}/")

        try:
            existing = _count_rows_for_version(
                bq,
                table_ref,
                config.data_version,
                config.silver_schema_version,
            )
        except NotFound:
            existing = 0

        if existing == expected and expected > 0:
            actions[table_name] = "skipped_existing"
            table_counts[table_name] = existing
            continue
        if existing > 0 and existing != expected:
            raise RuntimeError(
                f"BigQuery immutable mismatch for {table_name}: existing={existing} expected={expected}"
            )
        if expected == 0:
            actions[table_name] = "skipped_empty"
            table_counts[table_name] = 0
            continue
        if not uris:
            raise RuntimeError(f"No parquet objects found for table {table_name}")

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            schema=table_bq_schema(table_name),
        )
        load_job = bq.load_table_from_uri(
            uris,
            table_ref,
            location=config.location,
            job_config=job_config,
        )
        load_job.result()

        after = _count_rows_for_version(
            bq,
            table_ref,
            config.data_version,
            config.silver_schema_version,
        )
        if after != expected:
            raise RuntimeError(
                f"Row count mismatch after load for {table_name}: after={after} expected={expected}"
            )

        actions[table_name] = "loaded"
        table_counts[table_name] = after

    joins: dict[str, int] = {}
    checks = {
        "shots_join_events_matches": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.shots` s
            JOIN `{dataset_ref}.events` e ON s.event_id = e.event_id
            JOIN `{dataset_ref}.matches` m ON e.match_id = m.match_id
            WHERE s.data_version = '{config.data_version}'
              AND s.silver_schema_version = '{config.silver_schema_version}'
        """,
        "three_sixty_frames_join_events_matches": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.three_sixty_frames` f
            JOIN `{dataset_ref}.events` e ON f.event_uuid = e.event_id AND f.match_id = e.match_id
            JOIN `{dataset_ref}.matches` m ON e.match_id = m.match_id
            WHERE f.data_version = '{config.data_version}'
              AND f.silver_schema_version = '{config.silver_schema_version}'
        """,
        "three_sixty_players_join_frames": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.three_sixty_players` p
            JOIN `{dataset_ref}.three_sixty_frames` f
              ON p.match_id = f.match_id AND p.event_uuid = f.event_uuid
            WHERE p.data_version = '{config.data_version}'
              AND p.silver_schema_version = '{config.silver_schema_version}'
        """,
        "possessions_join_events": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.possessions` p
            JOIN `{dataset_ref}.events` e ON p.match_id = e.match_id
            WHERE p.data_version = '{config.data_version}'
              AND p.silver_schema_version = '{config.silver_schema_version}'
        """,
        "passes_join_events": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.passes` p
            JOIN `{dataset_ref}.events` e ON p.event_id = e.event_id
            WHERE p.data_version = '{config.data_version}'
              AND p.silver_schema_version = '{config.silver_schema_version}'
        """,
    }

    for name, query in checks.items():
        rows = list(bq.query(query, location=config.location).result())
        joins[name] = int(rows[0]["c"]) if rows else 0

    return PublishResult(table_row_counts=table_counts, load_actions=actions, join_checks=joins)
