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


@dataclass(frozen=True)
class PublicationPlanEntry:
    table_name: str
    expected_row_count: int
    existing_row_count: int
    action: str
    uris: list[str]


def _version_query_config(data_version: str, silver_schema_version: str) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("data_version", "STRING", data_version),
            bigquery.ScalarQueryParameter("silver_schema_version", "STRING", silver_schema_version),
        ]
    )


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
        job_config=_version_query_config(data_version, silver_schema_version),
    )
    return int(next(iter(job.result()))["c"])


def _table_uris(storage_client: storage.Client, bucket_name: str, table_prefix: str) -> list[str]:
    blobs = list(storage_client.list_blobs(bucket_name, prefix=table_prefix))
    return [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".parquet")]


def _build_publication_plan(
    *,
    bq: bigquery.Client,
    st: storage.Client,
    config: PublishConfig,
    dataset_ref: str,
    manifest: dict,
) -> list[PublicationPlanEntry]:
    plan: list[PublicationPlanEntry] = []
    manifest_tables = manifest.get("tables")
    if not isinstance(manifest_tables, dict):
        raise RuntimeError("Manifest tables entry is missing or invalid")

    for table_name in CONTRACTS:
        try:
            table_manifest = manifest_tables[table_name]
            expected = int(table_manifest["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid manifest row count for {table_name}") from exc

        table_ref = f"{dataset_ref}.{table_name}"
        try:
            existing = _count_rows_for_version(
                bq, table_ref, config.data_version, config.silver_schema_version
            )
        except NotFound:
            existing = 0

        if existing > 0 and existing != expected:
            raise RuntimeError(
                f"BigQuery immutable mismatch for {table_name}: existing={existing} expected={expected}"
            )
        if expected == 0:
            plan.append(PublicationPlanEntry(table_name, expected, existing, "skipped_empty", []))
            continue
        if existing == expected:
            plan.append(
                PublicationPlanEntry(table_name, expected, existing, "skipped_existing", [])
            )
            continue

        uris = _table_uris(st, config.bucket_name, f"{config.output_prefix}/{table_name}/")
        if not uris:
            raise RuntimeError(f"No parquet objects found for table {table_name}")
        plan.append(PublicationPlanEntry(table_name, expected, existing, "loaded", uris))

    return plan


def _join_checks(dataset_ref: str) -> dict[str, str]:
    return {
        "shots_join_events_matches": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.shots` s
            JOIN `{dataset_ref}.events` e
              ON s.event_id = e.event_id
             AND s.data_version = e.data_version
             AND s.silver_schema_version = e.silver_schema_version
            JOIN `{dataset_ref}.matches` m
              ON e.match_id = m.match_id
             AND e.data_version = m.data_version
             AND e.silver_schema_version = m.silver_schema_version
            WHERE s.data_version = @data_version
              AND s.silver_schema_version = @silver_schema_version
        """,
        "three_sixty_frames_join_events_matches": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.three_sixty_frames` f
            JOIN `{dataset_ref}.events` e
              ON f.event_uuid = e.event_id AND f.match_id = e.match_id
             AND f.data_version = e.data_version
             AND f.silver_schema_version = e.silver_schema_version
            JOIN `{dataset_ref}.matches` m
              ON e.match_id = m.match_id
             AND e.data_version = m.data_version
             AND e.silver_schema_version = m.silver_schema_version
            WHERE f.data_version = @data_version
              AND f.silver_schema_version = @silver_schema_version
        """,
        "three_sixty_players_join_frames": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.three_sixty_players` p
            JOIN `{dataset_ref}.three_sixty_frames` f
              ON p.match_id = f.match_id AND p.event_uuid = f.event_uuid
             AND p.data_version = f.data_version
             AND p.silver_schema_version = f.silver_schema_version
            WHERE p.data_version = @data_version
              AND p.silver_schema_version = @silver_schema_version
        """,
        "possessions_join_events": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.possessions` p
            JOIN `{dataset_ref}.events` e
              ON p.match_id = e.match_id
             AND p.data_version = e.data_version
             AND p.silver_schema_version = e.silver_schema_version
            WHERE p.data_version = @data_version
              AND p.silver_schema_version = @silver_schema_version
        """,
        "passes_join_events": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.passes` p
            JOIN `{dataset_ref}.events` e
              ON p.event_id = e.event_id
             AND p.data_version = e.data_version
             AND p.silver_schema_version = e.silver_schema_version
            WHERE p.data_version = @data_version
              AND p.silver_schema_version = @silver_schema_version
        """,
        "starting_xi_players_join_events_matches": f"""
            SELECT COUNT(1) AS c
            FROM `{dataset_ref}.starting_xi_players` x
            JOIN `{dataset_ref}.events` e
              ON x.event_id = e.event_id AND x.match_id = e.match_id
             AND x.data_version = e.data_version
             AND x.silver_schema_version = e.silver_schema_version
            JOIN `{dataset_ref}.matches` m
              ON e.match_id = m.match_id
             AND e.data_version = m.data_version
             AND e.silver_schema_version = m.silver_schema_version
            WHERE x.data_version = @data_version
              AND x.silver_schema_version = @silver_schema_version
        """,
    }


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

    plan = _build_publication_plan(
        bq=bq, st=st, config=config, dataset_ref=dataset_ref, manifest=manifest
    )
    actions: dict[str, str] = {}
    table_counts: dict[str, int] = {}

    for entry in plan:
        table_name = entry.table_name
        if entry.action != "loaded":
            actions[table_name] = entry.action
            table_counts[table_name] = entry.existing_row_count
            continue

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            schema=table_bq_schema(table_name),
        )
        load_job = bq.load_table_from_uri(
            entry.uris,
            f"{dataset_ref}.{table_name}",
            location=config.location,
            job_config=job_config,
        )
        load_job.result()

        after = _count_rows_for_version(
            bq,
            f"{dataset_ref}.{table_name}",
            config.data_version,
            config.silver_schema_version,
        )
        if after != entry.expected_row_count:
            raise RuntimeError(
                f"Row count mismatch after load for {table_name}: "
                f"after={after} expected={entry.expected_row_count}"
            )

        actions[table_name] = "loaded"
        table_counts[table_name] = after

    for entry in plan:
        after = _count_rows_for_version(
            bq,
            f"{dataset_ref}.{entry.table_name}",
            config.data_version,
            config.silver_schema_version,
        )
        if after != entry.expected_row_count:
            raise RuntimeError(
                f"Final completeness mismatch for {entry.table_name}: "
                f"after={after} expected={entry.expected_row_count}"
            )
        table_counts[entry.table_name] = after

    joins: dict[str, int] = {}
    for name, query in _join_checks(dataset_ref).items():
        rows = list(
            bq.query(
                query,
                job_config=_version_query_config(config.data_version, config.silver_schema_version),
                location=config.location,
            ).result()
        )
        joins[name] = int(rows[0]["c"]) if rows else 0

    return PublishResult(table_row_counts=table_counts, load_actions=actions, join_checks=joins)
