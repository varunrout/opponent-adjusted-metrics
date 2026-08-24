"""Version-scoped repair for the broken repeated-column BigQuery publication."""

from __future__ import annotations

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
DATASET = "oam_core"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

TABLES = (
    "competitions", "matches", "teams", "players", "starting_xi_players",
    "events", "shots", "passes", "carries", "dribbles", "pressures",
    "ball_receipts", "disciplinary_events", "substitutions", "possessions",
    "three_sixty_frames", "three_sixty_players", "shot_freeze_frame_players",
)


def main() -> None:
    client = bigquery.Client(project=PROJECT, location="europe-west2")
    for table in TABLES:
        query = f"""
        DELETE FROM `{PROJECT}.{DATASET}.{table}`
        WHERE data_version = @data_version
          AND silver_schema_version = @schema_version
        """
        job = client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
                    bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
                ]
            ),
            location="europe-west2",
        )
        job.result()
        print(f"deleted {table}: {job.num_dml_affected_rows}", flush=True)


if __name__ == "__main__":
    main()
