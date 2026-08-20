"""BigQuery-backed implementation of the dashboard serving store."""

from __future__ import annotations

from google.cloud import bigquery  # type: ignore[import-untyped]

from opponent_adjusted.api.interfaces import CompetitionRecord, MatchRecord

PROJECT = "oam-varun-260819"
DATASET = "oam_core"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


class BigQueryServingStore:
    """Read-only ServingStore backed by the oam_core BigQuery dataset."""

    def list_competitions(self) -> list[CompetitionRecord]:
        client = _client()
        query = f"""
            SELECT
                competition_id,
                season_id,
                competition_name,
                competition_gender,
                country_name,
                season_name,
                match_updated,
                match_available,
                match_updated_360,
                match_available_360
            FROM `{PROJECT}.{DATASET}.competitions`
        """
        rows = client.query(query).result()
        return [
            CompetitionRecord(
                competition_id=row["competition_id"],
                season_id=row["season_id"],
                competition_name=row["competition_name"],
                competition_gender=row["competition_gender"],
                country_name=row["country_name"],
                season_name=row["season_name"],
                match_updated=row["match_updated"],
                match_available=row["match_available"],
                match_updated_360=row["match_updated_360"],
                match_available_360=row["match_available_360"],
            )
            for row in rows
        ]

    def list_matches(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[MatchRecord]:
        client = _client()
        conditions: list[str] = []
        parameters: list[bigquery.ScalarQueryParameter] = []

        if competition_id is not None:
            conditions.append("competition_id = @competition_id")
            parameters.append(
                bigquery.ScalarQueryParameter("competition_id", "INT64", competition_id)
            )
        if season_id is not None:
            conditions.append("season_id = @season_id")
            parameters.append(bigquery.ScalarQueryParameter("season_id", "INT64", season_id))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                match_id,
                competition_id,
                season_id,
                match_date,
                kick_off,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_score,
                away_score,
                competition_stage,
                stadium,
                referee,
                match_status,
                match_status_360,
                last_updated,
                last_updated_360
            FROM `{PROJECT}.{DATASET}.matches`
            {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [
            MatchRecord(
                match_id=row["match_id"],
                competition_id=row["competition_id"],
                season_id=row["season_id"],
                match_date=row["match_date"],
                kick_off=row["kick_off"],
                home_team_id=row["home_team_id"],
                home_team_name=row["home_team_name"],
                away_team_id=row["away_team_id"],
                away_team_name=row["away_team_name"],
                home_score=row["home_score"],
                away_score=row["away_score"],
                competition_stage=row["competition_stage"],
                stadium=row["stadium"],
                referee=row["referee"],
                match_status=row["match_status"],
                match_status_360=row["match_status_360"],
                last_updated=row["last_updated"],
                last_updated_360=row["last_updated_360"],
            )
            for row in rows
        ]
