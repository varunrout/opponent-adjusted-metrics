"""BigQuery-backed implementation of the dashboard serving store."""

from __future__ import annotations

from google.cloud import bigquery  # type: ignore[import-untyped]

from opponent_adjusted.api.interfaces import (
    CompetitionRecord,
    LineupPlayerRecord,
    MatchRecord,
    ShotRecord,
)

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

    def get_match(self, match_id: int) -> MatchRecord | None:
        client = _client()
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
            WHERE match_id = @match_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("match_id", "INT64", match_id)]
        )
        rows = list(client.query(query, job_config=job_config).result())
        if not rows:
            return None
        row = rows[0]
        return MatchRecord(
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

    def list_lineups(self, match_id: int) -> list[LineupPlayerRecord]:
        client = _client()
        query = f"""
            SELECT
                x.match_id AS match_id,
                x.team_id AS team_id,
                x.team_name AS team_name,
                x.formation AS formation,
                x.player_id AS player_id,
                x.player_name AS player_name,
                x.position_name AS position_name,
                x.jersey_number AS jersey_number
            FROM `{PROJECT}.{DATASET}.starting_xi_players` x
            JOIN `{PROJECT}.{DATASET}.events` e
              ON x.event_id = e.event_id AND x.match_id = e.match_id
             AND x.data_version = e.data_version
             AND x.silver_schema_version = e.silver_schema_version
             AND e.event_type_name = 'Starting XI'
            JOIN `{PROJECT}.{DATASET}.matches` m
              ON e.match_id = m.match_id
             AND e.data_version = m.data_version
             AND e.silver_schema_version = m.silver_schema_version
            WHERE x.match_id = @match_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("match_id", "INT64", match_id)]
        )
        rows = client.query(query, job_config=job_config).result()
        return [
            LineupPlayerRecord(
                match_id=row["match_id"],
                team_id=row["team_id"],
                team_name=row["team_name"],
                formation=row["formation"],
                player_id=row["player_id"],
                player_name=row["player_name"],
                position_name=row["position_name"],
                jersey_number=row["jersey_number"],
            )
            for row in rows
        ]

    def list_shots(self, match_id: int) -> list[ShotRecord]:
        client = _client()
        query = f"""
            SELECT
                s.event_id AS event_id,
                s.match_id AS match_id,
                s.team_id AS team_id,
                s.player_id AS player_id,
                e.player_name AS player_name,
                e.minute AS minute,
                e.period AS period,
                s.location_x AS location_x,
                s.location_y AS location_y,
                s.end_x AS end_x,
                s.end_y AS end_y,
                s.statsbomb_xg AS statsbomb_xg,
                s.outcome_name AS outcome_name,
                s.body_part_name AS body_part_name
            FROM `{PROJECT}.{DATASET}.shots` s
            JOIN `{PROJECT}.{DATASET}.events` e
              ON s.event_id = e.event_id
             AND s.data_version = e.data_version
             AND s.silver_schema_version = e.silver_schema_version
            JOIN `{PROJECT}.{DATASET}.matches` m
              ON e.match_id = m.match_id
             AND e.data_version = m.data_version
             AND e.silver_schema_version = m.silver_schema_version
            WHERE s.match_id = @match_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("match_id", "INT64", match_id)]
        )
        rows = client.query(query, job_config=job_config).result()
        return [
            ShotRecord(
                event_id=row["event_id"],
                match_id=row["match_id"],
                team_id=row["team_id"],
                player_id=row["player_id"],
                player_name=row["player_name"],
                minute=row["minute"],
                period=row["period"],
                location_x=row["location_x"],
                location_y=row["location_y"],
                end_x=row["end_x"],
                end_y=row["end_y"],
                statsbomb_xg=row["statsbomb_xg"],
                outcome_name=row["outcome_name"],
                body_part_name=row["body_part_name"],
                is_goal=row["outcome_name"] == "Goal",
            )
            for row in rows
        ]
