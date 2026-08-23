"""BigQuery-backed implementation of the dashboard serving store."""

from __future__ import annotations

import threading

from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from google.cloud import bigquery  # type: ignore[import-untyped]

from opponent_adjusted.api.interfaces import (
    CompetitionRecord,
    LineupPlayerRecord,
    MatchRecord,
    PlayerSeasonRecord,
    ShotRecord,
    TeamSeasonRecord,
)

PROJECT = "oam-varun-260819"
DATASET = "oam_core"

_client_instance: bigquery.Client | None = None

# Per docs/dashboard_design_spec_v2.md §5 ("match/player/team aggregates
# don't need to be query-fresh to the second") and Hard gate 4: the Explore
# zone is guest-visible and genuinely dynamic, so repeat visits to the same
# competition/season previously re-billed BigQuery on every request — the
# real cost incident this session (34 unfiltered/repeat oam_core queries,
# 6.1GB, ~£2.5) came from exactly this pattern.
#
# 300s (5 minutes): long enough that a visitor re-browsing the same filter
# within a session, or two visitors hitting the same competition/season,
# gets served from cache instead of re-billing BigQuery; short enough that
# newly-published match/shot data shows up within one refresh cycle rather
# than requiring a redeploy to bust a stale cache. oam_core's own data is
# append-only-ish silver output that doesn't change second-to-second, so
# there's no real staleness cost to this window.
CACHE_TTL_SECONDS = 300
# Bounds per-method memory use; not a hard requirement from the design doc,
# just enough headroom for realistic distinct filter combinations (a handful
# of competitions x seasons x teams) within one TTL window without unbounded
# growth if this ever got hit by scripted/adversarial traffic.
CACHE_MAXSIZE = 256

# cachetools.TTLCache is not thread-safe on its own, and FastAPI serves
# concurrent requests (a sync route handler runs in a threadpool) — a single
# shared lock, passed to every `cached(..., lock=_cache_lock)` below, is
# cachetools' own documented pattern for making `cached` safe under
# concurrent access. The lock only guards the cache dict's own get/set, not
# the BigQuery call itself (see cachetools' implementation) — a burst of
# concurrent first-time requests for the same uncached key can still each
# reach BigQuery once before the cache populates, which is an accepted,
# standard tradeoff of this pattern, not a bug.
_cache_lock = threading.Lock()

_competitions_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_matches_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_match_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_lineups_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_shots_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_player_seasons_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_player_shots_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_team_seasons_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
_team_shots_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)


def _ignore_self_key(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    """Cache key that excludes `self`.

    BigQueryServingStore is stateless and dependencies.get_store() builds a
    fresh instance per request, so keying on instance identity (cachetools'
    default) would mean every request misses — the cache would never hit
    across requests at all. Every argument that changes the underlying
    query (competition_id, season_id, team_id, match_id, player_id) still
    flows into the key via *args/**kwargs.
    """
    return hashkey(*args, **kwargs)


def _client() -> bigquery.Client:
    """Return a shared, lazily-constructed BigQuery client (module-level singleton).

    Mirrors the lazy-init pattern in dependencies.py's _init_firebase_admin.
    Building a fresh bigquery.Client() on every call is wasteful once a
    single request can fan out into several BigQuery-backed calls in quick
    succession (e.g. the Analysis page firing six near-simultaneous
    requests, each of which previously built its own client) —
    bigquery_analysis_store.py imports this same function, so it benefits
    automatically.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = bigquery.Client(project=PROJECT)
    return _client_instance


class BigQueryServingStore:
    """Read-only ServingStore backed by the oam_core BigQuery dataset."""

    @cached(cache=_competitions_cache, key=_ignore_self_key, lock=_cache_lock)
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

    @cached(cache=_matches_cache, key=_ignore_self_key, lock=_cache_lock)
    def list_matches(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
        team_id: int | None = None,
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
        if team_id is not None:
            conditions.append("(home_team_id = @team_id OR away_team_id = @team_id)")
            parameters.append(bigquery.ScalarQueryParameter("team_id", "INT64", team_id))

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

    @cached(cache=_match_cache, key=_ignore_self_key, lock=_cache_lock)
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

    @cached(cache=_lineups_cache, key=_ignore_self_key, lock=_cache_lock)
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

    @cached(cache=_shots_cache, key=_ignore_self_key, lock=_cache_lock)
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

    @cached(cache=_player_seasons_cache, key=_ignore_self_key, lock=_cache_lock)
    def list_player_seasons(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[PlayerSeasonRecord]:
        client = _client()
        conditions: list[str] = []
        parameters: list[bigquery.ScalarQueryParameter] = []

        if competition_id is not None:
            conditions.append("s.competition_id = @competition_id")
            parameters.append(
                bigquery.ScalarQueryParameter("competition_id", "INT64", competition_id)
            )
        if season_id is not None:
            conditions.append("s.season_id = @season_id")
            parameters.append(bigquery.ScalarQueryParameter("season_id", "INT64", season_id))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                s.player_id AS player_id,
                ANY_VALUE(e.player_name) AS player_name,
                ANY_VALUE(s.team_id) AS team_id,
                ANY_VALUE(e.team_name) AS team_name,
                COUNT(*) AS shots,
                COUNTIF(s.outcome_name = 'Goal') AS goals,
                SUM(s.statsbomb_xg) AS total_xg
            FROM `{PROJECT}.{DATASET}.shots` s
            JOIN `{PROJECT}.{DATASET}.events` e
              ON s.event_id = e.event_id
             AND s.data_version = e.data_version
             AND s.silver_schema_version = e.silver_schema_version
            {where_clause}
            GROUP BY s.player_id
            ORDER BY total_xg DESC
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [
            PlayerSeasonRecord(
                player_id=row["player_id"],
                player_name=row["player_name"],
                team_id=row["team_id"],
                team_name=row["team_name"],
                shots=row["shots"],
                goals=row["goals"],
                total_xg=row["total_xg"],
            )
            for row in rows
        ]

    @cached(cache=_player_shots_cache, key=_ignore_self_key, lock=_cache_lock)
    def list_player_shots(
        self,
        player_id: int,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[ShotRecord]:
        client = _client()
        conditions: list[str] = ["s.player_id = @player_id"]
        parameters: list[bigquery.ScalarQueryParameter] = [
            bigquery.ScalarQueryParameter("player_id", "INT64", player_id)
        ]

        if competition_id is not None:
            conditions.append("s.competition_id = @competition_id")
            parameters.append(
                bigquery.ScalarQueryParameter("competition_id", "INT64", competition_id)
            )
        if season_id is not None:
            conditions.append("s.season_id = @season_id")
            parameters.append(bigquery.ScalarQueryParameter("season_id", "INT64", season_id))

        where_clause = f"WHERE {' AND '.join(conditions)}"
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
            {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
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

    @cached(cache=_team_seasons_cache, key=_ignore_self_key, lock=_cache_lock)
    def list_team_seasons(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[TeamSeasonRecord]:
        client = _client()
        conditions: list[str] = []
        parameters: list[bigquery.ScalarQueryParameter] = []

        if competition_id is not None:
            conditions.append("s.competition_id = @competition_id")
            parameters.append(
                bigquery.ScalarQueryParameter("competition_id", "INT64", competition_id)
            )
        if season_id is not None:
            conditions.append("s.season_id = @season_id")
            parameters.append(bigquery.ScalarQueryParameter("season_id", "INT64", season_id))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                s.team_id AS team_id,
                ANY_VALUE(e.team_name) AS team_name,
                COUNT(*) AS shots,
                COUNTIF(s.outcome_name = 'Goal') AS goals,
                SUM(s.statsbomb_xg) AS total_xg
            FROM `{PROJECT}.{DATASET}.shots` s
            JOIN `{PROJECT}.{DATASET}.events` e
              ON s.event_id = e.event_id
             AND s.data_version = e.data_version
             AND s.silver_schema_version = e.silver_schema_version
            {where_clause}
            GROUP BY s.team_id
            ORDER BY total_xg DESC
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [
            TeamSeasonRecord(
                team_id=row["team_id"],
                team_name=row["team_name"],
                shots=row["shots"],
                goals=row["goals"],
                total_xg=row["total_xg"],
            )
            for row in rows
        ]

    @cached(cache=_team_shots_cache, key=_ignore_self_key, lock=_cache_lock)
    def list_team_shots(
        self,
        team_id: int,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[ShotRecord]:
        client = _client()
        conditions: list[str] = ["s.team_id = @team_id"]
        parameters: list[bigquery.ScalarQueryParameter] = [
            bigquery.ScalarQueryParameter("team_id", "INT64", team_id)
        ]

        if competition_id is not None:
            conditions.append("s.competition_id = @competition_id")
            parameters.append(
                bigquery.ScalarQueryParameter("competition_id", "INT64", competition_id)
            )
        if season_id is not None:
            conditions.append("s.season_id = @season_id")
            parameters.append(bigquery.ScalarQueryParameter("season_id", "INT64", season_id))

        where_clause = f"WHERE {' AND '.join(conditions)}"
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
            {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
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
