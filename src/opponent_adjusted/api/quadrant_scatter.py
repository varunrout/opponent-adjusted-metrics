"""Player-season CxG/CxA quadrant-scatter read path -- Track B (Hard gate 2,
`docs/dashboard_design_spec_v2.md` section 9), "build-your-own quadrant scatter,
originally scoped for Analysis" -- confirmed still the right home: the component
inventory (section 10) lists it as an Analysis-tab chart, and Analysis is this
project's real-time research tool, matching "build-your-own" (a user-selected
metric pair, not a fixed pre-built chart).

Backed by `oam_serving.player_season_cxg_cxa_v1` (materialized by
`scripts/materialize_player_season_cxg_cxa_v1.py`, see
docs/analysis/quadrant_scatter_v1.md for the full design writeup): one row per
(player_id, competition_id, season_id, split), rolling up both CxG (from
`oam_ml.cxg_{track}_v3_predictions`) and CxA (from
`oam_serving.cxa_{track}_combined_v1`) to player-season grain, for both the
event and plus tracks of each.

**Test-split only (decision, matching CxG's and CxA's own established
precedent exactly -- see `cxg_coverage.py`'s `COVERAGE_SPLIT` and
`cxa_models.py`'s decision 4):** the materialized table carries all three
splits as a `split` column (never baking the filter into the table), but this
read path only ever queries `split='test'`.

**Null vs zero (same discipline as everywhere else in this project):** a
player-season with zero covered shots/passes for a given metric has NULL for
that metric's mean/total (there is no "average CxG" to report), while its own
`_n` count is a real 0 -- never conflated. A player-season absent from ALL four
metrics for a given split isn't written to the table at all (see the
materialization script's own key-union logic).
"""

from __future__ import annotations

import threading

from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from google.cloud import bigquery
from pydantic import BaseModel, ConfigDict

from opponent_adjusted.api.bigquery_store import CACHE_TTL_SECONDS, PROJECT, _client

SERVING_DATASET = "oam_serving"
TABLE = "player_season_cxg_cxa_v1"

COVERAGE_SPLIT = "test"


class PlayerSeasonQuadrantRow(BaseModel):
    """One player-season's full CxG/CxA rollup, both tracks. Any `*_mean`/`*_total`
    field is `None` (never 0) when that track's `*_n` count is 0 for this player-
    season -- there being nothing to average is a different fact than the average
    being zero."""

    model_config = ConfigDict(from_attributes=True)

    player_id: int
    player_name: str | None
    team_id: int | None
    team_name: str | None
    competition_id: int
    season_id: int
    split: str

    cxg_event_n_shots: int
    cxg_event_mean: float | None
    cxg_event_total: float | None
    cxg_event_total_xg: float | None
    cxg_event_goals: int

    cxg_plus_n_shots: int
    cxg_plus_mean: float | None
    cxg_plus_total: float | None
    cxg_plus_total_xg: float | None
    cxg_plus_goals: int

    cxa_event_n_passes_created: int
    cxa_event_mean: float | None
    cxa_event_total: float | None

    cxa_plus_n_passes_created: int
    cxa_plus_mean: float | None
    cxa_plus_total: float | None


class QuadrantScatterResponse(BaseModel):
    """API response shape -- rows for whichever (competition_id, season_id) scope
    was requested, test split only."""

    model_config = ConfigDict(from_attributes=True)

    split: str
    rows: list[PlayerSeasonQuadrantRow]


class CxaRollup(BaseModel):
    """One track's CxA rollup for a player or team, aggregated (by summing each
    matched player-season row's own `_total`/`_n`, never re-averaging an already-
    averaged mean) across whatever (competition_id, season_id) scope was
    requested. `n=0` means `mean`/`total` are `None` -- there is nothing to
    average over zero chance-creating passes, never reported as `0.0`."""

    n: int
    mean: float | None
    total: float | None


class PlayerCxaResponse(BaseModel):
    """CxA rollup for one player, both tracks, guest-accessible (see
    docs/analysis/cxa_players_teams_v1.md: the underlying data is already public
    elsewhere -- Models pages, per-shot coverage -- so this is a convenience
    read, not a new access boundary)."""

    model_config = ConfigDict(from_attributes=True)

    player_id: int
    event: CxaRollup
    plus: CxaRollup


class TeamCxaResponse(BaseModel):
    """CxA rollup for one team, both tracks -- the sum of every one of that
    team's players' own player-season `_total`/`_n` for the requested scope.
    Algebraically identical to computing the mean directly over the team's whole
    population of chance-creating passes (sum-of-totals / sum-of-counts), not an
    average-of-averages."""

    model_config = ConfigDict(from_attributes=True)

    team_id: int
    event: CxaRollup
    plus: CxaRollup


def _rollup(rows: list[PlayerSeasonQuadrantRow], track: str) -> CxaRollup:
    n_field = f"cxa_{track}_n_passes_created"
    total_field = f"cxa_{track}_total"
    n = sum(getattr(r, n_field) for r in rows)
    if n == 0:
        return CxaRollup(n=0, mean=None, total=None)
    # Every matched row's own total is non-None exactly when its own n > 0 (the
    # materialization script's own invariant), so this sum only ever includes
    # real contributions -- no row silently contributes a phantom 0.
    total = sum(getattr(r, total_field) for r in rows if getattr(r, total_field) is not None)
    return CxaRollup(n=n, mean=total / n, total=total)


_scatter_cache: TTLCache = TTLCache(maxsize=32, ttl=CACHE_TTL_SECONDS)
_scatter_lock = threading.Lock()


def _scatter_cache_key(
    self, *, competition_id: int | None = None, season_id: int | None = None  # noqa: ANN001
) -> tuple:
    return hashkey(competition_id, season_id)


class BigQueryQuadrantScatterStore:
    """QuadrantScatterStore backed by `oam_serving.player_season_cxg_cxa_v1`.

    Cached per (competition_id, season_id) filter combination, same reasoning as
    every other Explore/Analysis store in this project: the underlying table is
    small (one row per player-season, test split only) and identical for every
    caller requesting the same scope within the TTL window."""

    @cached(cache=_scatter_cache, key=_scatter_cache_key, lock=_scatter_lock)
    def list_player_season_rows(
        self, *, competition_id: int | None = None, season_id: int | None = None
    ) -> list[PlayerSeasonQuadrantRow]:
        client = _client()
        conditions = ["split = @split"]
        parameters: list[bigquery.ScalarQueryParameter] = [
            bigquery.ScalarQueryParameter("split", "STRING", COVERAGE_SPLIT)
        ]
        if competition_id is not None:
            conditions.append("competition_id = @competition_id")
            parameters.append(bigquery.ScalarQueryParameter("competition_id", "INT64", competition_id))
        if season_id is not None:
            conditions.append("season_id = @season_id")
            parameters.append(bigquery.ScalarQueryParameter("season_id", "INT64", season_id))
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                player_id, player_name, team_id, team_name, competition_id, season_id, split,
                cxg_event_n_shots, cxg_event_mean, cxg_event_total, cxg_event_total_xg, cxg_event_goals,
                cxg_plus_n_shots, cxg_plus_mean, cxg_plus_total, cxg_plus_total_xg, cxg_plus_goals,
                cxa_event_n_passes_created, cxa_event_mean, cxa_event_total,
                cxa_plus_n_passes_created, cxa_plus_mean, cxa_plus_total
            FROM `{PROJECT}.{SERVING_DATASET}.{TABLE}`
            WHERE {where_clause}
            ORDER BY player_id
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [PlayerSeasonQuadrantRow.model_validate(dict(row.items())) for row in rows]

    def get_player_cxa(
        self, player_id: int, *, competition_id: int | None = None, season_id: int | None = None
    ) -> PlayerCxaResponse:
        """Filters the same cached table read `list_player_season_rows` already
        does (no second BigQuery query path) down to this one player's rows --
        the whole test-split table is small (822 rows) and already cached, so
        filtering in Python here is cheaper than adding a second query shape."""
        rows = self.list_player_season_rows(competition_id=competition_id, season_id=season_id)
        matched = [r for r in rows if r.player_id == player_id]
        return PlayerCxaResponse(
            player_id=player_id, event=_rollup(matched, "event"), plus=_rollup(matched, "plus")
        )

    def get_team_cxa(
        self, team_id: int, *, competition_id: int | None = None, season_id: int | None = None
    ) -> TeamCxaResponse:
        """Team-grain rollup over the SAME player-grain table -- confirmed live
        (docs/analysis/cxa_players_teams_v1.md section on team grain) that no
        (player_id, competition_id, season_id, split) key in this table ever
        carries more than one team_id, so summing every matched player-season
        row's own totals/counts per team_id cannot double-count a mid-season
        transfer. No `oam_core` re-join, no new materialization."""
        rows = self.list_player_season_rows(competition_id=competition_id, season_id=season_id)
        matched = [r for r in rows if r.team_id == team_id]
        return TeamCxaResponse(team_id=team_id, event=_rollup(matched, "event"), plus=_rollup(matched, "plus"))
