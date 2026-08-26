"""CxG v3 test-set coverage read path for the Explore zone (Matches/Players/Teams).

Per docs/dashboard_design_spec_v2.md §4a: CxG values may appear alongside
StatsBomb xG on Matches/Players/Teams, but only where the underlying shot
falls inside oam_ml's v3 test-set coverage — a fixed train/test split, not
full oam_core coverage. This module is the read path for that: given a
set of event_ids (from an already-fetched ShotResponse list) and a track,
return CxG values for whichever of those event_ids are actually covered.

Join key, verified against live BigQuery rather than assumed: event_id
(STRING/UUID) is a clean 1:1 join between oam_ml's *_v3_predictions
tables and oam_core.shots.event_id — confirmed no duplicate event_ids on
the oam_ml side. Deliberately does NOT join against oam_core.shots at
all here — the caller already has the shot list (and its event_ids) from
the existing ServingStore endpoints, so this only needs to look up CxG
values by event_id, which avoids re-touching oam_core (and therefore
avoids needing its own silver_schema_version lineage filtering — see
bigquery_store.py's SILVER_SCHEMA_VERSION fix for that separate issue).

Kept as its own small module rather than bigquery_store.py (whose
existing nine oam_core methods and TTL caches this deliberately doesn't
touch) or the three-file analysis_* pattern (that split exists because
the oam_analysis feature has 10+ record types across 8 endpoints; this
is one endpoint with a dict[str, float] shape, and doesn't warrant the
same ceremony).
"""

from __future__ import annotations

import threading
from typing import Protocol

from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from pydantic import BaseModel, ConfigDict

from opponent_adjusted.api.bigquery_store import CACHE_TTL_SECONDS, PROJECT, _client

ML_DATASET = "oam_ml"
ANALYSIS_DATASET = "oam_analysis"

# Real BigQuery track values, per §4a/§11 — NOT "event_wide".
TRACK_TABLE_PREFIXES = {
    "cxg_event": "cxg_event_v3",
    "cxg_plus": "cxg_plus_v3",
}

# The one split this whole module treats as "covered". Defined once and
# reused by both the per-shot coverage lookup below and the per-match scope
# lookup (§9.3 GET /v1/cxg/matches) — the two must never disagree about
# what "covered" means, per content_spec_v3.md §9.3.
COVERAGE_SPLIT = "test"

MATCH_SPLITS_TABLE = "cxg_match_splits_v1"

# Genuinely guest-visible (Matches/Players/Teams are Explore-zone, per
# design_spec_v2.md §5/§7), unlike the admin-only Analysis tab — so this
# gets the same TTL caching discipline as Hard gate 4's oam_core fix, not
# the "caching is optional here" treatment the Analysis endpoints get.
#
# Cached per-track, not per-request event_id set: each track's full test
# split is small (~2400 rows) and the exact same for every visitor, so
# fetching it once per TTL window and doing the event_id lookup in memory
# gets a far better cache-hit rate than trying to cache per distinct
# (track, event_ids) combination, which would almost never repeat across
# different matches/players/teams.
_coverage_cache: TTLCache = TTLCache(maxsize=8, ttl=CACHE_TTL_SECONDS)
_coverage_lock = threading.Lock()


def _track_cache_key(self, track: str) -> tuple:  # noqa: ANN001
    return hashkey(track)


class CxgCoverageStore(Protocol):
    """Read-only contract for CxG v3 test-set coverage lookups."""

    def get_cxg_for_events(self, event_ids: list[str], *, track: str) -> dict[str, float]:
        """Return {event_id: cxg_value} for whichever of the given event_ids
        have v3 test-set coverage on the given track. event_ids with no
        coverage are simply absent from the result — never a placeholder
        value, per §4a ("no CxG placeholder, dash, zero, or N/A")."""


class BigQueryCxgCoverageStore:
    """CxgCoverageStore backed by oam_ml's v3 prediction tables."""

    @cached(cache=_coverage_cache, key=_track_cache_key, lock=_coverage_lock)
    def _get_track_coverage(self, track: str) -> dict[str, float]:
        if track not in TRACK_TABLE_PREFIXES:
            raise ValueError(f"Unknown track: {track!r}")
        prefix = TRACK_TABLE_PREFIXES[track]
        client = _client()
        query = f"""
            SELECT event_id, v3_predicted_prob
            FROM `{PROJECT}.{ML_DATASET}.{prefix}_predictions`
            WHERE split = '{COVERAGE_SPLIT}'
        """
        rows = client.query(query).result()
        return {row["event_id"]: row["v3_predicted_prob"] for row in rows}

    def get_cxg_for_events(self, event_ids: list[str], *, track: str) -> dict[str, float]:
        coverage = self._get_track_coverage(track)
        return {event_id: coverage[event_id] for event_id in event_ids if event_id in coverage}


class CxgCoverageResponse(BaseModel):
    """API response shape for a CxG coverage lookup."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    values: dict[str, float]


# --- Per-match scope (§9.3 GET /v1/cxg/matches) -----------------------------
#
# Answers "which matches carry CxG predictions at all" — a question the
# *_predictions tables above can't answer themselves, since they key on
# event_id only, with no match_id. oam_analysis.cxg_match_splits_v1 is a
# small (610-row, one-per-match) table purpose-built for this.

_match_scope_cache: TTLCache = TTLCache(maxsize=8, ttl=CACHE_TTL_SECONDS)
_match_scope_lock = threading.Lock()


class CxgMatchScopeRow(BaseModel):
    """One row of oam_analysis.cxg_match_splits_v1, filtered to COVERAGE_SPLIT."""

    model_config = ConfigDict(from_attributes=True)

    match_id: int
    split: str
    has_360_match: bool
    event_shot_count: int
    plus_shot_count: int
    event_goal_count: int
    plus_goal_count: int


class CxgMatchScopeStore(Protocol):
    """Read-only contract for "which matches have CxG coverage" lookups."""

    def list_covered_matches(self, *, track: str) -> list[CxgMatchScopeRow]:
        """Return one row per match with test-split CxG coverage on the given
        track. For track="cxg_plus" this is further filtered to matches with
        360 data (has_360_match = TRUE) — CxG+ has no prediction otherwise."""


class BigQueryCxgMatchScopeStore:
    """CxgMatchScopeStore backed by oam_analysis.cxg_match_splits_v1."""

    @cached(cache=_match_scope_cache, key=_track_cache_key, lock=_match_scope_lock)
    def _get_track_matches(self, track: str) -> list[CxgMatchScopeRow]:
        if track not in TRACK_TABLE_PREFIXES:
            raise ValueError(f"Unknown track: {track!r}")
        client = _client()
        # Only cxg_plus needs the 360 filter — cxg_event has no such
        # constraint (8 features, no tracking data required).
        plus_filter = "AND has_360_match = TRUE" if track == "cxg_plus" else ""
        query = f"""
            SELECT match_id, split, has_360_match,
                   event_shot_count, plus_shot_count,
                   event_goal_count, plus_goal_count
            FROM `{PROJECT}.{ANALYSIS_DATASET}.{MATCH_SPLITS_TABLE}`
            WHERE split = '{COVERAGE_SPLIT}'
            {plus_filter}
        """
        rows = client.query(query).result()
        return [
            CxgMatchScopeRow(
                match_id=row["match_id"],
                split=row["split"],
                has_360_match=row["has_360_match"],
                event_shot_count=row["event_shot_count"],
                plus_shot_count=row["plus_shot_count"],
                event_goal_count=row["event_goal_count"],
                plus_goal_count=row["plus_goal_count"],
            )
            for row in rows
        ]

    def list_covered_matches(self, *, track: str) -> list[CxgMatchScopeRow]:
        return self._get_track_matches(track)
