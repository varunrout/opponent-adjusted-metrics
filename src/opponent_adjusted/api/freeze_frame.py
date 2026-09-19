"""Per-shot StatsBomb 360 freeze-frame read path (Explore zone).

oam_core.three_sixty_frames / three_sixty_players are finished, published
Silver data (see pipelines/silver/contracts.py, publish_core.py's
three_sixty_frames_join_events_matches check) — real raw 360 positions, not
the aggregate distances/roles in oam_analysis.cxg_analysis_opponent_adjusted_v1
(that table stays untouched by this module; see cxg_coverage.py's
OpponentContextStore for it).

Join key, confirmed live: three_sixty_frames.event_uuid == shots.event_id,
1:1, keyed together with match_id and silver_schema_version.

Orientation: a 360 frame's `teammate` flag is relative to the event/team the
frame is attached to (see features/cxg/three_sixty_frame.py's docstring and
its orient_players() escape hatch for frames attached to other event types).
Every frame this module queries is the one attached to the shot event itself
(event_uuid == shot.event_id), so the frame's own acting team is the
shooting team by construction: teammate=True already means "teammate of the
shooter", confirmed against live rows (the actor=True row's x/y matches the
shot's own location_x/location_y, and teammate=True on that row, in every
sample checked). No orient_players()/event-team-lookup needed here.

Unlike cxg_coverage.py's whole-track cache, this deliberately queries
per-shot (three_sixty_players is ~25M rows, not ~3,960) — targeted by
match_id + event_id on every call, no full-table fetch.
"""

from __future__ import annotations

from typing import Protocol

from google.cloud import bigquery  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from opponent_adjusted.api.bigquery_store import PROJECT, SILVER_SCHEMA_VERSION, _client

DATASET = "oam_core"


class FreezeFramePlayerResponse(BaseModel):
    """One player's position in a shot's 360 freeze frame."""

    model_config = ConfigDict(from_attributes=True)

    ordinal: int
    teammate: bool | None
    actor: bool | None
    keeper: bool | None
    x: float | None
    y: float | None


class ShotFreezeFrameResponse(BaseModel):
    """API response shape for a shot's 360 freeze frame."""

    model_config = ConfigDict(from_attributes=True)

    event_id: str
    match_id: int
    visible_area: list[float]
    players: list[FreezeFramePlayerResponse]


class FreezeFrameStore(Protocol):
    """Read-only contract for per-shot 360 freeze-frame lookups."""

    def get_freeze_frame(self, match_id: int, event_id: str) -> ShotFreezeFrameResponse | None:
        """Return the shot's freeze frame, or None when it has no 360 frame
        at all — never a placeholder."""


class BigQueryFreezeFrameStore:
    """FreezeFrameStore backed by oam_core.three_sixty_frames/three_sixty_players."""

    def get_freeze_frame(self, match_id: int, event_id: str) -> ShotFreezeFrameResponse | None:
        client = _client()
        parameters = [
            bigquery.ScalarQueryParameter("match_id", "INT64", match_id),
            bigquery.ScalarQueryParameter("event_id", "STRING", event_id),
            bigquery.ScalarQueryParameter("silver_schema_version", "STRING", SILVER_SCHEMA_VERSION),
        ]

        frame_query = f"""
            SELECT visible_area, frame_player_count
            FROM `{PROJECT}.{DATASET}.three_sixty_frames`
            WHERE match_id = @match_id AND event_uuid = @event_id
              AND silver_schema_version = @silver_schema_version
        """
        frame_rows = list(
            client.query(frame_query, job_config=bigquery.QueryJobConfig(query_parameters=parameters)).result()
        )
        if not frame_rows:
            return None
        frame_row = frame_rows[0]

        players_query = f"""
            SELECT frame_player_ordinal, teammate, actor, keeper, x, y
            FROM `{PROJECT}.{DATASET}.three_sixty_players`
            WHERE match_id = @match_id AND event_uuid = @event_id
              AND silver_schema_version = @silver_schema_version
            ORDER BY frame_player_ordinal
        """
        player_rows = client.query(
            players_query, job_config=bigquery.QueryJobConfig(query_parameters=parameters)
        ).result()

        return ShotFreezeFrameResponse(
            event_id=event_id,
            match_id=match_id,
            visible_area=list(frame_row["visible_area"] or []),
            players=[
                FreezeFramePlayerResponse(
                    ordinal=row["frame_player_ordinal"],
                    teammate=row["teammate"],
                    actor=row["actor"],
                    keeper=row["keeper"],
                    x=row["x"],
                    y=row["y"],
                )
                for row in player_rows
            ],
        )
