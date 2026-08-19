"""GCS raw StatsBomb JSON storage."""

from __future__ import annotations

import json

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

from opponent_adjusted.storage.interfaces import JsonPayload


class GCSRawStatsBombStore:
    """Persist raw StatsBomb payloads to versioned immutable GCS object keys."""

    def __init__(
        self,
        bucket_name: str,
        data_version: str,
        *,
        client: storage.Client | None = None,
    ) -> None:
        self.client = client or storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.data_version = data_version
        self.prefix = f"raw/statsbomb/{data_version}"

    def _upload_create_only(self, object_name: str, payload: JsonPayload) -> bool:
        blob = self.bucket.blob(object_name)
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        try:
            # Enforce create-only semantics for immutable published versions.
            blob.upload_from_string(
                body,
                content_type="application/json",
                if_generation_match=0,
            )
            return True
        except PreconditionFailed:
            return False

    def _events_object_name(self, match_id: int) -> str:
        return f"{self.prefix}/events/{match_id}.json"

    def has_events(self, match_id: int) -> bool:
        blob = self.bucket.blob(self._events_object_name(match_id))
        return bool(blob.exists(client=self.client))

    def write_competitions(self, payload: JsonPayload, *, force: bool = False) -> bool:
        if force:
            raise ValueError("force overwrite is not supported for immutable GCS raw landing")
        return self._upload_create_only(f"{self.prefix}/competitions.json", payload)

    def write_matches(
        self,
        competition_id: int,
        season_id: int,
        payload: JsonPayload,
        *,
        force: bool = False,
    ) -> bool:
        if force:
            raise ValueError("force overwrite is not supported for immutable GCS raw landing")
        return self._upload_create_only(
            f"{self.prefix}/matches/{competition_id}/{season_id}.json",
            payload,
        )

    def write_events(self, match_id: int, payload: JsonPayload, *, force: bool = False) -> bool:
        if force:
            raise ValueError("force overwrite is not supported for immutable GCS raw landing")
        return self._upload_create_only(self._events_object_name(match_id), payload)
