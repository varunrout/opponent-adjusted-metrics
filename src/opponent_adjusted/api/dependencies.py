"""FastAPI dependency providers for the dashboard API."""

from __future__ import annotations

from typing import Literal

from opponent_adjusted.api.bigquery_store import BigQueryServingStore
from opponent_adjusted.api.interfaces import ServingStore

Role = Literal["guest", "viewer", "admin"]


def get_role() -> Role:
    """Stub role dependency; Firebase Auth will replace this in a later phase.

    Always resolves to admin for now.
    """
    return "admin"


def get_store() -> ServingStore:
    """FastAPI dependency provider for the serving store; overridable in tests."""
    return BigQueryServingStore()
