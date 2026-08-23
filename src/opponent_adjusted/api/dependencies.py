"""FastAPI dependency providers for the dashboard API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

from fastapi import Depends, Header, HTTPException

from opponent_adjusted.api.analysis_interfaces import AnalysisStore
from opponent_adjusted.api.bigquery_analysis_store import BigQueryAnalysisStore
from opponent_adjusted.api.bigquery_store import BigQueryServingStore
from opponent_adjusted.api.cxg_coverage import BigQueryCxgCoverageStore, CxgCoverageStore
from opponent_adjusted.api.interfaces import ServingStore

logger = logging.getLogger(__name__)

Role = Literal["guest", "viewer", "admin"]
_VALID_ROLES = ("guest", "viewer", "admin")


class DecodedToken(TypedDict, total=False):
    uid: str
    email: str | None
    role: str


TokenVerifier = Callable[[str], DecodedToken]

_firebase_app = None
_firebase_init_attempted = False


def _init_firebase_admin():
    """Lazily initialize Firebase Admin; never raises, degrades to None on any failure."""
    global _firebase_app, _firebase_init_attempted
    if _firebase_init_attempted:
        return _firebase_app
    _firebase_init_attempted = True
    try:
        import firebase_admin
        from firebase_admin import credentials

        service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        cred = (
            credentials.Certificate(service_account_path)
            if service_account_path
            else credentials.ApplicationDefault()
        )
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin initialized")
    except Exception:
        logger.warning("Firebase Admin unavailable; degrading to guest-only auth", exc_info=True)
        _firebase_app = None
    return _firebase_app


def _default_token_verifier(token: str) -> DecodedToken:
    app = _init_firebase_admin()
    if app is None:
        raise RuntimeError("Firebase Admin is not initialized; cannot verify token")
    from firebase_admin import auth as firebase_auth

    decoded = firebase_auth.verify_id_token(token, app=app)
    return DecodedToken(
        uid=decoded.get("uid", ""),
        email=decoded.get("email"),
        role=decoded.get("role", "guest"),
    )


def get_token_verifier() -> TokenVerifier:
    """FastAPI dependency provider for the token verifier; overridable in tests."""
    return _default_token_verifier


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Return the bearer token, or None if no Authorization header was sent.

    Raises 401 if a header IS present but malformed (not a well-formed `Bearer <token>`).
    """
    if authorization is None:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    return parts[1]


@dataclass(frozen=True)
class AuthContext:
    role: Role
    uid: str | None
    email: str | None


def get_auth_context(
    authorization: str | None = Header(default=None),
    verifier: TokenVerifier = Depends(get_token_verifier),
) -> AuthContext:
    """Resolve the caller's auth context.

    No Authorization header at all resolves to guest (anonymous access by design).
    A malformed header, or a token that fails verification, is a 401 — not a silent
    fallback to guest, since presenting a bad credential is different from presenting none.
    """
    token = _extract_bearer_token(authorization)
    if token is None:
        return AuthContext(role="guest", uid=None, email=None)
    try:
        decoded = verifier(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    role = decoded.get("role")
    if role not in _VALID_ROLES:
        role = "guest"
    return AuthContext(role=role, uid=decoded.get("uid"), email=decoded.get("email"))


def get_role(ctx: AuthContext = Depends(get_auth_context)) -> Role:
    """Thin wrapper kept for backward compatibility — every existing router already
    depends on `get_role` with this exact signature; don't change those call sites."""
    return ctx.role


def get_store() -> ServingStore:
    """FastAPI dependency provider for the serving store; overridable in tests."""
    return BigQueryServingStore()


def require_admin(role: Role = Depends(get_role)) -> Role:
    """Gate a route to admin only. Raises 403 for guest/viewer."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return role


def get_analysis_store() -> AnalysisStore:
    """FastAPI dependency provider for the oam_analysis store; overridable in tests."""
    return BigQueryAnalysisStore()


def get_cxg_coverage_store() -> CxgCoverageStore:
    """FastAPI dependency provider for the CxG v3 coverage store; overridable in tests."""
    return BigQueryCxgCoverageStore()
