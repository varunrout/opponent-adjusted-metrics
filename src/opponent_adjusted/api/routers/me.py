"""Auth identity endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from opponent_adjusted.api.dependencies import AuthContext, get_auth_context
from opponent_adjusted.api.models import MeResponse

router = APIRouter(prefix="/v1", tags=["auth"])


@router.get("/me", response_model=MeResponse)
def get_me(ctx: AuthContext = Depends(get_auth_context)) -> MeResponse:
    return MeResponse(role=ctx.role, uid=ctx.uid, email=ctx.email)
