"""FastAPI application entrypoint for the OAM dashboard backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opponent_adjusted.api.routers import (
    analysis,
    competitions,
    cxg_coverage,
    matches,
    me,
    players,
    shots,
    teams,
)

app = FastAPI(title="OAM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        # TODO(deploy): fill in the real Firebase Hosting URL once the
        # hosting site exists -- it's <site-id>.web.app, and <site-id>
        # defaults to the Firebase project ID (oam-varun-260819) but can
        # differ if a non-default hosting site was created. Confirm the
        # actual value from `firebase hosting:sites:list` rather than
        # assuming it matches the project ID, then replace this line.
        # "https://oam-varun-260819.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(competitions.router)
app.include_router(cxg_coverage.router)
app.include_router(matches.router)
app.include_router(me.router)
app.include_router(players.router)
app.include_router(shots.router)
app.include_router(teams.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
