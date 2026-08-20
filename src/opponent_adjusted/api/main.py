"""FastAPI application entrypoint for the OAM dashboard backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opponent_adjusted.api.routers import competitions, matches, players, shots, teams

app = FastAPI(title="OAM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(competitions.router)
app.include_router(matches.router)
app.include_router(players.router)
app.include_router(shots.router)
app.include_router(teams.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
