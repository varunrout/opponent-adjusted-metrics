"""FastAPI service for opponent-adjusted metrics inference."""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from opponent_adjusted.config import settings
from opponent_adjusted.api.schemas import (
    HealthResponse,
    ModelVersionResponse,
    ShotPredictionRequest,
    ShotPredictionResponse,
    PlayerAggregateResponse,
    TeamAggregateResponse,
)
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.db.models import (
    ModelRegistry,
    ShotPrediction,
    AggregatesPlayer,
    AggregatesTeam,
    Player,
    Team,
)
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Opponent-Adjusted Metrics API",
    description="API for contextual, opponent-adjusted football metrics",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/models/cxg/version", response_model=ModelVersionResponse)
async def get_cxg_model_version():
    """Get current CxG model version."""
    with session_scope() as session:
        model = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.model_name == "cxg")
            .order_by(ModelRegistry.created_at.desc())
            .first()
        )

        if not model:
            raise HTTPException(status_code=404, detail="CxG model not found")

        return ModelVersionResponse(
            model_name=model.model_name,
            version=model.version,
            algorithm=model.algorithm,
            trained_on_version_tag=model.trained_on_version_tag,
            created_at=model.created_at,
        )


@app.post("/predict/cxg", response_model=ShotPredictionResponse)
async def predict_cxg(request: ShotPredictionRequest):
    """Predict CxG for a shot (placeholder - requires trained model)."""
    # This is a placeholder implementation
    # In production, would load model and make actual prediction
    raise HTTPException(
        status_code=501,
        detail="Prediction endpoint requires trained model artifact. "
        "Please run training pipeline first.",
    )


@app.get("/aggregates/player", response_model=List[PlayerAggregateResponse])
async def get_player_aggregates(
    model: str = Query("cxg_v1", description="Model version"),
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
):
    """Get player-level aggregates."""
    with session_scope() as session:
        # Get model
        model_obj = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.version == model)
            .first()
        )

        if not model_obj:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")

        # Get aggregates
        aggregates = (
            session.query(AggregatesPlayer, Player)
            .join(Player, AggregatesPlayer.player_id == Player.id)
            .filter(AggregatesPlayer.model_id == model_obj.id)
            .order_by(AggregatesPlayer.summed_cxg.desc())
            .limit(limit)
            .all()
        )

        results = []
        for agg, player in aggregates:
            results.append(
                PlayerAggregateResponse(
                    player_id=player.id,
                    player_name=player.name,
                    shots_count=agg.shots_count,
                    summed_cxg=agg.summed_cxg,
                    summed_neutral_cxg=agg.summed_neutral_cxg,
                    summed_oppadj_diff=agg.summed_oppadj_diff,
                    avg_oppadj_diff=agg.avg_oppadj_diff,
                )
            )

        return results


@app.get("/aggregates/team", response_model=List[TeamAggregateResponse])
async def get_team_aggregates(
    model: str = Query("cxg_v1", description="Model version"),
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
):
    """Get team-level aggregates."""
    with session_scope() as session:
        # Get model
        model_obj = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.version == model)
            .first()
        )

        if not model_obj:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")

        # Get aggregates
        aggregates = (
            session.query(AggregatesTeam, Team)
            .join(Team, AggregatesTeam.team_id == Team.id)
            .filter(AggregatesTeam.model_id == model_obj.id)
            .order_by(AggregatesTeam.summed_cxg.desc())
            .limit(limit)
            .all()
        )

        results = []
        for agg, team in aggregates:
            results.append(
                TeamAggregateResponse(
                    team_id=team.id,
                    team_name=team.name,
                    shots_count=agg.shots_count,
                    summed_cxg=agg.summed_cxg,
                    summed_neutral_cxg=agg.summed_neutral_cxg,
                    summed_oppadj_diff=agg.summed_oppadj_diff,
                    avg_oppadj_diff=agg.avg_oppadj_diff,
                )
            )

        return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
