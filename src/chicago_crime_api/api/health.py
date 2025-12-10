"""API routes for health and system info."""

from typing import Annotated

from fastapi import APIRouter, Depends

from chicago_crime_api.config import Settings, get_settings
from chicago_crime_api.schemas import HealthResponse
from chicago_crime_api.services.prediction import PredictionService, get_prediction_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> HealthResponse:
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        model_loaded=service.is_loaded,
        model_version=service.model_version if service.is_loaded else None,
    )


@router.get("/ready")
async def readiness_check(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict:
    """Kubernetes readiness probe."""
    if not service.is_loaded:
        return {"status": "not_ready", "reason": "model_not_loaded"}
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}
