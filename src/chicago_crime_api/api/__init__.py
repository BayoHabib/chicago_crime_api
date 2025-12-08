"""API routes package."""

from fastapi import APIRouter

from chicago_crime_api.api.health import router as health_router
from chicago_crime_api.api.predictions import router as predictions_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(predictions_router)

__all__ = ["api_router"]
