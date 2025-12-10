"""Pydantic schemas for API request/response models."""

import datetime as dt

from pydantic import BaseModel, Field

# ============================================================================
# Request Schemas
# ============================================================================


class PredictionRequest(BaseModel):
    """Request schema for crime prediction."""

    latitude: float = Field(..., ge=41.64, le=42.02, description="Latitude within Chicago bounds")
    longitude: float = Field(
        ..., ge=-87.94, le=-87.52, description="Longitude within Chicago bounds"
    )
    prediction_date: dt.date = Field(..., description="Date for prediction")
    horizon_days: int = Field(default=7, ge=1, le=30, description="Prediction horizon in days")

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 41.88,
                "longitude": -87.63,
                "prediction_date": "2024-12-15",
                "horizon_days": 7,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=100)


class GridPredictionRequest(BaseModel):
    """Request schema for grid-based prediction."""

    prediction_date: dt.date = Field(..., description="Date for prediction")
    horizon_days: int = Field(default=7, ge=1, le=30)
    grid_resolution: int = Field(default=10, ge=5, le=50, description="Grid resolution (NxN)")


# ============================================================================
# Response Schemas
# ============================================================================


class CrimePrediction(BaseModel):
    """Single crime prediction result."""

    latitude: float
    longitude: float
    cell_id: int
    prediction_date: dt.date
    predicted_count: float = Field(..., description="Predicted crime count")
    confidence_lower: float = Field(..., description="95% CI lower bound")
    confidence_upper: float = Field(..., description="95% CI upper bound")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""

    prediction: CrimePrediction
    model_version: str
    prediction_timestamp: dt.datetime
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[CrimePrediction]
    model_version: str
    prediction_timestamp: dt.datetime
    total_inference_time_ms: float


class GridPredictionResponse(BaseModel):
    """Response schema for grid prediction."""

    grid: list[list[float]] = Field(..., description="2D grid of predicted counts")
    risk_grid: list[list[str]] = Field(..., description="2D grid of risk levels")
    prediction_date: dt.date
    horizon_days: int
    grid_resolution: int
    bounds: dict = Field(
        ...,
        description="Geographic bounds",
        json_schema_extra={
            "example": {"lat_min": 41.64, "lat_max": 42.02, "lon_min": -87.94, "lon_max": -87.52}
        },
    )
    model_version: str
    prediction_timestamp: dt.datetime


# ============================================================================
# Health & Info Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    environment: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_version: str
    trained_at: dt.datetime
    features: list[str]
    metrics: dict
    grid_shape: tuple[int, int]


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: str | None = None
    status_code: int
