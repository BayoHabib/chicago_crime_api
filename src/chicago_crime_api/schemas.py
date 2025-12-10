"""Pydantic schemas for API request/response models."""

import datetime as dt

from pydantic import BaseModel, Field

# ============================================================================
# Request Schemas
# ============================================================================


class PredictionRequest(BaseModel):
    """Request schema for weekly crime prediction."""

    prediction_date: dt.date = Field(..., description="Start date for prediction week")
    horizon_weeks: int = Field(
        default=1, ge=1, le=4, description="Number of weeks to predict (1-4)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction_date": "2024-12-15",
                "horizon_weeks": 2,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions (multiple dates)."""

    dates: list[dt.date] = Field(
        ..., min_length=1, max_length=52, description="List of dates to predict"
    )
    horizon_weeks: int = Field(default=1, ge=1, le=4)


class GridPredictionRequest(BaseModel):
    """Request schema for grid-based prediction (kept for compatibility)."""

    prediction_date: dt.date = Field(..., description="Date for prediction")
    horizon_weeks: int = Field(default=1, ge=1, le=4)


# ============================================================================
# Response Schemas
# ============================================================================


class WeeklyPrediction(BaseModel):
    """Single weekly crime prediction result."""

    week_start: dt.date
    week_number: int = Field(..., description="Week number of the year (1-52)")
    predicted_count: float = Field(..., description="Predicted total crime count for the week")
    confidence_lower: float = Field(..., description="95% CI lower bound")
    confidence_upper: float = Field(..., description="95% CI upper bound")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")


class PredictionResponse(BaseModel):
    """Response schema for weekly prediction."""

    predictions: list[WeeklyPrediction]
    model_version: str
    prediction_timestamp: dt.datetime
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[WeeklyPrediction]
    model_version: str
    prediction_timestamp: dt.datetime
    total_inference_time_ms: float


class GridPredictionResponse(BaseModel):
    """Response schema for grid prediction (legacy compatibility)."""

    grid: list[list[float]] = Field(..., description="2D grid of predicted counts")
    risk_grid: list[list[str]] = Field(..., description="2D grid of risk levels")
    prediction_date: dt.date
    horizon_weeks: int
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
