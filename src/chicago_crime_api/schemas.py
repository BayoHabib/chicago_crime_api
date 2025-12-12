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


class LocationPredictionRequest(BaseModel):
    """Request schema for location-based prediction."""

    latitude: float = Field(..., ge=41.64, le=42.02, description="Latitude (Chicago bounds)")
    longitude: float = Field(..., ge=-87.94, le=-87.52, description="Longitude (Chicago bounds)")
    prediction_date: dt.date = Field(..., description="Date to predict for")

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 41.8781,
                "longitude": -87.6298,
                "prediction_date": "2024-12-15",
            }
        }
    }


class HotspotRequest(BaseModel):
    """Request schema for hotspot prediction."""

    prediction_date: dt.date = Field(..., description="Date to predict for")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of hotspots to return")


class HorizonPredictionRequest(BaseModel):
    """Request schema for multi-week horizon prediction."""

    latitude: float = Field(..., ge=41.64, le=42.02)
    longitude: float = Field(..., ge=-87.94, le=-87.52)
    start_date: dt.date = Field(..., description="First prediction date")
    horizon_weeks: int = Field(default=4, ge=1, le=12, description="Number of weeks to forecast")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions (multiple dates)."""

    dates: list[dt.date] = Field(
        ..., min_length=1, max_length=52, description="List of dates to predict"
    )
    horizon_weeks: int = Field(default=1, ge=1, le=4)


class GridPredictionRequest(BaseModel):
    """Request schema for grid-based prediction."""

    prediction_date: dt.date = Field(..., description="Date for prediction")
    bounds: dict[str, float] | None = Field(
        default=None, description="Optional geographic bounds to filter"
    )


# ============================================================================
# Response Schemas
# ============================================================================


class CellCenter(BaseModel):
    """Grid cell center coordinates."""

    latitude: float
    longitude: float


class LocationPredictionResponse(BaseModel):
    """Response for location-based prediction."""

    grid_id: int = Field(..., description="Grid cell identifier")
    latitude: float = Field(..., description="Request latitude")
    longitude: float = Field(..., description="Request longitude")
    cell_center: CellCenter | None = Field(None, description="Grid cell center")
    prediction_date: str = Field(..., description="Prediction date ISO format")
    predicted_count: float = Field(..., description="Predicted weekly crime count")
    confidence_lower: float | None = Field(None, description="95% CI lower bound")
    confidence_upper: float | None = Field(None, description="95% CI upper bound")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    is_baseline: bool = Field(default=False, description="Whether baseline was used")
    note: str | None = Field(default=None, description="Additional notes")


class HotspotPrediction(BaseModel):
    """Single hotspot prediction."""

    rank: int = Field(..., description="Hotspot rank (1 = highest)")
    grid_id: int
    cell_center: CellCenter | None = None
    prediction_date: str
    predicted_count: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    risk_level: str
    historical_average: float = Field(0, description="Historical average for comparison")


class HotspotResponse(BaseModel):
    """Response for hotspot predictions."""

    prediction_date: str
    hotspots: list[HotspotPrediction]
    model_version: str
    total_hotspots: int


class GridCellPrediction(BaseModel):
    """Single grid cell prediction."""

    grid_id: int
    cell_center: CellCenter | None = None
    predicted_count: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    risk_level: str


class GridSummary(BaseModel):
    """Summary statistics for grid predictions."""

    total_cells: int
    total_predicted_crimes: float
    mean_per_cell: float
    max_per_cell: float
    high_risk_cells: int


class GridPredictionResponse(BaseModel):
    """Response schema for grid prediction."""

    prediction_date: str
    model_id: str
    model_version: str
    grid_data: list[GridCellPrediction]
    summary: GridSummary


class HorizonWeekPrediction(BaseModel):
    """Single week prediction in horizon forecast."""

    week: int = Field(..., description="Week number (1-based)")
    prediction_date: str
    predicted_count: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    risk_level: str


class HorizonPredictionResponse(BaseModel):
    """Response for multi-week horizon prediction."""

    grid_id: int
    latitude: float
    longitude: float
    start_date: str
    horizon_weeks: int
    predictions: list[HorizonWeekPrediction]
    model_version: str


class WeeklyPrediction(BaseModel):
    """Single weekly crime prediction result (legacy)."""

    week_start: dt.date
    week_number: int = Field(..., description="Week number of the year (1-52)")
    predicted_count: float = Field(..., description="Predicted total crime count for the week")
    confidence_lower: float = Field(..., description="95% CI lower bound")
    confidence_upper: float = Field(..., description="95% CI upper bound")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")


class PredictionResponse(BaseModel):
    """Response schema for weekly prediction (legacy)."""

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
