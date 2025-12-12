"""API routes for crime predictions.

Provides endpoints for location-based predictions, hotspots,
grid predictions, and multi-week forecasts.
"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from chicago_crime_api.schemas import (
    CellCenter,
    GridCellPrediction,
    GridPredictionRequest,
    GridPredictionResponse,
    GridSummary,
    HorizonPredictionRequest,
    HorizonPredictionResponse,
    HorizonWeekPrediction,
    HotspotPrediction,
    HotspotRequest,
    HotspotResponse,
    LocationPredictionRequest,
    LocationPredictionResponse,
)
from chicago_crime_api.services.prediction import (
    PredictionService,
    get_prediction_service,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/location", response_model=LocationPredictionResponse)
async def predict_for_location(
    request: LocationPredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> LocationPredictionResponse:
    """
    Predict weekly crime count for a specific location.

    Uses historical data for the grid cell containing the location
    to predict expected crimes for the specified week.

    - **latitude**: Latitude within Chicago bounds (41.64 - 42.02)
    - **longitude**: Longitude within Chicago bounds (-87.94 - -87.52)
    - **prediction_date**: Target date for prediction (start of week)
    """
    try:
        result = service.predict_for_location(
            latitude=request.latitude,
            longitude=request.longitude,
            target_date=request.prediction_date,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from None

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Build response
    cell_center = None
    if result.get("cell_center"):
        cell_center = CellCenter(
            latitude=result["cell_center"]["latitude"],
            longitude=result["cell_center"]["longitude"],
        )

    return LocationPredictionResponse(
        grid_id=result["grid_id"],
        latitude=result["latitude"],
        longitude=result["longitude"],
        cell_center=cell_center,
        prediction_date=result["prediction_date"],
        predicted_count=result["predicted_count"],
        confidence_lower=result.get("confidence_lower"),
        confidence_upper=result.get("confidence_upper"),
        risk_level=result["risk_level"],
        model_id=result.get("model_id", "unknown"),
        model_version=result.get("model_version", "unknown"),
        is_baseline=result.get("is_baseline", False),
        note=result.get("note"),
    )


@router.get("/location", response_model=LocationPredictionResponse)
async def predict_for_location_get(
    latitude: float = Query(..., ge=41.64, le=42.02, description="Latitude"),
    longitude: float = Query(..., ge=-87.94, le=-87.52, description="Longitude"),
    prediction_date: date = Query(..., description="Prediction date"),
    service: PredictionService = Depends(get_prediction_service),
) -> LocationPredictionResponse:
    """
    Predict weekly crime count for a specific location (GET method).

    Convenience endpoint for simple GET requests.
    """
    request = LocationPredictionRequest(
        latitude=latitude,
        longitude=longitude,
        prediction_date=prediction_date,
    )
    return await predict_for_location(request, service)


@router.post("/hotspots", response_model=HotspotResponse)
async def predict_hotspots(
    request: HotspotRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> HotspotResponse:
    """
    Predict top crime hotspots for a given date.

    Returns the N cells with highest predicted crime counts,
    ranked by predicted count.

    - **prediction_date**: Target date for prediction
    - **top_n**: Number of hotspots to return (default: 10, max: 100)
    """
    try:
        results = service.predict_hotspots(
            target_date=request.prediction_date,
            top_n=request.top_n,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Hotspot prediction failed: {str(e)}"
        ) from None

    # Build hotspot predictions
    hotspots = []
    for r in results:
        cell_center = None
        if r.get("cell_center"):
            cell_center = CellCenter(
                latitude=r["cell_center"]["latitude"],
                longitude=r["cell_center"]["longitude"],
            )

        hotspots.append(
            HotspotPrediction(
                rank=r["rank"],
                grid_id=r["grid_id"],
                cell_center=cell_center,
                prediction_date=r["prediction_date"],
                predicted_count=r["predicted_count"],
                confidence_lower=r.get("confidence_lower"),
                confidence_upper=r.get("confidence_upper"),
                risk_level=r["risk_level"],
                historical_average=r.get("historical_average", 0),
            )
        )

    return HotspotResponse(
        prediction_date=request.prediction_date.isoformat(),
        hotspots=hotspots,
        model_version=service.model_version,
        total_hotspots=len(hotspots),
    )


@router.get("/hotspots", response_model=HotspotResponse)
async def predict_hotspots_get(
    prediction_date: date = Query(..., description="Prediction date"),
    top_n: int = Query(default=10, ge=1, le=100, description="Number of hotspots"),
    service: PredictionService = Depends(get_prediction_service),
) -> HotspotResponse:
    """
    Predict top crime hotspots (GET method).

    Convenience endpoint for simple GET requests.
    """
    request = HotspotRequest(prediction_date=prediction_date, top_n=top_n)
    return await predict_hotspots(request, service)


@router.post("/grid", response_model=GridPredictionResponse)
async def predict_grid(
    request: GridPredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> GridPredictionResponse:
    """
    Predict crime counts for all active grid cells.

    Returns predictions for all historically active grid cells
    in Chicago, along with summary statistics.

    - **prediction_date**: Target date for prediction
    - **bounds**: Optional geographic bounds to filter results
    """
    try:
        result = service.predict_grid(
            target_date=request.prediction_date,
            bounds=request.bounds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid prediction failed: {str(e)}") from None

    # Build grid cell predictions
    grid_data = []
    for cell in result.get("grid_data", []):
        cell_center = None
        if cell.get("cell_center"):
            cell_center = CellCenter(
                latitude=cell["cell_center"]["latitude"],
                longitude=cell["cell_center"]["longitude"],
            )

        grid_data.append(
            GridCellPrediction(
                grid_id=cell["grid_id"],
                cell_center=cell_center,
                predicted_count=cell["predicted_count"],
                confidence_lower=cell.get("confidence_lower"),
                confidence_upper=cell.get("confidence_upper"),
                risk_level=cell["risk_level"],
            )
        )

    # Build summary
    summary_data = result.get("summary", {})
    summary = GridSummary(
        total_cells=summary_data.get("total_cells", 0),
        total_predicted_crimes=summary_data.get("total_predicted_crimes", 0),
        mean_per_cell=summary_data.get("mean_per_cell", 0),
        max_per_cell=summary_data.get("max_per_cell", 0),
        high_risk_cells=summary_data.get("high_risk_cells", 0),
    )

    return GridPredictionResponse(
        prediction_date=result.get("prediction_date", request.prediction_date.isoformat()),
        model_id=result.get("model_id", "unknown"),
        model_version=result.get("model_version", service.model_version),
        grid_data=grid_data,
        summary=summary,
    )


@router.post("/horizon", response_model=HorizonPredictionResponse)
async def predict_horizon(
    request: HorizonPredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> HorizonPredictionResponse:
    """
    Predict multiple weeks into the future for a location.

    Uses autoregressive approach - predictions become inputs
    for subsequent weeks.

    - **latitude**: Location latitude
    - **longitude**: Location longitude
    - **start_date**: First prediction date
    - **horizon_weeks**: Number of weeks to forecast (1-12)
    """
    try:
        results = service.predict_horizon(
            latitude=request.latitude,
            longitude=request.longitude,
            start_date=request.start_date,
            horizon_weeks=request.horizon_weeks,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Horizon prediction failed: {str(e)}"
        ) from None

    if results and "error" in results[0]:
        raise HTTPException(status_code=400, detail=results[0]["error"])

    # Get grid_id from first result or compute
    grid_id = results[0].get("grid_id", 0) if results else 0

    # Build weekly predictions
    predictions = [
        HorizonWeekPrediction(
            week=r["week"],
            prediction_date=r["prediction_date"],
            predicted_count=r["predicted_count"],
            confidence_lower=r.get("confidence_lower"),
            confidence_upper=r.get("confidence_upper"),
            risk_level=r["risk_level"],
        )
        for r in results
        if "error" not in r
    ]

    return HorizonPredictionResponse(
        grid_id=grid_id,
        latitude=request.latitude,
        longitude=request.longitude,
        start_date=request.start_date.isoformat(),
        horizon_weeks=request.horizon_weeks,
        predictions=predictions,
        model_version=service.model_version,
    )


@router.get("/model/info")
async def get_model_info(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict:
    """
    Get information about the currently loaded prediction model.

    Returns model ID, version, description, and statistics.
    """
    return service.model_info


@router.get("/model/health")
async def get_model_health(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict:
    """
    Check model health status.

    Returns health status of all registered models and cache statistics.
    """
    return service.health_check()
