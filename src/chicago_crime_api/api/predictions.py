"""API routes for predictions."""

import time
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from chicago_crime_api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    GridPredictionRequest,
    GridPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from chicago_crime_api.services.prediction import PredictionService, get_prediction_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
async def predict_crime(
    request: PredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResponse:
    """
    Predict crime count for a specific location and date.

    - **latitude**: Latitude within Chicago bounds (41.64 - 42.02)
    - **longitude**: Longitude within Chicago bounds (-87.94 - -87.52)
    - **date**: Target date for prediction
    - **horizon_days**: Number of days to predict (default: 7)
    """
    start_time = time.time()

    try:
        prediction = service.predict(
            lat=request.latitude,
            lon=request.longitude,
            prediction_date=request.date,
            horizon_days=request.horizon_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    inference_time = (time.time() - start_time) * 1000

    return PredictionResponse(
        prediction=prediction,
        model_version=service.model_version,
        prediction_timestamp=datetime.utcnow(),
        inference_time_ms=round(inference_time, 2),
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_crime_batch(
    request: BatchPredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> BatchPredictionResponse:
    """
    Predict crime counts for multiple locations.

    Maximum 100 predictions per batch request.
    """
    start_time = time.time()

    predictions = []
    for req in request.requests:
        try:
            pred = service.predict(
                lat=req.latitude,
                lon=req.longitude,
                prediction_date=req.date,
                horizon_days=req.horizon_days,
            )
            predictions.append(pred)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for ({req.latitude}, {req.longitude}): {str(e)}",
            )

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=service.model_version,
        prediction_timestamp=datetime.utcnow(),
        total_inference_time_ms=round(total_time, 2),
    )


@router.post("/grid", response_model=GridPredictionResponse)
async def predict_crime_grid(
    request: GridPredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> GridPredictionResponse:
    """
    Predict crime counts for entire Chicago grid.

    Returns a 2D grid of predicted crime counts and risk levels.
    """
    try:
        return service.predict_grid(
            prediction_date=request.date,
            horizon_days=request.horizon_days,
            resolution=request.grid_resolution,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid prediction failed: {str(e)}")
