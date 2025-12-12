"""Services package."""

from chicago_crime_api.services.cache import DataCache, TTLCache, get_data_cache
from chicago_crime_api.services.feature_builder import FeatureBuilder
from chicago_crime_api.services.grid_mapper import GridCell, GridMapper
from chicago_crime_api.services.historical_data import (
    HistoricalDataService,
    WeeklyCrimeRecord,
    get_historical_data_service,
)
from chicago_crime_api.services.prediction import PredictionService, get_prediction_service

__all__ = [
    # Cache
    "DataCache",
    "TTLCache",
    "get_data_cache",
    # Grid
    "GridCell",
    "GridMapper",
    # Historical Data
    "HistoricalDataService",
    "WeeklyCrimeRecord",
    "get_historical_data_service",
    # Features
    "FeatureBuilder",
    # Prediction
    "PredictionService",
    "get_prediction_service",
]
