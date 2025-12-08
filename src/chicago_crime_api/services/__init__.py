"""Services package."""

from chicago_crime_api.services.prediction import PredictionService, get_prediction_service

__all__ = ["PredictionService", "get_prediction_service"]
