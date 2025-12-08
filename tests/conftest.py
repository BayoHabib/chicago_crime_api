"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from chicago_crime_api.main import create_app
from chicago_crime_api.services.prediction import PredictionService


@pytest.fixture(scope="session")
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = [0.5, 0.3, 0.7]
    return model


@pytest.fixture
def mock_prediction_service(mock_model):
    """Create a mock prediction service."""
    with patch.object(PredictionService, "load_model"):
        service = PredictionService()
        service.model = mock_model
        service._model_loaded = True
        yield service


@pytest.fixture
def app(mock_prediction_service):
    """Create test application with mocked services."""
    with patch(
        "chicago_crime_api.api.predictions.get_prediction_service",
        return_value=mock_prediction_service,
    ):
        application = create_app()
        yield application


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request payload."""
    return {
        "latitude": 41.8781,
        "longitude": -87.6298,
        "hour": 14,
        "day_of_week": 2,
        "month": 6,
    }


@pytest.fixture
def sample_grid_request():
    """Sample grid prediction request payload."""
    return {
        "center_latitude": 41.8781,
        "center_longitude": -87.6298,
        "radius_km": 1.0,
        "grid_size": 5,
        "hour": 14,
        "day_of_week": 2,
        "month": 6,
    }
