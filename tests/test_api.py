"""Tests for prediction endpoints."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from chicago_crime_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data

    def test_readiness_check(self, client):
        """Test readiness probe."""
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ready", "not_ready"]

    def test_liveness_check(self, client):
        """Test liveness probe."""
        response = client.get("/api/v1/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_single_prediction(self, client):
        """Test single prediction endpoint."""
        response = client.post(
            "/api/v1/predictions/",
            json={
                "latitude": 41.88,
                "longitude": -87.63,
                "prediction_date": "2024-12-15",
                "horizon_days": 7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert "inference_time_ms" in data

        pred = data["prediction"]
        assert "predicted_count" in pred
        assert "confidence_lower" in pred
        assert "confidence_upper" in pred
        assert "risk_level" in pred
        assert pred["risk_level"] in ["low", "medium", "high", "critical"]

    def test_prediction_invalid_latitude(self, client):
        """Test prediction with invalid latitude."""
        response = client.post(
            "/api/v1/predictions/",
            json={
                "latitude": 50.0,  # Outside Chicago
                "longitude": -87.63,
                "prediction_date": "2024-12-15",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_prediction_invalid_longitude(self, client):
        """Test prediction with invalid longitude."""
        response = client.post(
            "/api/v1/predictions/",
            json={
                "latitude": 41.88,
                "longitude": -80.0,  # Outside Chicago
                "prediction_date": "2024-12-15",
            },
        )
        assert response.status_code == 422

    def test_batch_prediction(self, client):
        """Test batch prediction endpoint."""
        response = client.post(
            "/api/v1/predictions/batch",
            json={
                "requests": [
                    {"latitude": 41.88, "longitude": -87.63, "prediction_date": "2024-12-15"},
                    {"latitude": 41.79, "longitude": -87.68, "prediction_date": "2024-12-15"},
                    {"latitude": 41.95, "longitude": -87.65, "prediction_date": "2024-12-15"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3

    def test_grid_prediction(self, client):
        """Test grid prediction endpoint."""
        response = client.post(
            "/api/v1/predictions/grid",
            json={
                "prediction_date": "2024-12-15",
                "horizon_days": 7,
                "grid_resolution": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "grid" in data
        assert "risk_grid" in data
        assert len(data["grid"]) == 10
        assert len(data["grid"][0]) == 10


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
