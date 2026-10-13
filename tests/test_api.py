"""Regression tests: unvalidated models must never produce predictions."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from chicago_crime_api.main import create_app
from chicago_crime_api.services.prediction import PredictionService


@pytest.fixture
def client():
    from chicago_crime_api.api.predictions import get_prediction_service

    service = PredictionService.__new__(PredictionService)
    service.model = None
    service.model_version = "not_loaded"
    application = create_app()
    application.dependency_overrides[get_prediction_service] = lambda: service
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/api/v1/predictions/", {
            "latitude": 41.88, "longitude": -87.63,
            "prediction_date": "2024-12-15", "horizon_days": 7,
        }),
        ("/api/v1/predictions/batch", {
            "requests": [{"latitude": 41.88, "longitude": -87.63,
                          "prediction_date": "2024-12-15"}],
        }),
        ("/api/v1/predictions/grid", {
            "prediction_date": "2024-12-15", "grid_resolution": 10,
        }),
    ],
)
def test_predictions_unavailable_without_validated_model(client, endpoint, payload):
    response = client.post(endpoint, json=payload)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "MODEL_NOT_VALIDATED"


def test_no_synthetic_fallback(tmp_path):
    """Missing model must never silently fit a model on random samples."""
    class Settings:
        model_path = str(tmp_path)

    service = PredictionService.__new__(PredictionService)
    service.settings = Settings()
    service._load_model()
    assert service.model is None
    assert service.model_version == "not_loaded"


def test_unvalidated_artifact_does_not_get_deserialized(tmp_path):
    (tmp_path / "crime_model.joblib").write_bytes(b"not-a-trusted-model")
    (tmp_path / "model_metadata.json").write_text(
        '{"version":"test","validation_status":"unvalidated"}',
        encoding="utf-8",
    )

    class Settings:
        model_path = str(tmp_path)

    service = PredictionService.__new__(PredictionService)
    service.settings = Settings()
    with patch("chicago_crime_api.services.prediction.joblib.load") as unsafe_load:
        service._load_model()
    unsafe_load.assert_not_called()
    assert service.model is None


def test_invalid_coordinates_rejected_before_inference(client):
    response = client.post("/api/v1/predictions/", json={
        "latitude": 50.0, "longitude": -87.63,
        "prediction_date": "2024-12-15",
    })
    assert response.status_code == 422


def test_liveness_independent_of_model(client):
    response = client.get("/api/v1/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
