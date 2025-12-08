"""Crime prediction service using trained models."""

import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl

from chicago_crime_api.config import get_settings
from chicago_crime_api.schemas import CrimePrediction, GridPredictionResponse


class PredictionService:
    """Service for making crime predictions."""

    # Chicago geographic bounds
    LAT_MIN, LAT_MAX = 41.64, 42.02
    LON_MIN, LON_MAX = -87.94, -87.52

    def __init__(self) -> None:
        """Initialize prediction service."""
        self.settings = get_settings()
        self.model: Optional[object] = None
        self.model_version: str = "not_loaded"
        self.model_info: dict = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        model_path = Path(self.settings.model_path)

        # Try to load the latest model
        model_file = model_path / "crime_model.joblib"
        if model_file.exists():
            self.model = joblib.load(model_file)
            self.model_version = self._get_model_version(model_path)
            self.model_info = self._load_model_info(model_path)
        else:
            # Use baseline model if no trained model exists
            self._create_baseline_model()

    def _create_baseline_model(self) -> None:
        """Create a simple baseline model for initial deployment."""
        from sklearn.linear_model import PoissonRegressor

        # Simple Poisson baseline
        self.model = PoissonRegressor(alpha=1.0)
        # Fit on dummy data (will be replaced by real training)
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.poisson(5, 100)
        self.model.fit(X_dummy, y_dummy)
        self.model_version = "baseline_v1"
        self.model_info = {
            "name": "PoissonRegressor",
            "type": "baseline",
            "features": ["lat_bin", "lon_bin", "day_of_week", "month", "hour"],
        }

    def _get_model_version(self, model_path: Path) -> str:
        """Get model version from metadata file."""
        meta_file = model_path / "model_metadata.json"
        if meta_file.exists():
            import json

            with open(meta_file) as f:
                meta = json.load(f)
                return meta.get("version", "unknown")
        return "unknown"

    def _load_model_info(self, model_path: Path) -> dict:
        """Load model information from metadata."""
        meta_file = model_path / "model_metadata.json"
        if meta_file.exists():
            import json

            with open(meta_file) as f:
                return json.load(f)
        return {}

    def _location_to_cell(self, lat: float, lon: float) -> tuple[int, int, int]:
        """Convert lat/lon to grid cell."""
        grid_h, grid_w = self.settings.grid_size

        lat_bin = int((lat - self.LAT_MIN) / (self.LAT_MAX - self.LAT_MIN) * grid_h)
        lon_bin = int((lon - self.LON_MIN) / (self.LON_MAX - self.LON_MIN) * grid_w)

        # Clamp to valid range
        lat_bin = max(0, min(lat_bin, grid_h - 1))
        lon_bin = max(0, min(lon_bin, grid_w - 1))

        cell_id = lat_bin * grid_w + lon_bin
        return lat_bin, lon_bin, cell_id

    def _extract_features(
        self, lat_bin: int, lon_bin: int, prediction_date: date
    ) -> np.ndarray:
        """Extract features for prediction."""
        features = [
            lat_bin,
            lon_bin,
            prediction_date.weekday(),
            prediction_date.month,
            12,  # Default hour (noon)
        ]
        return np.array(features).reshape(1, -1)

    def _get_risk_level(self, predicted_count: float) -> str:
        """Determine risk level from predicted count."""
        if predicted_count < 5:
            return "low"
        elif predicted_count < 15:
            return "medium"
        elif predicted_count < 30:
            return "high"
        else:
            return "critical"

    def predict(
        self, lat: float, lon: float, prediction_date: date, horizon_days: int = 7
    ) -> CrimePrediction:
        """Make a single prediction."""
        lat_bin, lon_bin, cell_id = self._location_to_cell(lat, lon)
        features = self._extract_features(lat_bin, lon_bin, prediction_date)

        # Predict
        predicted_count = float(self.model.predict(features)[0])

        # Scale by horizon
        predicted_count *= horizon_days

        # Confidence intervals (simple approximation for Poisson)
        std = np.sqrt(predicted_count)
        ci_lower = max(0, predicted_count - 1.96 * std)
        ci_upper = predicted_count + 1.96 * std

        return CrimePrediction(
            latitude=lat,
            longitude=lon,
            cell_id=cell_id,
            date=prediction_date,
            predicted_count=round(predicted_count, 2),
            confidence_lower=round(ci_lower, 2),
            confidence_upper=round(ci_upper, 2),
            risk_level=self._get_risk_level(predicted_count),
        )

    def predict_grid(
        self, prediction_date: date, horizon_days: int = 7, resolution: int = 10
    ) -> GridPredictionResponse:
        """Predict crime counts for entire grid."""
        start_time = time.time()

        grid = np.zeros((resolution, resolution))
        risk_grid = [["low"] * resolution for _ in range(resolution)]

        for lat_bin in range(resolution):
            for lon_bin in range(resolution):
                features = self._extract_features(lat_bin, lon_bin, prediction_date)
                predicted = float(self.model.predict(features)[0]) * horizon_days
                grid[lat_bin, lon_bin] = round(predicted, 2)
                risk_grid[lat_bin][lon_bin] = self._get_risk_level(predicted)

        inference_time = (time.time() - start_time) * 1000

        return GridPredictionResponse(
            grid=grid.tolist(),
            risk_grid=risk_grid,
            date=prediction_date,
            horizon_days=horizon_days,
            grid_resolution=resolution,
            bounds={
                "lat_min": self.LAT_MIN,
                "lat_max": self.LAT_MAX,
                "lon_min": self.LON_MIN,
                "lon_max": self.LON_MAX,
            },
            model_version=self.model_version,
            prediction_timestamp=datetime.utcnow(),
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Global service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get or create prediction service singleton."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
