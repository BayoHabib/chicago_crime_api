"""Crime prediction service using EventFlow adapters and trained models."""

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from chicago_crime_api.config import get_settings
from chicago_crime_api.schemas import CrimePrediction, GridPredictionResponse

# Import EventFlow adapters
try:
    from eventflow.adapters import RasterAdapter, TableAdapter

    EVENTFLOW_AVAILABLE = True
except ImportError:
    EVENTFLOW_AVAILABLE = False

# Check chicago_crime_downloader availability
import importlib.util

DOWNLOADER_AVAILABLE = importlib.util.find_spec("chicago_crime_downloader") is not None


class PredictionService:
    """Service for making crime predictions using EventFlow adapters."""

    # Chicago geographic bounds
    LAT_MIN, LAT_MAX = 41.64, 42.02
    LON_MIN, LON_MAX = -87.94, -87.52

    def __init__(self) -> None:
        """Initialize prediction service."""
        self.settings = get_settings()
        self.model: Any = None
        self.table_adapter: Any = None
        self.raster_adapter: Any = None
        self.model_version: str = "not_loaded"
        self.model_info: dict[str, Any] = {}
        self._load_model()
        self._init_eventflow_adapters()

    def _init_eventflow_adapters(self) -> None:
        """Initialize EventFlow adapters for data transformation."""
        if not EVENTFLOW_AVAILABLE:
            return

        try:
            # TableAdapter for tabular crime features
            self.table_adapter = TableAdapter(
                categorical_columns=["crime_type", "location_type", "day_of_week"],
                numerical_columns=["latitude", "longitude", "hour", "month"],
                target_column="crime_count",
            )

            # RasterAdapter for spatial grid predictions
            self.raster_adapter = RasterAdapter(
                lat_column="latitude",
                lon_column="longitude",
                value_column="crime_count",
                resolution=(50, 50),
                bounds=(self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX),
            )

            self.model_info["eventflow_adapters"] = ["TableAdapter", "RasterAdapter"]
        except Exception as e:
            print(f"Warning: Failed to initialize EventFlow adapters: {e}")

    def _load_model(self) -> None:
        """Fail closed unless a locally approved and integrity-checked model exists.

        A valid training/validation pipeline must create the metadata manifest.
        Joblib/pickle can execute code when loaded: never point this service at
        untrusted model artifacts, even when a checksum is present.
        """
        model_path = Path(self.settings.model_path)
        artifact = model_path / "crime_model.joblib"
        manifest_path = model_path / "model_metadata.json"
        self.model = None
        self.model_version = "not_loaded"
        self.model_info = {}

        if not artifact.is_file() or not manifest_path.is_file():
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return
            if manifest.get("validation_status") != "validated":
                return
            if not manifest.get("validation_run_id") or not manifest.get("version"):
                return
            expected_sha = manifest.get("artifact_sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                return
            with artifact.open("rb") as model_file:
                digest = hashlib.file_digest(model_file, "sha256").hexdigest()
            if digest != expected_sha.lower():
                return
            # Artifact provenance and approval must be guaranteed by the deployment
            # pipeline; a checksum supplied alongside an untrusted pickle is insufficient.
            self.model = joblib.load(artifact)
            self.model_version = str(manifest["version"])
            self.model_info = manifest
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.model = None
            self.model_version = "not_loaded"
            self.model_info = {}

    def _get_model_version(self, model_path: Path) -> str:
        """Get model version from metadata file."""
        meta_file = model_path / "model_metadata.json"
        if meta_file.exists():
            import json

            with open(meta_file) as f:
                meta = json.load(f)
                version: str = meta.get("version", "unknown")
                return version
        return "unknown"

    def _load_model_info(self, model_path: Path) -> dict[str, Any]:
        """Load model information from metadata."""
        meta_file = model_path / "model_metadata.json"
        if meta_file.exists():
            import json

            with open(meta_file) as f:
                result: dict[str, Any] = json.load(f)
                return result
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

    def _extract_features(self, lat_bin: int, lon_bin: int, prediction_date: date) -> np.ndarray:
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
        if self.model is None:
            raise RuntimeError("Model not loaded")
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
            prediction_date=prediction_date,
            predicted_count=round(predicted_count, 2),
            confidence_lower=round(ci_lower, 2),
            confidence_upper=round(ci_upper, 2),
            risk_level=self._get_risk_level(predicted_count),
        )

    def predict_grid(
        self, prediction_date: date, horizon_days: int = 7, resolution: int = 10
    ) -> GridPredictionResponse:
        """Predict crime counts for entire grid."""
        grid = np.zeros((resolution, resolution))
        risk_grid = [["low"] * resolution for _ in range(resolution)]

        if self.model is None:
            raise RuntimeError("Model not loaded")
        for lat_bin in range(resolution):
            for lon_bin in range(resolution):
                features = self._extract_features(lat_bin, lon_bin, prediction_date)
                predicted = float(self.model.predict(features)[0]) * horizon_days
                grid[lat_bin, lon_bin] = round(predicted, 2)
                risk_grid[lat_bin][lon_bin] = self._get_risk_level(predicted)

        return GridPredictionResponse(
            grid=grid.tolist(),
            risk_grid=risk_grid,
            prediction_date=prediction_date,
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

    @property
    def has_eventflow(self) -> bool:
        """Check if EventFlow adapters are available."""
        return EVENTFLOW_AVAILABLE and self.table_adapter is not None

    @property
    def has_downloader(self) -> bool:
        """Check if chicago_crime_downloader is available."""
        return DOWNLOADER_AVAILABLE


# Global service instance
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get or create prediction service singleton."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
