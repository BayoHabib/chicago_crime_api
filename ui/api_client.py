"""API client for Chicago Crime Prediction API.

This module provides a clean interface to the prediction API.
Designed to be reusable - same interface can be used by Streamlit or React.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single prediction result."""

    latitude: float
    longitude: float
    predicted_crimes: float
    prediction_date: str
    grid_id: int | None = None
    confidence_lower: float | None = None
    confidence_upper: float | None = None


@dataclass
class HotspotResult:
    """Hotspot prediction result."""

    grid_id: int
    latitude: float
    longitude: float
    predicted_crimes: float
    risk_level: str
    rank: int


class CrimeAPIClient:
    """Client for Chicago Crime Prediction API.

    This client abstracts the API calls, making it easy to:
    - Use with Streamlit now
    - Port to React/TypeScript later (same interface)
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize API client.

        Args:
            base_url: Base URL of the prediction API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> dict[str, Any]:
        """Check API health status.

        Returns:
            Health status dict with status, model_loaded, etc.
        """
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def predict_single(
        self,
        latitude: float,
        longitude: float,
        prediction_date: date | None = None,
        weeks_ahead: int = 1,
    ) -> PredictionResult | None:
        """Get prediction for a single location.

        Args:
            latitude: Location latitude (41.64 to 42.02)
            longitude: Location longitude (-87.95 to -87.50)
            prediction_date: Date to predict for (default: next week)
            weeks_ahead: Weeks ahead to predict

        Returns:
            PredictionResult or None if request fails
        """
        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "weeks_ahead": weeks_ahead,
        }
        if prediction_date:
            payload["prediction_date"] = prediction_date.isoformat()

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/predictions/location",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Handle cell_center for lat/lon
            cell_center = data.get("cell_center", {})
            return PredictionResult(
                latitude=data["latitude"],
                longitude=data["longitude"],
                predicted_crimes=data["predicted_count"],
                prediction_date=data["prediction_date"],
                grid_id=data.get("grid_id"),
                confidence_lower=data.get("confidence_lower"),
                confidence_upper=data.get("confidence_upper"),
            )
        except requests.RequestException as e:
            logger.error(f"Single prediction failed: {e}")
            return None

    def predict_hotspots(
        self,
        prediction_date: date | None = None,
        top_n: int = 20,
    ) -> list[HotspotResult]:
        """Get top crime hotspots.

        Args:
            prediction_date: Date to predict for
            top_n: Number of hotspots to return

        Returns:
            List of HotspotResult sorted by predicted crimes (descending)
        """
        payload = {"top_n": top_n}
        if prediction_date:
            payload["prediction_date"] = prediction_date.isoformat()

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/predictions/hotspots",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            return [
                HotspotResult(
                    grid_id=h["grid_id"],
                    latitude=h.get("cell_center", {}).get("latitude", 0),
                    longitude=h.get("cell_center", {}).get("longitude", 0),
                    predicted_crimes=h["predicted_count"],
                    risk_level=h.get("risk_level", "unknown"),
                    rank=h.get("rank", i + 1),
                )
                for i, h in enumerate(data.get("hotspots", []))
            ]
        except requests.RequestException as e:
            logger.error(f"Hotspots prediction failed: {e}")
            return []

    def predict_grid(
        self,
        prediction_date: date | None = None,
        weeks_ahead: int = 1,
    ) -> list[dict[str, Any]]:
        """Get predictions for entire grid.

        Args:
            prediction_date: Date to predict for
            weeks_ahead: Weeks ahead to predict

        Returns:
            List of grid cell predictions
        """
        payload = {}
        if prediction_date:
            payload["prediction_date"] = prediction_date.isoformat()

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/predictions/grid",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            # API returns grid_data, convert to expected format
            grid_data = data.get("grid_data", [])
            # Map cell_center to lat/lon for map visualization
            return [
                {
                    "grid_id": cell.get("grid_id"),
                    "latitude": cell.get("cell_center", {}).get("latitude"),
                    "longitude": cell.get("cell_center", {}).get("longitude"),
                    "predicted_crimes": cell.get("predicted_count", 0),
                    "risk_level": cell.get("risk_level", "low"),
                }
                for cell in grid_data
                if cell.get("cell_center")
            ]
        except requests.RequestException as e:
            logger.error(f"Grid prediction failed: {e}")
            return []

    def get_crime_type_distribution(
        self,
        prediction_date: date | None = None,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Get crime type distribution across hotspots.

        Args:
            prediction_date: Date to predict for
            top_n: Number of hotspots to aggregate

        Returns:
            Dict with crime_types breakdown and total
        """
        params = {"top_n": top_n}
        if prediction_date:
            params["prediction_date"] = prediction_date.isoformat()

        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/predictions/crime-types",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Crime type distribution request failed: {e}")
            return {"error": str(e), "crime_types": {}}

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Model info dict with version, features, etc.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/predictions/model/info",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Model info request failed: {e}")
            return {"error": str(e)}
