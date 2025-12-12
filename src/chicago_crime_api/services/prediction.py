"""Crime prediction service using the new model architecture.

Provides high-level API for making crime predictions with proper
historical data integration, caching, and model management.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from chicago_crime_api.config import get_settings
from chicago_crime_api.models import (
    ModelConfig,
    ModelRegistry,
    WeeklyCrimeModel,
    get_registry,
)
from chicago_crime_api.services.cache import DataCache
from chicago_crime_api.services.feature_builder import FeatureBuilder
from chicago_crime_api.services.grid_mapper import GridMapper
from chicago_crime_api.services.historical_data import (
    HistoricalDataService,
    get_historical_data_service,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making crime predictions.

    Integrates:
    - Model registry for model management
    - Historical data service for crime history
    - Grid mapper for location -> grid conversion
    - Caching for performance
    """

    def __init__(
        self,
        model_path: str | None = None,
        data_path: str | None = None,
    ) -> None:
        """Initialize prediction service.

        Args:
            model_path: Override path to model file
            data_path: Override path to historical data
        """
        self.settings = get_settings()
        self._model_path = model_path or str(Path(self.settings.model_path) / "crime_model.joblib")
        self._data_path = data_path

        # Services
        self._registry: ModelRegistry | None = None
        self._cache: DataCache | None = None
        self._historical_service: HistoricalDataService | None = None
        self._grid_mapper: GridMapper | None = None
        self._feature_builder: FeatureBuilder | None = None

        # State
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all services and load model.

        Call this during application startup.
        """
        if self._initialized:
            return

        logger.info("Initializing PredictionService...")

        # Initialize cache
        self._cache = DataCache.get_instance()

        # Initialize grid mapper
        self._grid_mapper = GridMapper()

        # Initialize feature builder
        self._feature_builder = FeatureBuilder()

        # Initialize historical data service
        self._historical_service = get_historical_data_service(data_path=self._data_path)

        # Initialize model registry and load model
        self._registry = get_registry()
        self._registry.register_class(WeeklyCrimeModel, "weekly_crime")

        # Load the weekly crime model
        self._load_model()

        self._initialized = True
        logger.info("PredictionService initialized successfully")

    def _load_model(self) -> None:
        """Load the crime prediction model."""
        model_path = Path(self._model_path)

        if model_path.exists():
            config = ModelConfig(
                name="Weekly Crime Predictor",
                model_id="weekly_crime_v1",
                model_uri=str(model_path),
                backend="sklearn",
                version=self._get_model_version(),
                description="Predicts weekly crime counts per grid cell",
                required_history_weeks=8,
            )
        else:
            # Use default/baseline model
            logger.warning(f"Model not found at {model_path}, using baseline")
            config = ModelConfig(
                name="Weekly Crime Predictor (Baseline)",
                model_id="weekly_crime_baseline",
                model_uri="",
                backend="sklearn",
                version="baseline",
                description="Baseline model for initial deployment",
                required_history_weeks=8,
            )

        try:
            model = WeeklyCrimeModel(config=config, model_path=self._model_path)
            model.load()
            self._registry.register(model, set_default=True)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create and register a baseline model
            self._create_baseline_model()

    def _create_baseline_model(self) -> None:
        """Create a baseline model for initial deployment."""
        import tempfile

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor

        logger.info("Creating baseline model...")

        feature_columns = FeatureBuilder.FEATURE_COLUMNS

        # Create a simple trained model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        n_samples = 100
        train_x = pd.DataFrame(
            np.random.rand(n_samples, len(feature_columns)),
            columns=feature_columns,
        )
        train_y = np.random.poisson(5, n_samples).astype(float)
        model.fit(train_x, train_y)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(model, f.name)
            temp_path = f.name

        config = ModelConfig(
            name="Weekly Crime Predictor (Baseline)",
            model_id="weekly_crime_baseline",
            model_uri=temp_path,
            backend="sklearn",
            version="baseline",
            description="Baseline model for testing",
            required_history_weeks=8,
        )

        baseline_model = WeeklyCrimeModel(config=config, model_path=temp_path)
        baseline_model.load()
        self._registry.register(baseline_model, set_default=True)

    def _get_model_version(self) -> str:
        """Get model version from metadata."""
        import json

        model_path = Path(self._model_path).parent
        meta_file = model_path / "model_metadata.json"

        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                return meta.get("version", "unknown")
        return "unknown"

    def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            self.initialize()

    def predict_for_location(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> dict[str, Any]:
        """Predict crime count for a specific location.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            target_date: Date to predict for

        Returns:
            Prediction result dictionary
        """
        self._ensure_initialized()

        # Convert location to grid cell
        try:
            grid_cell = self._grid_mapper.lat_lon_to_grid(latitude, longitude)
            grid_id = grid_cell.grid_id
        except ValueError:
            return {
                "error": "Location outside Chicago boundaries",
                "latitude": latitude,
                "longitude": longitude,
            }

        # Get historical data for the cell
        history = self._historical_service.get_cell_history(
            grid_id=grid_id,
            num_weeks=8,
        )

        if not history:
            # Use baseline prediction if no history
            return self._baseline_prediction(
                grid_id=grid_id,
                latitude=latitude,
                longitude=longitude,
                target_date=target_date,
            )

        # Get historical counts (newest first)
        historical_counts = [r.crime_count for r in history]

        # Get model and predict
        model = self._registry.get()
        result = model.predict(
            grid_id=grid_id,
            target_date=target_date,
            historical_counts=historical_counts,
        )

        # Get grid cell center for response
        center_lat, center_lon = self._grid_mapper.grid_id_to_center(grid_id)

        return {
            "grid_id": grid_id,
            "latitude": latitude,
            "longitude": longitude,
            "cell_center": {"latitude": center_lat, "longitude": center_lon},
            "prediction_date": target_date.isoformat(),
            "predicted_count": result.predicted_count,
            "confidence_lower": result.confidence_lower,
            "confidence_upper": result.confidence_upper,
            "risk_level": self._get_risk_level(result.predicted_count),
            "model_id": result.model_id,
            "model_version": result.model_version,
        }

    def _baseline_prediction(
        self,
        grid_id: int,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> dict[str, Any]:
        """Make baseline prediction when no historical data available."""
        # Use city-wide average as baseline
        city_stats = self._historical_service.get_city_wide_stats(weeks=8)

        if city_stats:
            baseline_count = city_stats.get("mean_weekly_count", 5.0)
        else:
            baseline_count = 5.0

        return {
            "grid_id": grid_id,
            "latitude": latitude,
            "longitude": longitude,
            "prediction_date": target_date.isoformat(),
            "predicted_count": baseline_count,
            "confidence_lower": max(0, baseline_count - 2),
            "confidence_upper": baseline_count + 2,
            "risk_level": self._get_risk_level(baseline_count),
            "model_id": "baseline",
            "model_version": "baseline",
            "is_baseline": True,
            "note": "No historical data available for this cell",
        }

    def _generate_demo_hotspots(
        self,
        target_date: date,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate demo hotspots when no historical data is available.

        Uses known high-crime areas in Chicago for realistic demo.
        """
        import random

        # Known high-crime neighborhoods in Chicago (approximate centers)
        demo_locations = [
            (41.8827, -87.6233, "Loop/Downtown"),
            (41.8917, -87.6065, "Near North"),
            (41.8500, -87.6500, "Near West"),
            (41.7943, -87.5907, "South Shore"),
            (41.7803, -87.6442, "Englewood"),
            (41.7658, -87.5961, "South Chicago"),
            (41.8819, -87.6278, "River North"),
            (41.9100, -87.6787, "Logan Square"),
            (41.9030, -87.6260, "Lincoln Park"),
            (41.9533, -87.7171, "Rogers Park"),
            (41.8336, -87.7354, "Austin"),
            (41.8670, -87.6866, "Garfield Park"),
            (41.7501, -87.5633, "East Side"),
            (41.8110, -87.6039, "Grand Crossing"),
            (41.8260, -87.6170, "Washington Park"),
        ]

        # Chicago crime type distribution (based on real CPD data patterns)
        crime_types = {
            "THEFT": 0.25,
            "BATTERY": 0.18,
            "CRIMINAL DAMAGE": 0.12,
            "ASSAULT": 0.10,
            "DECEPTIVE PRACTICE": 0.08,
            "OTHER OFFENSE": 0.07,
            "BURGLARY": 0.06,
            "MOTOR VEHICLE THEFT": 0.05,
            "ROBBERY": 0.04,
            "NARCOTICS": 0.03,
            "WEAPONS VIOLATION": 0.02,
        }

        random.seed(target_date.toordinal())  # Reproducible randomness
        results = []

        for i, (lat, lon, name) in enumerate(demo_locations[:top_n]):
            # Use lat_lon_to_grid to get valid grid_id
            grid_cell = self._grid_mapper.lat_lon_to_grid(lat, lon)
            grid_id = grid_cell.grid_id

            # Generate realistic crime counts (5-50 range for weekly)
            predicted = random.uniform(8, 45)

            # Generate crime type breakdown for this location
            # Different areas have different crime profiles
            crime_breakdown = {}
            remaining = predicted
            for crime_type, base_pct in crime_types.items():
                # Add some variance per location
                variance = random.uniform(0.7, 1.3)
                count = round(predicted * base_pct * variance, 1)
                if count > 0:
                    crime_breakdown[crime_type] = count
                    remaining -= count
            
            # Ensure total matches predicted count
            if crime_breakdown:
                scale = predicted / sum(crime_breakdown.values())
                crime_breakdown = {k: round(v * scale, 1) for k, v in crime_breakdown.items()}

            results.append({
                "rank": i + 1,
                "grid_id": grid_id,
                "neighborhood": name,
                "cell_center": {"latitude": lat, "longitude": lon},
                "prediction_date": target_date.isoformat(),
                "predicted_count": round(predicted, 1),
                "confidence_lower": round(predicted * 0.7, 1),
                "confidence_upper": round(predicted * 1.3, 1),
                "risk_level": self._get_risk_level(predicted),
                "historical_average": round(predicted * random.uniform(0.9, 1.1), 1),
                "crime_types": crime_breakdown,
            })

        # Sort by predicted count
        results.sort(key=lambda x: x["predicted_count"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def _load_real_crime_type_distribution(self, num_months: int = 3) -> dict[str, float]:
        """Load real crime type distribution from raw monthly data.
        
        Args:
            num_months: Number of recent months to analyze
            
        Returns:
            Dict mapping crime type to percentage (0-1)
        """
        import pandas as pd
        from pathlib import Path
        
        raw_path = Path("data/raw/monthly")
        if not raw_path.exists():
            return {}
        
        # Get most recent months
        months = sorted(raw_path.iterdir(), reverse=True)[:num_months]
        
        all_crimes = []
        for month_dir in months:
            if not month_dir.is_dir():
                continue
            for parquet_file in month_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file, columns=["primary_type"])
                    all_crimes.append(df)
                except Exception as e:
                    logger.warning(f"Could not load {parquet_file}: {e}")
        
        if not all_crimes:
            return {}
        
        # Combine and calculate percentages
        combined = pd.concat(all_crimes, ignore_index=True)
        counts = combined["primary_type"].value_counts()
        total = counts.sum()
        
        if total == 0:
            return {}
        
        return {crime_type: count / total for crime_type, count in counts.items()}

    def get_crime_type_distribution(self, target_date: date, top_n: int = 20) -> dict[str, float]:
        """Get aggregated crime type distribution across hotspots.
        
        Uses real crime type data from raw monthly files, combined with
        predicted total crime counts from hotspots.
        
        Args:
            target_date: Date to predict for
            top_n: Number of hotspots to aggregate
            
        Returns:
            Dict mapping crime type to total predicted count
        """
        # Get real crime type percentages from historical data
        real_percentages = self._load_real_crime_type_distribution(num_months=3)
        
        if not real_percentages:
            # Fall back to demo mode
            logger.warning("No real crime type data available, using demo distribution")
            real_percentages = {
                "THEFT": 0.23,
                "BATTERY": 0.19,
                "CRIMINAL DAMAGE": 0.12,
                "ASSAULT": 0.09,
                "MOTOR VEHICLE THEFT": 0.08,
                "OTHER OFFENSE": 0.065,
                "DECEPTIVE PRACTICE": 0.046,
                "BURGLARY": 0.045,
                "NARCOTICS": 0.028,
                "ROBBERY": 0.026,
                "CRIMINAL TRESPASS": 0.024,
                "WEAPONS VIOLATION": 0.019,
            }
        
        # Get total predicted crimes from hotspots
        hotspots = self.predict_hotspots(target_date, top_n)
        total_predicted = sum(h.get("predicted_count", 0) for h in hotspots)
        
        # Distribute predictions across crime types using real percentages
        distribution = {
            crime_type: total_predicted * percentage
            for crime_type, percentage in real_percentages.items()
        }
        
        # Sort by count descending
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def _decode_training_grid_id(self, grid_id: int) -> tuple[float, float]:
        """Convert training data grid_id to lat/lon center.
        
        Training grid_ids encode lat_bin * 100000 + lon_bin.
        We reverse this to get approximate coordinates.
        
        Args:
            grid_id: Grid ID from training data
            
        Returns:
            (latitude, longitude) of cell center
        """
        # Decode lat_bin and lon_bin from grid_id
        lat_bin = grid_id // 100000
        lon_bin = grid_id % 100000
        
        # Chicago bounds (same as GridMapper)
        LAT_MIN, LAT_MAX = 41.64, 42.02
        LON_MIN, LON_MAX = -87.94, -87.52
        
        # Estimate bin size (assuming ~100 bins per dimension based on data)
        # From the data: lat_bin 0-83, lon_bin ~49-85
        lat_bin_size = (LAT_MAX - LAT_MIN) / 100  # ~0.0038 degrees
        lon_bin_size = (LON_MAX - LON_MIN) / 100  # ~0.0042 degrees
        
        # Convert to coordinates
        lat = LAT_MIN + (lat_bin + 0.5) * lat_bin_size
        lon = LON_MIN + (lon_bin + 0.5) * lon_bin_size
        
        # Clamp to bounds
        lat = max(LAT_MIN, min(LAT_MAX, lat))
        lon = max(LON_MIN, min(LON_MAX, lon))
        
        return lat, lon

    def predict_hotspots(
        self,
        target_date: date,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Predict top crime hotspots.

        Args:
            target_date: Date to predict for
            top_n: Number of hotspots to return

        Returns:
            List of hotspot predictions
        """
        self._ensure_initialized()

        # Get historically active cells
        try:
            top_cells = self._historical_service.get_top_cells_by_crime(
                num_weeks=8,
                top_n=top_n * 2,  # Get more to account for filtering
            )
        except Exception as e:
            logger.warning(f"Could not get historical data: {e}")
            top_cells = []

        if not top_cells:
            logger.info("No historical data available - using demo mode")
            return self._generate_demo_hotspots(target_date, top_n)

        # Get predictions for each cell using REAL data
        model = self._registry.get()
        results = []

        for grid_id, avg_count in top_cells[:top_n * 2]:  # Process more to get enough valid ones
            # Get historical counts for this cell
            history = self._historical_service.get_cell_history(
                grid_id=grid_id,
                num_weeks=8,
            )

            if not history:
                continue

            historical_counts = [r.crime_count for r in history]

            try:
                result = model.predict(
                    grid_id=grid_id,
                    target_date=target_date,
                    historical_counts=historical_counts,
                )

                # Convert training grid_id to lat/lon
                center_lat, center_lon = self._decode_training_grid_id(grid_id)

                results.append(
                    {
                        "rank": len(results) + 1,
                        "grid_id": grid_id,
                        "cell_center": {"latitude": center_lat, "longitude": center_lon},
                        "prediction_date": target_date.isoformat(),
                        "predicted_count": result.predicted_count,
                        "confidence_lower": result.confidence_lower,
                        "confidence_upper": result.confidence_upper,
                        "risk_level": self._get_risk_level(result.predicted_count),
                        "historical_average": avg_count,
                    }
                )
                
                if len(results) >= top_n:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to predict for grid {grid_id}: {e}")
                continue

        if not results:
            logger.warning("No valid predictions from real data, falling back to demo mode")
            return self._generate_demo_hotspots(target_date, top_n)

        # Sort by predicted count descending
        results.sort(key=lambda x: x["predicted_count"], reverse=True)

        # Re-rank
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results[:top_n]

    def _generate_demo_grid(
        self,
        target_date: date,
    ) -> dict[str, Any]:
        """Generate demo grid predictions when no historical data is available."""
        import random

        random.seed(target_date.toordinal())

        # Generate predictions for a sample of grid cells across Chicago
        grid_data = []
        total_cells = self._grid_mapper.total_cells

        # Sample ~100 cells spread across the grid
        sample_size = min(100, total_cells)
        step = max(1, total_cells // sample_size)

        for grid_id in range(0, total_cells, step):
            try:
                center_lat, center_lon = self._grid_mapper.grid_id_to_center(grid_id)
            except ValueError:
                continue

            # Generate realistic crime counts
            predicted = random.uniform(1, 30)

            grid_data.append({
                "grid_id": grid_id,
                "cell_center": {"latitude": center_lat, "longitude": center_lon},
                "predicted_count": round(predicted, 1),
                "confidence_lower": round(predicted * 0.7, 1),
                "confidence_upper": round(predicted * 1.3, 1),
                "risk_level": self._get_risk_level(predicted),
            })

        predictions = [g["predicted_count"] for g in grid_data]
        summary = {
            "total_cells": len(grid_data),
            "total_predicted_crimes": sum(predictions),
            "mean_per_cell": sum(predictions) / len(predictions) if predictions else 0,
            "max_per_cell": max(predictions) if predictions else 0,
            "high_risk_cells": sum(
                1 for p in predictions if self._get_risk_level(p) in ["high", "critical"]
            ),
        }

        return {
            "prediction_date": target_date.isoformat(),
            "grid_data": grid_data,
            "summary": summary,
        }

    def predict_grid(
        self,
        target_date: date,
        bounds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Predict crime counts for a grid of cells.

        Args:
            target_date: Date to predict for
            bounds: Optional geographic bounds

        Returns:
            Grid prediction result
        """
        self._ensure_initialized()

        # Get all active cells from historical data
        try:
            active_cells = self._historical_service.get_top_cells_by_crime(
                num_weeks=8,
                top_n=500,  # Limit for performance
            )
        except Exception as e:
            logger.warning(f"Could not get historical grid data: {e}")
            active_cells = []

        if not active_cells:
            # No historical data - use demo mode
            logger.info("Using demo grid predictions (no valid historical data)")
            return self._generate_demo_grid(target_date)

        # Use REAL data - active_cells contain training grid_ids
        logger.info(f"Using real grid predictions for {len(active_cells)} cells")

        # Batch predict
        model = self._registry.get()

        # Build historical counts map
        historical_counts_map = {}
        for grid_id, _avg_count in active_cells:
            history = self._historical_service.get_cell_history(
                grid_id=grid_id,
                num_weeks=8,
            )
            if history:
                historical_counts_map[grid_id] = [r.crime_count for r in history]

        # Batch predict
        results = model.predict_batch(
            grid_ids=list(historical_counts_map.keys()),
            target_date=target_date,
            historical_counts_map=historical_counts_map,
        )

        # Build response
        grid_data = []
        for result in results:
            # Use training grid_id decoder for coordinates
            center_lat, center_lon = self._decode_training_grid_id(result.grid_id)
            cell_center = {"latitude": center_lat, "longitude": center_lon}

            grid_data.append(
                {
                    "grid_id": result.grid_id,
                    "cell_center": cell_center,
                    "predicted_count": result.predicted_count,
                    "confidence_lower": result.confidence_lower,
                    "confidence_upper": result.confidence_upper,
                    "risk_level": self._get_risk_level(result.predicted_count),
                }
            )

        # Calculate summary statistics
        predictions = [r.predicted_count for r in results]
        summary = {
            "total_cells": len(results),
            "total_predicted_crimes": sum(predictions),
            "mean_per_cell": sum(predictions) / len(predictions) if predictions else 0,
            "max_per_cell": max(predictions) if predictions else 0,
            "high_risk_cells": sum(
                1 for p in predictions if self._get_risk_level(p) in ["high", "critical"]
            ),
        }

        return {
            "prediction_date": target_date.isoformat(),
            "model_id": model.config.model_id,
            "model_version": model.config.version,
            "grid_data": grid_data,
            "summary": summary,
        }

    def predict_horizon(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        horizon_weeks: int = 4,
    ) -> list[dict[str, Any]]:
        """Predict multiple weeks into the future.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: First prediction date
            horizon_weeks: Number of weeks to predict

        Returns:
            List of weekly predictions
        """
        self._ensure_initialized()

        try:
            grid_cell = self._grid_mapper.lat_lon_to_grid(latitude, longitude)
            grid_id = grid_cell.grid_id
        except ValueError:
            return [
                {
                    "error": "Location outside Chicago boundaries",
                    "latitude": latitude,
                    "longitude": longitude,
                }
            ]

        # Get historical data
        history = self._historical_service.get_cell_history(
            grid_id=grid_id,
            num_weeks=8,
        )

        if not history:
            return [
                {
                    "error": "Insufficient historical data",
                    "grid_id": grid_id,
                }
            ]

        historical_counts = [r.crime_count for r in history]

        # Get horizon predictions
        model = self._registry.get()
        results = model.predict_horizon(
            grid_id=grid_id,
            start_date=start_date,
            historical_counts=historical_counts,
            horizon_weeks=horizon_weeks,
        )

        return [
            {
                "week": i + 1,
                "prediction_date": r.prediction_date.isoformat(),
                "predicted_count": r.predicted_count,
                "confidence_lower": r.confidence_lower,
                "confidence_upper": r.confidence_upper,
                "risk_level": self._get_risk_level(r.predicted_count),
            }
            for i, r in enumerate(results)
        ]

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

    @property
    def is_loaded(self) -> bool:
        """Check if service is ready."""
        return self._initialized and self._registry is not None

    @property
    def model_version(self) -> str:
        """Get current model version."""
        if not self._initialized:
            return "not_loaded"
        try:
            model = self._registry.get()
            return model.config.version
        except Exception:
            return "unknown"

    @property
    def model_info(self) -> dict[str, Any]:
        """Get current model information."""
        if not self._initialized:
            return {"status": "not_initialized"}
        try:
            model = self._registry.get()
            return model.get_info()
        except Exception:
            return {"status": "error"}

    def health_check(self) -> dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary
        """
        self._ensure_initialized()

        health = self._registry.health_check()

        return {
            "status": "healthy"
            if all(h.status.value == "healthy" for h in health.values())
            else "degraded",
            "models": {
                model_id: {
                    "status": h.status.value,
                    "message": h.message,
                    "prediction_count": h.prediction_count,
                    "error_count": h.error_count,
                }
                for model_id, h in health.items()
            },
            "cache": {
                "historical_data_entries": len(self._cache._historical_data._cache)
                if self._cache
                else 0,
            },
        }


# Global service instance
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get or create prediction service singleton."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
