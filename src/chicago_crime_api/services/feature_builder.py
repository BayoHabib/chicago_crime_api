"""Feature builder service for constructing model input features.

This module builds features dynamically based on a FeatureConfig,
allowing the system to adapt when features are added or removed.

The default feature set (v1.0.0) includes 14 features:
- Lag features: crime_count_lag1-4
- Rolling stats: rolling_mean_4, rolling_std_4, rolling_mean_8
- Trend: crime_trend
- Seasonality: week_sin, week_cos, week_sin2, week_cos2
- Time context: month, is_weekend_ratio
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..features.config import (
    CURRENT_FEATURE_VERSION,
    FeatureConfig,
    get_feature_config,
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds model features from historical crime data.

    Features are built according to a FeatureConfig, which can be:
    - Loaded from a known version (default)
    - Derived from model metadata
    - Custom feature set

    This allows the builder to adapt when features change.
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        feature_version: str | None = None,
    ) -> None:
        """Initialize feature builder.

        Args:
            config: Explicit feature configuration (takes precedence)
            feature_version: Version string to load config from
        """
        if config is not None:
            self._config = config
        elif feature_version is not None:
            self._config = get_feature_config(feature_version)
        else:
            self._config = get_feature_config(CURRENT_FEATURE_VERSION)

        logger.debug(
            f"FeatureBuilder initialized with {self._config.n_features} features "
            f"(version {self._config.version})"
        )

    @property
    def config(self) -> FeatureConfig:
        """Get the feature configuration."""
        return self._config

    @property
    def feature_columns(self) -> list[str]:
        """Return list of feature column names."""
        return self._config.feature_names

    # Backwards compatibility alias
    FEATURE_COLUMNS = property(lambda self: self._config.feature_names)

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return self._config.n_features

    @property
    def required_history_weeks(self) -> int:
        """Return minimum history weeks required."""
        return self._config.required_history_weeks

    @classmethod
    def from_model_metadata(cls, metadata: dict[str, Any]) -> FeatureBuilder:
        """Create a FeatureBuilder matching a model's expected features.

        Args:
            metadata: Model metadata containing feature information

        Returns:
            FeatureBuilder configured for the model's features
        """
        config = FeatureConfig.from_model_metadata(metadata)
        return cls(config=config)

    def _compute_fourier_features(self, week_of_year: int) -> dict[str, float]:
        """Compute Fourier encoding for week of year.

        Args:
            week_of_year: Week number (1-52)

        Returns:
            Dict with week_sin, week_cos, week_sin2, week_cos2
        """
        # First harmonic (annual cycle)
        angle1 = week_of_year * 2 * math.pi / 52
        # Second harmonic (semi-annual cycle)
        angle2 = week_of_year * 4 * math.pi / 52

        return {
            "week_sin": math.sin(angle1),
            "week_cos": math.cos(angle1),
            "week_sin2": math.sin(angle2),
            "week_cos2": math.cos(angle2),
        }

    def build_features(
        self,
        historical_counts: list[int],
        prediction_date: date,
        latitude: float | None = None,
        longitude: float | None = None,
        grid_id: int | None = None,
    ) -> pd.DataFrame:
        """Build feature DataFrame for model prediction.

        Only computes features that are in the current config.

        Args:
            historical_counts: Last 8+ weeks of crime counts, most recent first
                              [week-1, week-2, week-3, ..., week-8, ...]
            prediction_date: Date for which to predict
            latitude: Optional latitude (for spatial features)
            longitude: Optional longitude (for spatial features)
            grid_id: Optional grid ID (for spatial features)

        Returns:
            DataFrame with one row containing features in config order
        """
        # Compute all possible features
        all_features = self._compute_all_features(
            historical_counts=historical_counts,
            prediction_date=prediction_date,
            latitude=latitude,
            longitude=longitude,
            grid_id=grid_id,
        )

        # Filter to only features in config, maintaining order
        features = {name: all_features.get(name, 0.0) for name in self._config.feature_names}

        # Warn if any requested features weren't computed
        missing = [n for n in self._config.feature_names if n not in all_features]
        if missing:
            logger.warning(f"Features not computed (using default 0.0): {missing}")

        return pd.DataFrame([features])

    def _compute_all_features(
        self,
        historical_counts: list[int],
        prediction_date: date,
        latitude: float | None = None,
        longitude: float | None = None,
        grid_id: int | None = None,
    ) -> dict[str, float]:
        """Compute all possible features.

        This method computes every feature the builder knows about.
        The caller can then filter to only the needed features.

        Returns:
            Dict mapping feature name to computed value
        """
        # Ensure we have enough history (pad with zeros if needed)
        counts = list(historical_counts)
        min_history = max(8, self._config.required_history_weeks)
        while len(counts) < min_history:
            counts.append(0)

        features: dict[str, float] = {}

        # Lag features
        for i in range(1, min(len(counts) + 1, 9)):
            features[f"crime_count_lag{i}"] = float(counts[i - 1])

        # Rolling statistics
        features["crime_count_rolling_mean_4"] = float(np.mean(counts[:4]))
        features["crime_count_rolling_std_4"] = float(np.std(counts[:4], ddof=0))
        features["crime_count_rolling_mean_8"] = float(np.mean(counts[:8]))

        # Trend
        features["crime_trend"] = (
            features["crime_count_lag1"] - features["crime_count_rolling_mean_4"]
        )

        # Temporal features
        week_of_year = prediction_date.isocalendar()[1]
        features["week_of_year"] = float(week_of_year)
        features["month"] = float(prediction_date.month)
        features["is_weekend_ratio"] = 2.0 / 7.0
        features["hour_mean"] = 12.0  # Default midday

        # Fourier encoding
        angle1 = week_of_year * 2 * math.pi / 52
        angle2 = week_of_year * 4 * math.pi / 52
        features["week_sin"] = math.sin(angle1)
        features["week_cos"] = math.cos(angle1)
        features["week_sin2"] = math.sin(angle2)
        features["week_cos2"] = math.cos(angle2)

        # Spatial features (for legacy v0 models)
        if latitude is not None and longitude is not None:
            features["lat_bin"] = self._compute_lat_bin(latitude)
            features["lon_bin"] = self._compute_lon_bin(longitude)
        elif grid_id is not None:
            features["lat_bin"] = float((grid_id // 100) % 20)
            features["lon_bin"] = float(grid_id % 20)
        else:
            features["lat_bin"] = 10.0
            features["lon_bin"] = 10.0

        return features

    def _compute_lat_bin(self, latitude: float) -> float:
        """Compute latitude bin index."""
        lat_min, lat_max, n_bins = 41.64, 42.02, 20
        bins = np.linspace(lat_min, lat_max, n_bins + 1)
        return float(np.clip(np.digitize(latitude, bins) - 1, 0, n_bins - 1))

    def _compute_lon_bin(self, longitude: float) -> float:
        """Compute longitude bin index."""
        lon_min, lon_max, n_bins = -87.95, -87.50, 20
        bins = np.linspace(lon_min, lon_max, n_bins + 1)
        return float(np.clip(np.digitize(longitude, bins) - 1, 0, n_bins - 1))

    def build_features_for_horizon(
        self,
        historical_counts: list[int],
        start_date: date,
        horizon_weeks: int = 4,
        latitude: float | None = None,
        longitude: float | None = None,
        grid_id: int | None = None,
    ) -> list[pd.DataFrame]:
        """Build features for multi-week prediction horizon.

        Args:
            historical_counts: Historical crime counts (most recent first)
                              Need at least 8 weeks for rolling_mean_8
            start_date: Start date for predictions
            horizon_weeks: Number of weeks to predict
            latitude: Optional latitude (not used in 14-feature model)
            longitude: Optional longitude (not used in 14-feature model)
            grid_id: Optional grid ID (not used in 14-feature model)

        Returns:
            List of DataFrames, one per prediction week
        """
        results = []
        current_counts = list(historical_counts)

        # Ensure we have enough history
        while len(current_counts) < 8:
            current_counts.append(0)

        for week_offset in range(horizon_weeks):
            prediction_date = start_date + timedelta(weeks=week_offset)

            features = self.build_features(
                historical_counts=current_counts,
                prediction_date=prediction_date,
                latitude=latitude,
                longitude=longitude,
                grid_id=grid_id,
            )
            results.append(features)

            # Shift counts for next iteration (simulated rolling)
            # Use the rolling mean as a proxy for next week's count
            next_estimate = int(np.mean(current_counts[:4]))
            current_counts = [next_estimate] + current_counts[:-1]

        return results

    def build_batch_features(
        self,
        grid_data: list[dict],
        prediction_date: date,
    ) -> pd.DataFrame:
        """Build features for multiple grid cells.

        Args:
            grid_data: List of dicts with keys:
                - historical_counts: list[int]
                - latitude: float
                - longitude: float
                - grid_id: int (optional)
            prediction_date: Prediction target date

        Returns:
            DataFrame with one row per grid cell
        """
        all_features = []

        for cell in grid_data:
            features = self.build_features(
                historical_counts=cell["historical_counts"],
                prediction_date=prediction_date,
                latitude=cell.get("latitude"),
                longitude=cell.get("longitude"),
                grid_id=cell.get("grid_id"),
            )
            all_features.append(features.iloc[0].to_dict())

        return pd.DataFrame(all_features)


# Module-level singleton
_feature_builder: FeatureBuilder | None = None


def get_feature_builder() -> FeatureBuilder:
    """Get the singleton feature builder instance."""
    global _feature_builder
    if _feature_builder is None:
        _feature_builder = FeatureBuilder()
    return _feature_builder
