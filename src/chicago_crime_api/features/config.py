"""Feature configuration - single source of truth for feature definitions.

This module defines all feature sets and their versions. When features change,
update this module and increment the version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureType(Enum):
    """Types of features for categorization."""

    LAG = "lag"
    ROLLING = "rolling"
    TREND = "trend"
    SEASONAL = "seasonal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""

    name: str
    feature_type: FeatureType
    description: str
    required_history_weeks: int = 1
    default_value: float = 0.0
    compute_fn: Callable[..., float] | None = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureDefinition):
            return self.name == other.name
        return False


@dataclass
class FeatureSet:
    """A versioned collection of features."""

    version: str
    features: list[FeatureDefinition]
    description: str = ""

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return [f.name for f in self.features]

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.features)

    @property
    def required_history_weeks(self) -> int:
        """Get minimum history required for all features."""
        return max(f.required_history_weeks for f in self.features)

    def get_feature(self, name: str) -> FeatureDefinition | None:
        """Get a feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "version": self.version,
            "features": self.feature_names,
            "n_features": self.n_features,
            "required_history_weeks": self.required_history_weeks,
        }

    @classmethod
    def from_feature_names(
        cls, version: str, feature_names: list[str], description: str = ""
    ) -> FeatureSet:
        """Create a FeatureSet from just feature names (for loading from metadata)."""
        features = [
            FeatureDefinition(
                name=name,
                feature_type=FeatureType.LAG,  # Default type
                description=f"Feature: {name}",
            )
            for name in feature_names
        ]
        return cls(version=version, features=features, description=description)


# =============================================================================
# FEATURE SET DEFINITIONS
# =============================================================================

# Version 1: 14-feature model (current production)
FEATURE_SET_V1 = FeatureSet(
    version="1.0.0",
    description="14-feature model with lag, rolling, trend, and seasonal features",
    features=[
        # Lag features (autoregressive - most important)
        FeatureDefinition(
            name="crime_count_lag1",
            feature_type=FeatureType.LAG,
            description="Crime count from 1 week ago",
            required_history_weeks=1,
        ),
        FeatureDefinition(
            name="crime_count_lag2",
            feature_type=FeatureType.LAG,
            description="Crime count from 2 weeks ago",
            required_history_weeks=2,
        ),
        FeatureDefinition(
            name="crime_count_lag3",
            feature_type=FeatureType.LAG,
            description="Crime count from 3 weeks ago",
            required_history_weeks=3,
        ),
        FeatureDefinition(
            name="crime_count_lag4",
            feature_type=FeatureType.LAG,
            description="Crime count from 4 weeks ago",
            required_history_weeks=4,
        ),
        # Rolling statistics
        FeatureDefinition(
            name="crime_count_rolling_mean_4",
            feature_type=FeatureType.ROLLING,
            description="4-week rolling mean of crime counts",
            required_history_weeks=4,
        ),
        FeatureDefinition(
            name="crime_count_rolling_std_4",
            feature_type=FeatureType.ROLLING,
            description="4-week rolling standard deviation",
            required_history_weeks=4,
        ),
        FeatureDefinition(
            name="crime_count_rolling_mean_8",
            feature_type=FeatureType.ROLLING,
            description="8-week rolling mean of crime counts",
            required_history_weeks=8,
        ),
        # Trend
        FeatureDefinition(
            name="crime_trend",
            feature_type=FeatureType.TREND,
            description="Difference between lag1 and rolling_mean_4",
            required_history_weeks=4,
        ),
        # Seasonality (Fourier encoding)
        FeatureDefinition(
            name="week_sin",
            feature_type=FeatureType.SEASONAL,
            description="Sine of week (first harmonic)",
            required_history_weeks=0,
        ),
        FeatureDefinition(
            name="week_cos",
            feature_type=FeatureType.SEASONAL,
            description="Cosine of week (first harmonic)",
            required_history_weeks=0,
        ),
        FeatureDefinition(
            name="week_sin2",
            feature_type=FeatureType.SEASONAL,
            description="Sine of week (second harmonic)",
            required_history_weeks=0,
        ),
        FeatureDefinition(
            name="week_cos2",
            feature_type=FeatureType.SEASONAL,
            description="Cosine of week (second harmonic)",
            required_history_weeks=0,
        ),
        # Time context
        FeatureDefinition(
            name="month",
            feature_type=FeatureType.TEMPORAL,
            description="Month number (1-12)",
            required_history_weeks=0,
        ),
        FeatureDefinition(
            name="is_weekend_ratio",
            feature_type=FeatureType.TEMPORAL,
            description="Ratio of weekend crimes",
            required_history_weeks=0,
            default_value=2.0 / 7.0,
        ),
    ],
)

# Legacy 9-feature model (kept for backwards compatibility)
FEATURE_SET_V0 = FeatureSet(
    version="0.1.0",
    description="Legacy 9-feature model",
    features=[
        FeatureDefinition(
            name="lat_bin",
            feature_type=FeatureType.SPATIAL,
            description="Latitude bin index",
        ),
        FeatureDefinition(
            name="lon_bin",
            feature_type=FeatureType.SPATIAL,
            description="Longitude bin index",
        ),
        FeatureDefinition(
            name="week_of_year",
            feature_type=FeatureType.TEMPORAL,
            description="Week of year (1-52)",
        ),
        FeatureDefinition(
            name="hour_mean",
            feature_type=FeatureType.TEMPORAL,
            description="Mean hour of crimes",
            default_value=12.0,
        ),
        FeatureDefinition(
            name="is_weekend_ratio",
            feature_type=FeatureType.TEMPORAL,
            description="Weekend ratio",
            default_value=2.0 / 7.0,
        ),
        FeatureDefinition(
            name="crime_count_lag1",
            feature_type=FeatureType.LAG,
            description="Crime count from 1 week ago",
            required_history_weeks=1,
        ),
        FeatureDefinition(
            name="crime_count_lag2",
            feature_type=FeatureType.LAG,
            description="Crime count from 2 weeks ago",
            required_history_weeks=2,
        ),
        FeatureDefinition(
            name="crime_count_lag4",
            feature_type=FeatureType.LAG,
            description="Crime count from 4 weeks ago",
            required_history_weeks=4,
        ),
        FeatureDefinition(
            name="crime_count_rolling_mean_4",
            feature_type=FeatureType.ROLLING,
            description="4-week rolling mean",
            required_history_weeks=4,
        ),
    ],
)

# Registry of all feature sets
FEATURE_SETS: dict[str, FeatureSet] = {
    "0.1.0": FEATURE_SET_V0,
    "1.0.0": FEATURE_SET_V1,
}

# Current default version
CURRENT_FEATURE_VERSION = "1.0.0"


@dataclass
class FeatureConfig:
    """Configuration for feature building.

    This class provides the interface between feature definitions
    and the feature builder.
    """

    feature_set: FeatureSet
    _feature_index: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Build feature index for fast lookup."""
        self._feature_index = {name: idx for idx, name in enumerate(self.feature_set.feature_names)}

    @property
    def version(self) -> str:
        """Get feature set version."""
        return self.feature_set.version

    @property
    def feature_names(self) -> list[str]:
        """Get ordered feature names."""
        return self.feature_set.feature_names

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return self.feature_set.n_features

    @property
    def required_history_weeks(self) -> int:
        """Get minimum history required."""
        return self.feature_set.required_history_weeks

    def get_feature_index(self, name: str) -> int | None:
        """Get index of a feature by name."""
        return self._feature_index.get(name)

    def has_feature(self, name: str) -> bool:
        """Check if feature exists in this config."""
        return name in self._feature_index

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration."""
        return self.feature_set.to_dict()

    @classmethod
    def from_version(cls, version: str) -> FeatureConfig:
        """Create config from a known version."""
        if version not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature version: {version}. Available: {list(FEATURE_SETS.keys())}"
            )
        return cls(feature_set=FEATURE_SETS[version])

    @classmethod
    def from_feature_names(
        cls, feature_names: list[str], version: str = "unknown"
    ) -> FeatureConfig:
        """Create config from a list of feature names.

        Used when loading a model with unknown/custom features.
        """
        feature_set = FeatureSet.from_feature_names(
            version=version,
            feature_names=feature_names,
            description=f"Custom feature set from model (n={len(feature_names)})",
        )
        return cls(feature_set=feature_set)

    @classmethod
    def from_model_metadata(cls, metadata: dict[str, Any]) -> FeatureConfig:
        """Create config from model metadata.

        Args:
            metadata: Model metadata dict containing 'features' or 'feature_version'

        Returns:
            FeatureConfig matching the model's expected features
        """
        # Try to get explicit feature list
        if "features" in metadata:
            features = metadata["features"]
            version = metadata.get("feature_version", "unknown")
            return cls.from_feature_names(features, version=version)

        # Try to get version
        if "feature_version" in metadata:
            version = metadata["feature_version"]
            if version in FEATURE_SETS:
                return cls.from_version(version)

        # Fall back to current version
        return cls.from_version(CURRENT_FEATURE_VERSION)


def get_feature_config(version: str | None = None) -> FeatureConfig:
    """Get feature configuration.

    Args:
        version: Feature set version, or None for current default

    Returns:
        FeatureConfig for the requested version
    """
    if version is None:
        version = CURRENT_FEATURE_VERSION
    return FeatureConfig.from_version(version)
