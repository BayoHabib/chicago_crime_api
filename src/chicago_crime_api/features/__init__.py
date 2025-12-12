"""Feature configuration and validation module.

This module provides a single source of truth for feature definitions
and handles feature versioning/compatibility between training and inference.
"""

from .config import (
    CURRENT_FEATURE_VERSION,
    FeatureConfig,
    FeatureSet,
    get_feature_config,
)
from .validation import FeatureValidator, validate_features

__all__ = [
    "FeatureConfig",
    "FeatureSet",
    "FeatureValidator",
    "CURRENT_FEATURE_VERSION",
    "get_feature_config",
    "validate_features",
]
