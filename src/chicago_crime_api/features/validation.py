"""Feature validation utilities.

Validates feature compatibility between models and feature builders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of feature validation."""

    is_valid: bool
    missing_features: list[str]
    extra_features: list[str]
    message: str

    @property
    def has_missing(self) -> bool:
        """Check if there are missing features."""
        return len(self.missing_features) > 0

    @property
    def has_extra(self) -> bool:
        """Check if there are extra features."""
        return len(self.extra_features) > 0


class FeatureValidator:
    """Validates feature compatibility."""

    def __init__(self, strict: bool = True) -> None:
        """Initialize validator.

        Args:
            strict: If True, extra features also cause validation failure.
                   If False, only missing features cause failure.
        """
        self.strict = strict

    def validate(
        self,
        expected: list[str],
        actual: list[str],
    ) -> ValidationResult:
        """Validate actual features against expected.

        Args:
            expected: Features the model expects
            actual: Features being provided

        Returns:
            ValidationResult with details
        """
        expected_set = set(expected)
        actual_set = set(actual)

        missing = list(expected_set - actual_set)
        extra = list(actual_set - expected_set)

        # Check validity
        if self.strict:
            is_valid = len(missing) == 0 and len(extra) == 0
        else:
            is_valid = len(missing) == 0

        # Build message
        if is_valid:
            message = f"Feature validation passed ({len(expected)} features)"
        else:
            parts = []
            if missing:
                parts.append(f"Missing: {missing}")
            if extra and self.strict:
                parts.append(f"Extra: {extra}")
            message = "Feature mismatch - " + "; ".join(parts)

        return ValidationResult(
            is_valid=is_valid,
            missing_features=missing,
            extra_features=extra,
            message=message,
        )

    def validate_configs(
        self,
        model_config: FeatureConfig,
        builder_config: FeatureConfig,
    ) -> ValidationResult:
        """Validate feature configs match.

        Args:
            model_config: Config from model metadata
            builder_config: Config from feature builder

        Returns:
            ValidationResult
        """
        return self.validate(
            expected=model_config.feature_names,
            actual=builder_config.feature_names,
        )


def validate_features(
    expected: list[str],
    actual: list[str],
    strict: bool = True,
) -> ValidationResult:
    """Convenience function to validate features.

    Args:
        expected: Features the model expects
        actual: Features being provided
        strict: Whether extra features also fail validation

    Returns:
        ValidationResult
    """
    validator = FeatureValidator(strict=strict)
    return validator.validate(expected, actual)


class FeatureMismatchError(Exception):
    """Raised when features don't match between model and builder."""

    def __init__(
        self,
        message: str,
        validation_result: ValidationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_result = validation_result


def ensure_feature_compatibility(
    model_features: list[str],
    builder_features: list[str],
    raise_on_mismatch: bool = True,
) -> ValidationResult:
    """Ensure model and builder features are compatible.

    Args:
        model_features: Features the model expects
        builder_features: Features the builder provides
        raise_on_mismatch: If True, raise FeatureMismatchError on mismatch

    Returns:
        ValidationResult

    Raises:
        FeatureMismatchError: If features don't match and raise_on_mismatch=True
    """
    result = validate_features(
        expected=model_features,
        actual=builder_features,
        strict=True,
    )

    if not result.is_valid:
        logger.error(f"Feature mismatch: {result.message}")
        logger.error(f"  Model expects: {model_features}")
        logger.error(f"  Builder provides: {builder_features}")

        if raise_on_mismatch:
            raise FeatureMismatchError(
                f"Feature mismatch between model and builder: {result.message}",
                validation_result=result,
            )

    return result
