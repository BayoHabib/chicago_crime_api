"""Caching services for crime prediction API.

Provides TTL-based caching for historical data, predictions, and grid lookups.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Thread-safe TTL cache for arbitrary data."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """Initialize cache with TTL in seconds.

        Args:
            ttl_seconds: Time-to-live for cached entries (default: 1 hour)
        """
        self._ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[T, datetime]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        """Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]
            if datetime.now() - timestamp > self._ttl:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        """Set value with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = (value, datetime.now())

    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear entire cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items() if now - timestamp > self._ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    @property
    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
            }


class DataCache:
    """Centralized cache for historical crime data and predictions.

    Singleton pattern ensures consistent caching across the application.
    """

    _instance: DataCache | None = None
    _lock = threading.Lock()

    def __new__(cls) -> DataCache:
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize cache stores."""
        if self._initialized:
            return

        self._historical_df: pd.DataFrame | None = None
        self._last_load: datetime | None = None
        self._data_path: Path | None = None

        # TTL caches for different data types
        self._cell_history_cache: TTLCache[list[int]] = TTLCache(ttl_seconds=3600)
        self._prediction_cache: TTLCache[dict[str, Any]] = TTLCache(ttl_seconds=3600)
        self._grid_stats_cache: TTLCache[dict[str, float]] = TTLCache(ttl_seconds=3600)

        self._initialized = True

    @classmethod
    def get_instance(cls) -> DataCache:
        """Get singleton instance.

        Returns:
            DataCache singleton instance
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
                cls._instance = None

    def load_historical_data(
        self,
        data_path: Path,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Load and cache historical data from parquet.

        Args:
            data_path: Path to features.parquet file
            force_reload: Force reload even if already cached

        Returns:
            Historical data DataFrame
        """
        if not force_reload and self._historical_df is not None and self._data_path == data_path:
            return self._historical_df

        if not data_path.exists():
            raise FileNotFoundError(f"Historical data not found: {data_path}")

        self._historical_df = pd.read_parquet(data_path)
        self._last_load = datetime.now()
        self._data_path = data_path

        # Clear dependent caches on reload
        self._cell_history_cache.clear()
        self._grid_stats_cache.clear()

        return self._historical_df

    @property
    def historical_data(self) -> pd.DataFrame | None:
        """Get cached historical data (may be None if not loaded)."""
        return self._historical_df

    @property
    def is_loaded(self) -> bool:
        """Check if historical data is loaded."""
        return self._historical_df is not None

    @property
    def last_load_time(self) -> datetime | None:
        """Get timestamp of last data load."""
        return self._last_load

    def refresh_if_stale(self, max_age_hours: int = 24) -> bool:
        """Reload data if older than max_age_hours.

        Args:
            max_age_hours: Maximum age before refresh

        Returns:
            True if data was refreshed
        """
        if self._last_load is None or self._data_path is None:
            return False

        age = datetime.now() - self._last_load
        if age > timedelta(hours=max_age_hours):
            self.load_historical_data(self._data_path, force_reload=True)
            return True
        return False

    # Cell history cache methods
    def get_cell_history(self, grid_id: int, num_weeks: int = 8) -> list[int] | None:
        """Get cached cell history."""
        key = f"{grid_id}:{num_weeks}"
        return self._cell_history_cache.get(key)

    def set_cell_history(self, grid_id: int, num_weeks: int, history: list[int]) -> None:
        """Cache cell history."""
        key = f"{grid_id}:{num_weeks}"
        self._cell_history_cache.set(key, history)

    # Prediction cache methods
    def get_prediction(self, cache_key: str) -> dict[str, Any] | None:
        """Get cached prediction result."""
        return self._prediction_cache.get(cache_key)

    def set_prediction(self, cache_key: str, result: dict[str, Any]) -> None:
        """Cache prediction result."""
        self._prediction_cache.set(cache_key, result)

    # Grid stats cache methods
    def get_grid_stats(self, key: str) -> dict[str, float] | None:
        """Get cached grid statistics."""
        return self._grid_stats_cache.get(key)

    def set_grid_stats(self, key: str, stats: dict[str, float]) -> None:
        """Cache grid statistics."""
        self._grid_stats_cache.set(key, stats)

    def get_cache_stats(self) -> dict[str, Any]:
        """Return comprehensive cache statistics."""
        return {
            "historical_data": {
                "loaded": self.is_loaded,
                "rows": len(self._historical_df) if self._historical_df is not None else 0,
                "last_load": self._last_load.isoformat() if self._last_load else None,
            },
            "cell_history": self._cell_history_cache.stats,
            "predictions": self._prediction_cache.stats,
            "grid_stats": self._grid_stats_cache.stats,
        }

    def clear_all(self) -> dict[str, int]:
        """Clear all caches.

        Returns:
            Count of cleared entries per cache
        """
        return {
            "cell_history": self._cell_history_cache.clear(),
            "predictions": self._prediction_cache.clear(),
            "grid_stats": self._grid_stats_cache.clear(),
        }


def get_data_cache() -> DataCache:
    """Dependency injection helper for FastAPI.

    Returns:
        DataCache singleton instance
    """
    return DataCache.get_instance()
