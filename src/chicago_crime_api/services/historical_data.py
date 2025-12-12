"""Historical data service for querying crime history.

Provides access to historical crime counts per grid cell,
used to build lag features for prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from chicago_crime_api.services.cache import DataCache, get_data_cache

if TYPE_CHECKING:
    pass


@dataclass
class WeeklyCrimeRecord:
    """Crime count for a specific grid cell and week."""

    grid_id: int
    year: int
    week: int
    crime_count: int
    month: int | None = None


class HistoricalDataService:
    """Service for querying historical crime data.

    Loads data from processed features.parquet and provides
    efficient lookups for grid cell history.
    """

    def __init__(
        self,
        data_path: Path | None = None,
        cache: DataCache | None = None,
    ) -> None:
        """Initialize service with data path and cache.

        Args:
            data_path: Path to features.parquet (default: data/processed/features.parquet)
            cache: DataCache instance (default: singleton)
        """
        self._data_path = data_path or Path("data/processed/features.parquet")
        self._cache = cache or get_data_cache()
        self._df: pd.DataFrame | None = None

    def _ensure_data_loaded(self) -> pd.DataFrame:
        """Ensure historical data is loaded.

        Returns:
            Historical data DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if self._df is not None:
            return self._df

        # Try to get from cache first
        if self._cache.is_loaded:
            self._df = self._cache.historical_data
            if self._df is not None:
                return self._df

        # Load and cache
        self._df = self._cache.load_historical_data(self._data_path)
        return self._df

    def get_cell_history(
        self,
        grid_id: int,
        num_weeks: int = 8,
        use_cache: bool = True,
    ) -> list[WeeklyCrimeRecord]:
        """Get last N weeks of crime data for a specific cell.

        Args:
            grid_id: Grid cell identifier
            num_weeks: Number of weeks to retrieve (default: 8)
            use_cache: Whether to use cache (default: True)

        Returns:
            List of WeeklyCrimeRecord, most recent first
        """
        # Check cache
        if use_cache:
            cached = self._cache.get_cell_history(grid_id, num_weeks)
            if cached is not None:
                # Convert cached counts back to records
                return self._counts_to_records(grid_id, cached)

        df = self._ensure_data_loaded()

        # Filter by grid_id and sort by time (descending)
        cell_df = df[df["grid_id"] == grid_id].copy()

        if cell_df.empty:
            # No history for this cell - return zeros
            return self._create_empty_history(grid_id, num_weeks)

        # Sort by year and week descending
        cell_df = cell_df.sort_values(["year", "week_of_year"], ascending=[False, False])

        # Take last N weeks
        cell_df = cell_df.head(num_weeks)

        # Convert to records
        records = []
        for _, row in cell_df.iterrows():
            records.append(
                WeeklyCrimeRecord(
                    grid_id=int(row["grid_id"]),
                    year=int(row["year"]),
                    week=int(row["week_of_year"]),
                    crime_count=int(row["crime_count"]),
                    month=int(row["month"]) if "month" in row else None,
                )
            )

        # Cache the counts
        if use_cache:
            counts = [r.crime_count for r in records]
            self._cache.set_cell_history(grid_id, num_weeks, counts)

        return records

    def get_cell_history_counts(
        self,
        grid_id: int,
        num_weeks: int = 8,
        use_cache: bool = True,
    ) -> list[int]:
        """Get last N weeks of crime counts for a specific cell.

        Convenience method that returns just the counts.

        Args:
            grid_id: Grid cell identifier
            num_weeks: Number of weeks to retrieve
            use_cache: Whether to use cache

        Returns:
            List of crime counts, most recent first
        """
        # Check cache directly for counts
        if use_cache:
            cached = self._cache.get_cell_history(grid_id, num_weeks)
            if cached is not None:
                return cached

        records = self.get_cell_history(grid_id, num_weeks, use_cache=False)
        counts = [r.crime_count for r in records]

        # Pad with zeros if not enough history
        while len(counts) < num_weeks:
            counts.append(0)

        # Cache
        if use_cache:
            self._cache.set_cell_history(grid_id, num_weeks, counts)

        return counts

    def _counts_to_records(self, grid_id: int, counts: list[int]) -> list[WeeklyCrimeRecord]:
        """Convert cached counts back to records (without full metadata)."""
        return [
            WeeklyCrimeRecord(
                grid_id=grid_id,
                year=0,  # Unknown from cache
                week=0,
                crime_count=count,
            )
            for count in counts
        ]

    def _create_empty_history(self, grid_id: int, num_weeks: int) -> list[WeeklyCrimeRecord]:
        """Create empty history for cells with no data."""
        return [
            WeeklyCrimeRecord(
                grid_id=grid_id,
                year=0,
                week=0,
                crime_count=0,
            )
            for _ in range(num_weeks)
        ]

    def get_latest_week(self) -> tuple[int, int]:
        """Return (year, week) of most recent data available.

        Returns:
            Tuple of (year, week_of_year)
        """
        df = self._ensure_data_loaded()

        # Get max year first, then max week within that year
        max_year = int(df["year"].max())
        max_week = int(df[df["year"] == max_year]["week_of_year"].max())

        return max_year, max_week

    def get_date_range(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get the date range of available data.

        Returns:
            ((min_year, min_week), (max_year, max_week))
        """
        df = self._ensure_data_loaded()

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

        min_week = int(df[df["year"] == min_year]["week_of_year"].min())
        max_week = int(df[df["year"] == max_year]["week_of_year"].max())

        return (min_year, min_week), (max_year, max_week)

    def get_top_cells_by_crime(
        self,
        num_weeks: int = 4,
        top_n: int = 20,
    ) -> list[tuple[int, float]]:
        """Return top N grid cells by average recent crime count.

        Args:
            num_weeks: Number of recent weeks to average
            top_n: Number of top cells to return

        Returns:
            List of (grid_id, avg_crime_count) tuples
        """
        cache_key = f"top_cells:{num_weeks}:{top_n}"
        cached = self._cache.get_grid_stats(cache_key)
        if cached is not None:
            # Convert back from dict
            return [(int(k), v) for k, v in cached.items()]

        df = self._ensure_data_loaded()

        # Get most recent weeks
        latest_year, latest_week = self.get_latest_week()

        # Filter to recent weeks (simplified - could be more accurate)
        recent_df = df[(df["year"] == latest_year) & (df["week_of_year"] > latest_week - num_weeks)]

        if recent_df.empty:
            # Fall back to all data from latest year
            recent_df = df[df["year"] == latest_year]

        # Aggregate by grid_id
        cell_stats = (
            recent_df.groupby("grid_id")["crime_count"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )

        result = [(int(grid_id), float(avg)) for grid_id, avg in cell_stats.items()]

        # Cache as dict
        self._cache.set_grid_stats(cache_key, {str(g): a for g, a in result})

        return result

    def get_city_wide_stats(self, num_weeks: int = 4) -> dict[str, float]:
        """Return city-wide crime statistics for baseline.

        Args:
            num_weeks: Number of recent weeks to compute stats

        Returns:
            Dict with mean, std, total, etc.
        """
        cache_key = f"city_stats:{num_weeks}"
        cached = self._cache.get_grid_stats(cache_key)
        if cached is not None:
            return cached

        df = self._ensure_data_loaded()

        # Get most recent weeks
        latest_year, latest_week = self.get_latest_week()

        recent_df = df[(df["year"] == latest_year) & (df["week_of_year"] > latest_week - num_weeks)]

        if recent_df.empty:
            recent_df = df[df["year"] == latest_year]

        stats = {
            "mean_per_cell": float(recent_df["crime_count"].mean()),
            "std_per_cell": float(recent_df["crime_count"].std()),
            "total_weekly_avg": float(
                recent_df.groupby("week_of_year")["crime_count"].sum().mean()
            ),
            "active_cells": int(recent_df["grid_id"].nunique()),
            "total_cells": int(df["grid_id"].nunique()),
        }

        self._cache.set_grid_stats(cache_key, stats)

        return stats

    def get_cell_baseline(self, grid_id: int) -> float:
        """Get historical average crime count for a cell.

        Args:
            grid_id: Grid cell identifier

        Returns:
            Average weekly crime count
        """
        df = self._ensure_data_loaded()

        cell_df = df[df["grid_id"] == grid_id]
        if cell_df.empty:
            return 0.0

        return float(cell_df["crime_count"].mean())

    def week_to_date(self, year: int, week: int) -> date:
        """Convert year and week number to date.

        Args:
            year: Year
            week: ISO week number

        Returns:
            Date of the Monday of that week
        """
        return date.fromisocalendar(year, week, 1)

    def preload_top_cells(self, top_n: int = 100) -> int:
        """Preload cache for most active cells.

        Args:
            top_n: Number of top cells to preload

        Returns:
            Number of cells preloaded
        """
        top_cells = self.get_top_cells_by_crime(num_weeks=4, top_n=top_n)

        for grid_id, _ in top_cells:
            self.get_cell_history_counts(grid_id, num_weeks=8, use_cache=True)

        return len(top_cells)


# Module-level singleton
_historical_service: HistoricalDataService | None = None


def get_historical_data_service(
    data_path: Path | None = None,
) -> HistoricalDataService:
    """Get or create HistoricalDataService singleton.

    Args:
        data_path: Path to features.parquet

    Returns:
        HistoricalDataService instance
    """
    global _historical_service
    if _historical_service is None:
        _historical_service = HistoricalDataService(data_path=data_path)
    return _historical_service
