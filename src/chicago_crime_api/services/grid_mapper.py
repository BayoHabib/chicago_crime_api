"""Grid mapping service for coordinate to grid cell conversion.

Uses pyproj for CRS transformation from WGS84 (lat/lon) to
Illinois State Plane (EPSG:26971) for accurate distance-based grid cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyproj import Transformer


@dataclass(frozen=True)
class GridCell:
    """Represents a spatial grid cell (immutable for caching)."""

    grid_id: int
    center_lat: float
    center_lon: float
    row: int
    col: int


class GridMapper:
    """Maps coordinates to grid cells using CRS transformation.

    Uses EPSG:26971 (Illinois State Plane East) for accurate
    meter-based grid cell assignment.
    """

    # Chicago geographic bounds (WGS84)
    LAT_MIN = 41.64
    LAT_MAX = 42.02
    LON_MIN = -87.94
    LON_MAX = -87.52

    # Default grid size in meters
    DEFAULT_GRID_SIZE_M = 500.0

    def __init__(self, grid_size_m: float = DEFAULT_GRID_SIZE_M) -> None:
        """Initialize mapper with grid resolution.

        Args:
            grid_size_m: Grid cell size in meters (default: 500m)
        """
        self._grid_size_m = grid_size_m
        self._transformer_to_proj: Transformer | None = None
        self._transformer_to_wgs: Transformer | None = None

        # Compute grid bounds in projected coordinates
        self._init_grid_bounds()

    def _ensure_transformers(self) -> None:
        """Lazy initialization of pyproj transformers."""
        if self._transformer_to_proj is None:
            from pyproj import Transformer

            # WGS84 (lat/lon) -> Illinois State Plane East (meters)
            self._transformer_to_proj = Transformer.from_crs(
                "EPSG:4326", "EPSG:26971", always_xy=True
            )
            # Reverse transformation
            self._transformer_to_wgs = Transformer.from_crs(
                "EPSG:26971", "EPSG:4326", always_xy=True
            )

    def _init_grid_bounds(self) -> None:
        """Compute grid bounds in projected coordinates."""
        self._ensure_transformers()

        # Transform corner points
        assert self._transformer_to_proj is not None
        self._x_min, self._y_min = self._transformer_to_proj.transform(self.LON_MIN, self.LAT_MIN)
        self._x_max, self._y_max = self._transformer_to_proj.transform(self.LON_MAX, self.LAT_MAX)

        # Calculate grid dimensions
        self._n_cols = int(np.ceil((self._x_max - self._x_min) / self._grid_size_m))
        self._n_rows = int(np.ceil((self._y_max - self._y_min) / self._grid_size_m))

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) of the grid."""
        return (self._n_rows, self._n_cols)

    @property
    def total_cells(self) -> int:
        """Return total number of grid cells."""
        return self._n_rows * self._n_cols

    @property
    def grid_size_m(self) -> float:
        """Return grid cell size in meters."""
        return self._grid_size_m

    def _project_coords(self, lat: float, lon: float) -> tuple[float, float]:
        """Transform lat/lon to projected coordinates.

        Args:
            lat: Latitude (WGS84)
            lon: Longitude (WGS84)

        Returns:
            (x, y) in EPSG:26971 meters
        """
        self._ensure_transformers()
        assert self._transformer_to_proj is not None
        x, y = self._transformer_to_proj.transform(lon, lat)
        return x, y

    def _unproject_coords(self, x: float, y: float) -> tuple[float, float]:
        """Transform projected coordinates back to lat/lon.

        Args:
            x: X coordinate (EPSG:26971)
            y: Y coordinate (EPSG:26971)

        Returns:
            (lat, lon) in WGS84
        """
        self._ensure_transformers()
        assert self._transformer_to_wgs is not None
        lon, lat = self._transformer_to_wgs.transform(x, y)
        return lat, lon

    def _coords_to_row_col(self, x: float, y: float) -> tuple[int, int]:
        """Convert projected coordinates to grid row/col.

        Args:
            x: X coordinate (EPSG:26971)
            y: Y coordinate (EPSG:26971)

        Returns:
            (row, col) indices
        """
        col = int((x - self._x_min) / self._grid_size_m)
        row = int((y - self._y_min) / self._grid_size_m)

        # Clamp to valid range
        col = max(0, min(col, self._n_cols - 1))
        row = max(0, min(row, self._n_rows - 1))

        return row, col

    def _row_col_to_grid_id(self, row: int, col: int) -> int:
        """Convert row/col to grid_id."""
        return row * self._n_cols + col

    def _grid_id_to_row_col(self, grid_id: int) -> tuple[int, int]:
        """Convert grid_id to row/col."""
        row = grid_id // self._n_cols
        col = grid_id % self._n_cols
        return row, col

    def _get_cell_center_projected(self, row: int, col: int) -> tuple[float, float]:
        """Get center of grid cell in projected coordinates."""
        x = self._x_min + (col + 0.5) * self._grid_size_m
        y = self._y_min + (row + 0.5) * self._grid_size_m
        return x, y

    @lru_cache(maxsize=10000)  # noqa: B019 - OK since GridMapper is singleton
    def lat_lon_to_grid(self, lat: float, lon: float) -> GridCell:
        """Convert lat/lon to grid cell with center coordinates.

        Args:
            lat: Latitude (WGS84)
            lon: Longitude (WGS84)

        Returns:
            GridCell with grid_id and center coordinates

        Raises:
            ValueError: If coordinates are outside Chicago bounds
        """
        # Validate bounds
        if not (self.LAT_MIN <= lat <= self.LAT_MAX):
            raise ValueError(
                f"Latitude {lat} outside Chicago bounds [{self.LAT_MIN}, {self.LAT_MAX}]"
            )
        if not (self.LON_MIN <= lon <= self.LON_MAX):
            raise ValueError(
                f"Longitude {lon} outside Chicago bounds [{self.LON_MIN}, {self.LON_MAX}]"
            )

        # Project and compute grid position
        x, y = self._project_coords(lat, lon)
        row, col = self._coords_to_row_col(x, y)
        grid_id = self._row_col_to_grid_id(row, col)

        # Get cell center
        center_x, center_y = self._get_cell_center_projected(row, col)
        center_lat, center_lon = self._unproject_coords(center_x, center_y)

        return GridCell(
            grid_id=grid_id,
            center_lat=round(center_lat, 6),
            center_lon=round(center_lon, 6),
            row=row,
            col=col,
        )

    @lru_cache(maxsize=5000)  # noqa: B019 - OK since GridMapper is singleton
    def grid_id_to_center(self, grid_id: int) -> tuple[float, float]:
        """Get lat/lon center of a grid cell.

        Args:
            grid_id: Grid cell identifier

        Returns:
            (lat, lon) of cell center

        Raises:
            ValueError: If grid_id is invalid
        """
        if not (0 <= grid_id < self.total_cells):
            raise ValueError(f"Invalid grid_id {grid_id}. Must be in [0, {self.total_cells})")

        row, col = self._grid_id_to_row_col(grid_id)
        center_x, center_y = self._get_cell_center_projected(row, col)
        center_lat, center_lon = self._unproject_coords(center_x, center_y)

        return round(center_lat, 6), round(center_lon, 6)

    def get_all_grid_ids(self) -> list[int]:
        """Return list of all valid grid IDs.

        Returns:
            List of grid IDs from 0 to total_cells-1
        """
        return list(range(self.total_cells))

    def get_neighboring_cells(self, grid_id: int, radius: int = 1) -> list[int]:
        """Get grid IDs of neighboring cells.

        Args:
            grid_id: Center cell
            radius: Number of cells in each direction (default: 1 = 8 neighbors)

        Returns:
            List of neighboring grid IDs (including center)
        """
        row, col = self._grid_id_to_row_col(grid_id)
        neighbors = []

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                new_row = row + dr
                new_col = col + dc
                if 0 <= new_row < self._n_rows and 0 <= new_col < self._n_cols:
                    neighbors.append(self._row_col_to_grid_id(new_row, new_col))

        return neighbors

    def clear_cache(self) -> None:
        """Clear LRU caches."""
        self.lat_lon_to_grid.cache_clear()
        self.grid_id_to_center.cache_clear()

    def get_cache_info(self) -> dict[str, dict[str, int]]:
        """Return cache statistics."""
        lat_lon_info = self.lat_lon_to_grid.cache_info()
        grid_id_info = self.grid_id_to_center.cache_info()

        return {
            "lat_lon_to_grid": {
                "hits": lat_lon_info.hits,
                "misses": lat_lon_info.misses,
                "size": lat_lon_info.currsize,
                "maxsize": lat_lon_info.maxsize or 0,
            },
            "grid_id_to_center": {
                "hits": grid_id_info.hits,
                "misses": grid_id_info.misses,
                "size": grid_id_info.currsize,
                "maxsize": grid_id_info.maxsize or 0,
            },
        }

    def is_valid_location(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Chicago bounds.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if within bounds
        """
        return self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX


# Module-level singleton instance
_grid_mapper: GridMapper | None = None


def get_grid_mapper(grid_size_m: float = GridMapper.DEFAULT_GRID_SIZE_M) -> GridMapper:
    """Get or create GridMapper singleton.

    Args:
        grid_size_m: Grid cell size in meters

    Returns:
        GridMapper instance
    """
    global _grid_mapper
    if _grid_mapper is None or _grid_mapper.grid_size_m != grid_size_m:
        _grid_mapper = GridMapper(grid_size_m=grid_size_m)
    return _grid_mapper
