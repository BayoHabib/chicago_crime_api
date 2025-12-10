"""Process raw crime data into features using EventFlow.

Uses EventFlow's spatial and temporal steps for feature engineering
with lazy Polars evaluation for efficiency.
"""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_data(database_path: Path, output_path: Path) -> None:
    """Process raw crime data into ML-ready features using EventFlow.

    Args:
        database_path: Path to DuckDB database with raw crime data
        output_path: Path to save processed features (Parquet)
    """
    eventflow_available = False
    try:
        from eventflow import EventFrame, EventMetadata, EventSchema  # noqa: F401

        eventflow_available = True
    except ImportError as e:
        logger.warning(f"EventFlow not fully available: {e}")

    # Load data from DuckDB
    logger.info(f"Loading data from {database_path}...")

    try:
        import duckdb

        conn = duckdb.connect(str(database_path), read_only=True)
        df = conn.execute("SELECT * FROM crimes").pl()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load from DuckDB: {e}")
        logger.info("Trying to load from Parquet files...")
        # Look for parquet files in data/raw directory
        raw_dir = database_path.parent / "raw"
        parquet_files = list(raw_dir.glob("**/*.parquet"))
        if not parquet_files:
            parquet_files = list(database_path.parent.glob("**/*.parquet"))
        if parquet_files:
            # Use diagonal concat to handle potential schema differences
            df = pl.concat([pl.read_parquet(f) for f in parquet_files], how="diagonal")
        else:
            raise RuntimeError("No data source available") from e

    logger.info(f"Loaded {len(df)} records")

    # Ensure date column is datetime
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Datetime).alias("timestamp"))
    elif "timestamp" not in df.columns:
        raise ValueError("No date/timestamp column found")

    # Ensure lat/lon are floats (may come as strings from some sources)
    if "latitude" in df.columns:
        df = df.with_columns(
            [
                pl.col("latitude").cast(pl.Float64),
                pl.col("longitude").cast(pl.Float64),
            ]
        )

    # Filter out null coordinates
    df = df.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

    # Create lazy frame for efficient processing
    lf = df.lazy()

    # Try EventFlow processing first, fall back if needed
    if eventflow_available:
        try:
            processed_lf = _process_with_eventflow(lf)
        except Exception as e:
            logger.warning(f"EventFlow processing failed: {e}, using fallback")
            processed_lf = _process_fallback(lf)
    else:
        processed_lf = _process_fallback(lf)

    # Collect and save
    result_df = processed_lf.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)

    logger.info(f"Saved {len(result_df)} processed records to {output_path}")
    _print_summary(result_df)


def _process_with_eventflow(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Process data using EventFlow pipeline with fast CRS transformation."""
    import numpy as np
    from eventflow import EventFrame, EventMetadata, EventSchema
    from eventflow.core.steps import (
        AssignToGridStep,
        ExtractTemporalComponentsStep,
    )
    from pyproj import Transformer

    logger.info("Processing with EventFlow...")

    # Step 1: Fast batch CRS transformation using pyproj (vectorized)
    # EventFlow's TransformCRSStep uses map_elements which is slow
    logger.info("Transforming coordinates to EPSG:26971 (Illinois State Plane)...")

    # Collect lat/lon for transformation (must be done in memory for pyproj)
    df = lf.collect()

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26971", always_xy=True)

    lons = df["longitude"].to_numpy().astype(np.float64)
    lats = df["latitude"].to_numpy().astype(np.float64)

    # Vectorized transformation (much faster than row-by-row)
    x_proj, y_proj = transformer.transform(lons, lats)

    # Add projected columns
    df = df.with_columns(
        [
            pl.Series("longitude_proj", x_proj),
            pl.Series("latitude_proj", y_proj),
        ]
    )

    logger.info(
        f"CRS transformation complete. X range: {x_proj.min():.0f} - {x_proj.max():.0f} meters"
    )

    # Convert back to lazy frame
    lf = df.lazy()

    # Define schema with projected coordinates
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude_proj",  # Use projected coordinates (in meters)
        lon_col="longitude_proj",
        categorical_cols=["primary_type", "location_description"],
        numeric_cols=[],
    )

    metadata = EventMetadata(
        dataset_name="chicago_crime",
        crs="EPSG:26971",  # Already transformed
        time_zone="America/Chicago",
    )

    # Create EventFrame with projected coordinates
    event_frame = EventFrame(lf, schema, metadata)

    # Step 2: Extract temporal components
    temporal_step = ExtractTemporalComponentsStep(
        components=["hour_of_day", "day_of_week", "month", "is_weekend"]
    )
    event_frame = temporal_step.run(event_frame)

    # Step 3: Assign to spatial grid (500m cells) - coordinates are already in meters
    grid_step = AssignToGridStep(grid_size_m=500.0)
    event_frame = grid_step.run(event_frame)

    # Get the processed lazy frame and add year/week columns needed for aggregation
    processed_lf = event_frame.lazy_frame.with_columns(
        [
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.week().alias("week_of_year"),
        ]
    )

    # Also need lat_bin/lon_bin for aggregation output
    # EventFlow uses grid_id directly, so derive bins from it
    processed_lf = processed_lf.with_columns(
        [
            (pl.col("grid_id") // 100000).alias("lat_bin"),
            (pl.col("grid_id") % 100000).alias("lon_bin"),
        ]
    )

    # Additional aggregations for weekly predictions
    processed_lf = _add_weekly_aggregations(processed_lf)

    return processed_lf


def _process_fallback(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fallback processing without EventFlow."""
    logger.info("Processing with fallback (no EventFlow)...")

    # Chicago geographic bounds
    lat_min, lat_max = 41.64, 42.02
    lon_min, lon_max = -87.94, -87.52
    grid_size = 50

    # Extract temporal features
    processed_lf = lf.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour_of_day"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.week().alias("week_of_year"),
            (pl.col("timestamp").dt.weekday() >= 5).cast(pl.Int32).alias("is_weekend"),
        ]
    )

    # Create spatial grid cells
    processed_lf = processed_lf.with_columns(
        [
            ((pl.col("latitude") - lat_min) / (lat_max - lat_min) * grid_size)
            .cast(pl.Int32)
            .clip(0, grid_size - 1)
            .alias("lat_bin"),
            ((pl.col("longitude") - lon_min) / (lon_max - lon_min) * grid_size)
            .cast(pl.Int32)
            .clip(0, grid_size - 1)
            .alias("lon_bin"),
        ]
    )

    processed_lf = processed_lf.with_columns(
        [
            (pl.col("lat_bin") * grid_size + pl.col("lon_bin")).alias("grid_id"),
        ]
    )

    # Add weekly aggregations
    processed_lf = _add_weekly_aggregations(processed_lf)

    return processed_lf


def _add_weekly_aggregations(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add weekly aggregation features for prediction."""
    # Aggregate: count crimes per cell per week
    weekly_lf = lf.group_by(["year", "week_of_year", "grid_id"]).agg(
        [
            pl.len().alias("crime_count"),
            pl.col("hour_of_day").mean().alias("hour_mean"),
            pl.col("is_weekend").mean().alias("is_weekend_ratio"),
            pl.col("lat_bin").first().alias("lat_bin"),
            pl.col("lon_bin").first().alias("lon_bin"),
            pl.col("month").first().alias("month"),
        ]
    )

    # Sort for lag calculations
    weekly_lf = weekly_lf.sort(["grid_id", "year", "week_of_year"])

    # Add lag features using over() for each grid cell
    weekly_lf = weekly_lf.with_columns(
        [
            pl.col("crime_count").shift(1).over("grid_id").alias("crime_count_lag1"),
            pl.col("crime_count").shift(2).over("grid_id").alias("crime_count_lag2"),
            pl.col("crime_count").shift(3).over("grid_id").alias("crime_count_lag3"),
            pl.col("crime_count").shift(4).over("grid_id").alias("crime_count_lag4"),
        ]
    )

    # Rolling statistics
    weekly_lf = weekly_lf.with_columns(
        [
            pl.col("crime_count")
            .rolling_mean(window_size=4)
            .over("grid_id")
            .alias("crime_count_rolling_mean_4"),
            pl.col("crime_count")
            .rolling_std(window_size=4)
            .over("grid_id")
            .alias("crime_count_rolling_std_4"),
            pl.col("crime_count")
            .rolling_mean(window_size=8)
            .over("grid_id")
            .alias("crime_count_rolling_mean_8"),
        ]
    )

    # Trend feature: difference from rolling mean
    weekly_lf = weekly_lf.with_columns(
        [
            (pl.col("crime_count_lag1") - pl.col("crime_count_rolling_mean_4")).alias(
                "crime_trend"
            ),
        ]
    )

    # Seasonality features (Fourier encoding for week_of_year)
    import math

    weekly_lf = weekly_lf.with_columns(
        [
            (pl.col("week_of_year") * 2 * math.pi / 52).sin().alias("week_sin"),
            (pl.col("week_of_year") * 2 * math.pi / 52).cos().alias("week_cos"),
            (pl.col("week_of_year") * 4 * math.pi / 52).sin().alias("week_sin2"),
            (pl.col("week_of_year") * 4 * math.pi / 52).cos().alias("week_cos2"),
        ]
    )

    # Drop rows with null lags (need at least 8 weeks of history)
    weekly_lf = weekly_lf.filter(
        pl.col("crime_count_lag4").is_not_null()
        & pl.col("crime_count_rolling_mean_8").is_not_null()
    )

    return weekly_lf


def _print_summary(df: pl.DataFrame) -> None:
    """Print data summary statistics."""
    logger.info("\nData Summary:")
    logger.info(f"  - Total records: {len(df)}")
    logger.info(f"  - Unique grid cells: {df['grid_id'].n_unique()}")
    stats = df.select(
        pl.col("crime_count").mean().alias("mean"),
        pl.col("crime_count").std().alias("std"),
    ).row(0)
    logger.info(f"  - Crime count mean: {stats[0]:.2f}")
    logger.info(f"  - Crime count std: {stats[1]:.2f}")
    logger.info(f"  - Features: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process crime data for ML")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/crimes.duckdb"),
        help="Path to DuckDB database with raw data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Output path for processed features",
    )

    args = parser.parse_args()
    process_data(args.database, args.output)
