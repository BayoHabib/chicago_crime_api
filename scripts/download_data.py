"""Download Chicago crime data using chicago-crime-downloader.

Uses the chicago-crime-downloader library to fetch data from Chicago Data Portal
and materialize it into a DuckDB database for efficient querying.
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_crime_data(
    output_dir: Path,
    database_path: Path,
    start_date: date,
    end_date: date,
    mode: str = "monthly",
    out_format: str = "parquet",
) -> None:
    """Download crime data from Chicago Data Portal.

    Uses chicago-crime-downloader with windowed downloads and DuckDB materialization.

    Args:
        output_dir: Directory to store raw chunk files
        database_path: Path to DuckDB database for materialized data
        start_date: Start date for data range
        end_date: End date for data range
        mode: Download mode ('daily', 'weekly', 'monthly')
        out_format: Output format ('csv', 'parquet')
    """
    try:
        from chicago_crime_downloader import (
            HttpConfig,
            RunConfig,
            day_windows,
            default_type_overrides,
            discover_chunks,
            headers_with_token,
            materialize_duckdb,
            month_windows,
            run_windowed_mode,
            setup_logging,
            week_windows,
        )
    except ImportError as e:
        logger.error(f"chicago-crime-downloader not available: {e}")
        raise

    # Setup logging
    setup_logging(log_file=str(output_dir / "download.log"), json_logs=False)

    # Configure HTTP client
    http_config = HttpConfig(
        timeout=300,
        retries=5,
        sleep=1.0,
    )

    # Configure download run
    run_config = RunConfig(
        mode=mode,
        out_root=output_dir,
        out_format=out_format,
        chunk_size=50000,
        max_chunks=None,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        select="id,date,primary_type,description,location_description,latitude,longitude,arrest,domestic,beat,district,ward,community_area",
        columns_file=None,
        compression="snappy" if out_format == "parquet" else None,
        layout="nested",
        preflight=True,
    )

    headers = headers_with_token(http_config)

    # Generate windows based on mode
    if mode == "monthly":
        windows = month_windows(start_date, end_date)
    elif mode == "weekly":
        windows = week_windows(start_date, end_date)
    else:
        windows = day_windows(start_date, end_date)

    logger.info(f"Downloading crime data from {start_date} to {end_date}...")
    logger.info(f"Mode: {mode}, Format: {out_format}, Windows: {len(windows)}")

    # Run windowed download
    try:
        run_windowed_mode(
            run_config,
            http_config,
            headers,
            run_config.select,
            windows,
            mode,
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

    # Materialize into DuckDB
    logger.info(f"Materializing data into DuckDB: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        chunks, manifests = discover_chunks(output_dir)
    except Exception as e:
        logger.error(f"Failed to discover chunks: {e}")
        raise

    if chunks:
        materialize_duckdb(
            files=chunks,
            manifests=manifests,
            database=database_path,
            table="crimes",
            manifest_table="chunk_manifests",
            replace=True,
            column_types=default_type_overrides(),
            all_varchar=False,
        )
        logger.info(f"Materialized {len(chunks)} chunks into {database_path}")
    else:
        logger.error("No chunks found to materialize. Check download logs.")
        raise RuntimeError("Download failed: no data chunks found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Chicago crime data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw chunk files",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/crimes.duckdb"),
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(date.today() - timedelta(days=365)).isoformat(),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["daily", "weekly", "monthly"],
        default="monthly",
        help="Download window mode",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format for chunks",
    )

    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    download_crime_data(
        output_dir=args.output_dir,
        database_path=args.database,
        start_date=start,
        end_date=end,
        mode=args.mode,
        out_format=args.format,
    )
