"""Tests for the ML pipeline scripts."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory structure."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_duckdb(temp_data_dir: Path) -> Path:
    """Create a sample DuckDB database with crime data.

    Generates enough data to survive weekly aggregation and lag filtering:
    - 2 distinct grid cells (nearby coordinates)
    - 30 weeks of data (need 8+ weeks for rolling_mean_8)
    - Multiple crimes per grid/week to have meaningful counts
    """
    import duckdb

    db_path = temp_data_dir / "data" / "crimes.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create crimes table
    conn.execute(
        """
        CREATE TABLE crimes (
            id VARCHAR,
            date TIMESTAMP,
            primary_type VARCHAR,
            description VARCHAR,
            location_description VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE,
            arrest BOOLEAN,
            domestic BOOLEAN,
            beat VARCHAR,
            district VARCHAR,
            ward VARCHAR,
            community_area VARCHAR
        )
    """
    )

    # Generate sample data - need many weeks across few grid cells
    # Using 30 weeks of data with multiple crimes per week per cell
    start_date = date.today() - timedelta(days=210)  # ~30 weeks ago
    records = []

    # Define 2 fixed locations that will map to distinct grid cells
    locations = [
        (41.7500, -87.6500),  # Location A
        (41.7600, -87.6400),  # Location B (different grid cell)
    ]

    record_id = 0
    for week in range(30):  # 30 weeks
        week_start = start_date + timedelta(weeks=week)
        for loc_idx, (lat, lon) in enumerate(locations):
            # Generate 5-10 crimes per grid cell per week
            crimes_this_week = 5 + (week % 5)
            for crime_num in range(crimes_this_week):
                day_offset = crime_num % 7
                hour = (crime_num * 3) % 24
                record_date = week_start + timedelta(days=day_offset, hours=hour)
                records.append(
                    (
                        f"ID{record_id:06d}",
                        record_date.isoformat(),
                        "THEFT"
                        if crime_num % 3 == 0
                        else "BATTERY"
                        if crime_num % 3 == 1
                        else "ASSAULT",
                        "TEST DESCRIPTION",
                        "STREET" if crime_num % 2 == 0 else "RESIDENCE",
                        lat + (crime_num % 3) * 0.001,  # Small variation within grid
                        lon + (crime_num % 3) * 0.001,
                        crime_num % 5 == 0,  # arrest
                        crime_num % 7 == 0,  # domestic
                        str(loc_idx * 10 + 1),
                        str(loc_idx + 1),
                        str(loc_idx * 25),
                        str(loc_idx * 38),
                    )
                )
                record_id += 1

    conn.executemany(
        "INSERT INTO crimes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.close()

    return db_path


@pytest.fixture
def sample_features_parquet(temp_data_dir: Path) -> Path:
    """Create a sample features parquet file.

    Creates data for 2 grid cells over 30 weeks to support temporal split.
    Uses the 14-feature schema expected by the training script.
    """
    import math

    features_path = temp_data_dir / "data" / "processed" / "features.parquet"

    # Create sample weekly aggregated data for multiple grid cells over 30 weeks
    records = []
    for grid_id in [100, 200]:  # 2 grid cells
        for week in range(1, 31):  # 30 weeks
            base_count = 50 + week * 2 + (grid_id // 10)
            rolling_mean_4 = float(base_count - 5)
            lag1 = base_count - 2

            # Fourier encoding for week_of_year
            angle1 = week * 2 * math.pi / 52
            angle2 = week * 4 * math.pi / 52

            records.append(
                {
                    "year": 2025,
                    "week_of_year": week,
                    "grid_id": grid_id,
                    "crime_count": base_count,
                    "hour_mean": 12.5,
                    "is_weekend_ratio": 0.3,
                    "lat_bin": grid_id // 100,
                    "lon_bin": grid_id % 100,
                    # 14 features for training:
                    "crime_count_lag1": lag1,
                    "crime_count_lag2": base_count - 4,
                    "crime_count_lag3": base_count - 6,
                    "crime_count_lag4": base_count - 8,
                    "crime_count_rolling_mean_4": rolling_mean_4,
                    "crime_count_rolling_std_4": 2.5,
                    "crime_count_rolling_mean_8": float(base_count - 10),
                    "crime_trend": float(lag1 - rolling_mean_4),
                    "week_sin": math.sin(angle1),
                    "week_cos": math.cos(angle1),
                    "week_sin2": math.sin(angle2),
                    "week_cos2": math.cos(angle2),
                    "month": (week // 4) + 1,
                }
            )

    df = pl.DataFrame(records)
    df.write_parquet(features_path)

    return features_path


class TestProcessData:
    """Tests for process_data.py script."""

    def test_process_data_creates_output(self, temp_data_dir: Path, sample_duckdb: Path) -> None:
        """Test that process_data creates output file."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from process_data import process_data

        output_path = temp_data_dir / "data" / "processed" / "features.parquet"

        process_data(sample_duckdb, output_path)

        assert output_path.exists()

    def test_process_data_output_has_required_columns(
        self, temp_data_dir: Path, sample_duckdb: Path
    ) -> None:
        """Test that processed data has all required feature columns."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from process_data import process_data

        output_path = temp_data_dir / "data" / "processed" / "test_features.parquet"

        process_data(sample_duckdb, output_path)

        df = pl.read_parquet(output_path)

        required_columns = [
            "year",
            "week_of_year",
            "grid_id",
            "crime_count",
            "hour_mean",
            "is_weekend_ratio",
            "lat_bin",
            "lon_bin",
            "crime_count_lag1",
            "crime_count_lag2",
            "crime_count_lag4",
            "crime_count_rolling_mean_4",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_process_data_removes_null_lags(self, temp_data_dir: Path, sample_duckdb: Path) -> None:
        """Test that rows with null lag values are filtered out."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from process_data import process_data

        output_path = temp_data_dir / "data" / "processed" / "no_nulls.parquet"

        process_data(sample_duckdb, output_path)

        df = pl.read_parquet(output_path)

        # Check no nulls in lag columns
        assert df["crime_count_lag1"].null_count() == 0
        assert df["crime_count_lag2"].null_count() == 0
        assert df["crime_count_lag4"].null_count() == 0

    def test_fallback_processing_without_eventflow(
        self, temp_data_dir: Path, sample_duckdb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that fallback processing works when EventFlow unavailable."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # Mock eventflow import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "eventflow" or name.startswith("eventflow."):
                raise ImportError("Mocked eventflow unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Re-import to pick up mocked import
        import importlib

        import process_data

        importlib.reload(process_data)

        output_path = temp_data_dir / "data" / "processed" / "fallback.parquet"

        process_data.process_data(sample_duckdb, output_path)

        assert output_path.exists()
        df = pl.read_parquet(output_path)
        assert len(df) > 0


class TestTrainModel:
    """Tests for train_model.py script."""

    def test_train_model_creates_output(
        self, temp_data_dir: Path, sample_features_parquet: Path
    ) -> None:
        """Test that train_model creates model file."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_model import train_model

        model_path = temp_data_dir / "models" / "test_model.joblib"

        train_model(sample_features_parquet, model_path, "random_forest", test_weeks=4)

        assert model_path.exists()

    def test_train_model_creates_metadata(
        self, temp_data_dir: Path, sample_features_parquet: Path
    ) -> None:
        """Test that train_model creates metadata file."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_model import train_model

        model_path = temp_data_dir / "models" / "meta_test.joblib"

        train_model(sample_features_parquet, model_path, "random_forest", test_weeks=4)

        metadata_path = temp_data_dir / "models" / "model_metadata.json"
        assert metadata_path.exists()

        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "model_type" in metadata
        assert "metrics" in metadata
        assert "features" in metadata
        assert metadata["model_type"] == "random_forest"

    def test_train_model_poisson(self, temp_data_dir: Path, sample_features_parquet: Path) -> None:
        """Test that Poisson model can be trained."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_model import train_model

        model_path = temp_data_dir / "models" / "poisson_model.joblib"

        train_model(sample_features_parquet, model_path, "poisson", test_weeks=4)

        assert model_path.exists()

        import json

        metadata_path = temp_data_dir / "models" / "model_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["model_type"] == "poisson"

    def test_train_model_metrics_are_valid(
        self, temp_data_dir: Path, sample_features_parquet: Path
    ) -> None:
        """Test that training produces valid metrics."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_model import train_model

        model_path = temp_data_dir / "models" / "metrics_test.joblib"

        train_model(sample_features_parquet, model_path, "random_forest", test_weeks=4)

        import json

        metadata_path = temp_data_dir / "models" / "model_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        metrics = metadata["metrics"]

        # Check all expected metrics exist
        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics
        assert "test_mae" in metrics
        assert "test_rmse" in metrics
        assert "test_r2" in metrics
        assert "test_mape" in metrics
        assert "improvement_vs_baseline" in metrics

        # Check metrics are reasonable values
        assert metrics["train_mae"] >= 0
        assert metrics["train_rmse"] >= 0
        assert metrics["test_mae"] >= 0
        assert metrics["test_rmse"] >= 0
        assert metrics["test_mape"] >= 0

        # Check baselines were calculated
        assert "baselines" in metadata
        assert "baseline_last_week" in metadata["baselines"]
        assert "baseline_rolling_mean" in metadata["baselines"]

    def test_train_model_can_make_predictions(
        self, temp_data_dir: Path, sample_features_parquet: Path
    ) -> None:
        """Test that trained model can make predictions."""
        import sys

        import joblib

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_model import train_model

        model_path = temp_data_dir / "models" / "predict_test.joblib"

        train_model(sample_features_parquet, model_path, "random_forest", test_weeks=4)

        # Load and use model
        model = joblib.load(model_path)

        # Create test input with 14 features matching training feature columns:
        # crime_count_lag1, lag2, lag3, lag4, rolling_mean_4, rolling_std_4,
        # rolling_mean_8, crime_trend, week_sin, week_cos, week_sin2, week_cos2,
        # month, is_weekend_ratio
        test_input = [
            [
                50.0,  # crime_count_lag1
                48.0,  # crime_count_lag2
                46.0,  # crime_count_lag3
                44.0,  # crime_count_lag4
                47.0,  # crime_count_rolling_mean_4
                2.5,  # crime_count_rolling_std_4
                45.0,  # crime_count_rolling_mean_8
                3.0,  # crime_trend
                0.5,  # week_sin
                0.87,  # week_cos
                0.97,  # week_sin2
                -0.26,  # week_cos2
                6.0,  # month
                0.3,  # is_weekend_ratio
            ]
        ]

        predictions = model.predict(test_input)

        assert len(predictions) == 1
        assert predictions[0] > 0  # Crime count should be positive


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_flow(self, temp_data_dir: Path, sample_duckdb: Path) -> None:
        """Test that the full pipeline can run end-to-end."""
        import sys

        import joblib

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from process_data import process_data
        from train_model import train_model

        # Step 1: Use the sample_duckdb fixture (already created)
        assert sample_duckdb.exists()

        # Step 2: Process data
        features_path = temp_data_dir / "data" / "processed" / "features.parquet"
        process_data(sample_duckdb, features_path)

        assert features_path.exists()

        # Step 3: Train model
        model_path = temp_data_dir / "models" / "model.joblib"
        train_model(features_path, model_path, "random_forest", test_weeks=4)

        assert model_path.exists()
        assert (temp_data_dir / "models" / "model_metadata.json").exists()

        # Verify model works with 14-feature input
        model = joblib.load(model_path)
        # Create test input with 14 features matching training feature columns:
        # crime_count_lag1, lag2, lag3, lag4, rolling_mean_4, rolling_std_4,
        # rolling_mean_8, crime_trend, week_sin, week_cos, week_sin2, week_cos2,
        # month, is_weekend_ratio
        test_input = [
            [
                50.0,  # crime_count_lag1
                48.0,  # crime_count_lag2
                46.0,  # crime_count_lag3
                44.0,  # crime_count_lag4
                47.0,  # crime_count_rolling_mean_4
                2.5,  # crime_count_rolling_std_4
                45.0,  # crime_count_rolling_mean_8
                3.0,  # crime_trend
                0.5,  # week_sin
                0.87,  # week_cos
                0.97,  # week_sin2
                -0.26,  # week_cos2
                6.0,  # month
                0.3,  # is_weekend_ratio
            ]
        ]
        predictions = model.predict(test_input)

        assert len(predictions) == 1
