"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Settings
    app_name: str = "Chicago Crime Prediction API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Data Settings
    data_path: str = "../chicago_crime_data_cli/data/monthly_2020_2025/monthly"
    model_path: str = "models"

    # MLflow Settings
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "chicago-crime-prediction"

    # Model Settings
    model_version: str = "latest"
    prediction_horizon_days: int = 7
    grid_size: tuple[int, int] = (10, 10)

    # Cache Settings
    cache_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
