"""Configuration module for opponent-adjusted metrics project."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://user:password@localhost:5432/opponent_metrics",
        alias="DATABASE_URL",
    )

    # Data paths
    data_root: Path = Field(default=Path("./data"), alias="DATA_ROOT")
    statsbomb_data_path: Path = Field(
        default=Path("./data/statsbomb"), alias="STATSBOMB_DATA_PATH"
    )
    model_artifacts_path: Path = Field(
        default=Path("./data/artifacts"), alias="MODEL_ARTIFACTS_PATH"
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Feature versioning
    default_feature_version: str = "v1"

    # Model configuration
    lightgbm_params: dict = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "min_child_samples": 20,
        "random_state": 42,
    }

    # Competition IDs for StatsBomb Open Data
    competitions: list = [
        {"name": "FIFA World Cup", "season": "2018"},
        {"name": "UEFA Euro", "season": "2020"},
        {"name": "FIFA World Cup", "season": "2022"},
        {"name": "UEFA Euro", "season": "2024"},
    ]

    # Zone definitions for opponent profiles
    zone_definitions: dict = {
        "A": {"max_distance": 12, "max_centrality": 10},  # Close central
        "B": {"max_distance": 12, "min_centrality": 10},  # Close wide
        "C": {"min_distance": 12, "max_distance": 20, "max_centrality": 10},  # Mid central
        "D": {"min_distance": 12, "max_distance": 20, "min_centrality": 10},  # Mid wide
        "E": {"min_distance": 20, "max_centrality": 10},  # Far central
        "F": {"min_distance": 20, "min_centrality": 10},  # Far wide
    }

    # Neutralization reference context
    neutralization_context: dict = {
        "score_diff_at_shot": 0,
        "minute": 55,
        "minute_bucket": "46-60",
        "under_pressure": False,
        "opponent_def_rating_global": 0.0,
        "opponent_def_zone_rating": 0.0,
    }

    # StatsBomb pitch dimensions
    pitch_length: float = 120.0
    pitch_width: float = 80.0
    goal_center_x: float = 120.0
    goal_center_y: float = 40.0
    goal_post_left_y: float = 36.8
    goal_post_right_y: float = 43.2

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        settings.data_root,
        settings.statsbomb_data_path,
        settings.model_artifacts_path,
        settings.data_root / "reports" / "calibration",
        settings.data_root / "reports" / "slices",
        settings.data_root / "reports" / "feature_importance",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    print("Configuration loaded successfully!")
    print(f"Database URL: {settings.database_url[:30]}...")
    print(f"Data root: {settings.data_root}")
    print(f"Log level: {settings.log_level}")
    ensure_directories()
    print("All directories created successfully!")
