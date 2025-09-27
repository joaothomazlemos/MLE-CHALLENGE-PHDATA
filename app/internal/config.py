"""
Configuration settings for the house price prediction API.

Using Pydantic for configuration management with environment variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Config(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": (),
    }

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_base_url: str = Field(default="http://localhost:8080", env="API_BASE_URL")

    model_path: str = Field(default="model/model.pkl", env="MODEL_PATH")
    model_features_path: str = Field(
        default="model/model_features.json", env="MODEL_FEATURES_PATH"
    )
    demographics_path: str = Field(
        default="data/zipcode_demographics.csv", env="DEMOGRAPHICS_PATH"
    )
    future_examples_path: str = Field(
        default="data/future_unseen_examples.csv", env="FUTURE_EXAMPLES_PATH"
    )

    title: str = "Sound Realty House Price Prediction API"
    description: str = (
        "REST API for predicting house prices in the Seattle area using machine learning"
    )
    version: str = "1.0.0"

    allowed_origins: List[str] = Field(default=["*"])

    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    environment: str = Field(default="development", env="ENVIRONMENT")

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    def get_model_path(self) -> Path:
        """Get the model file path as Path object."""
        return Path(self.model_path)

    def get_features_path(self) -> Path:
        """Get the features file path as Path object."""
        return Path(self.model_features_path)

    def get_demographics_path(self) -> Path:
        """Get the demographics file path as Path object."""
        return Path(self.demographics_path)

    def get_future_examples_path(self) -> Path:
        """Get the future examples file path as Path object."""
        return Path(self.future_examples_path)


@lru_cache()
def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
