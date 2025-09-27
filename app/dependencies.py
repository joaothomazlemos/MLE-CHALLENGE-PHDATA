"""
Shared dependencies for the FastAPI application.

This module contains dependency injection functions that can be used
across multiple routers and endpoints.
"""

from functools import lru_cache
from typing import Dict

from app.internal.config import get_config
from app.services.model_service import ModelService


@lru_cache()
def get_model_service() -> ModelService:
    """
    Get a cached instance of the ModelService.

    This dependency ensures that the model is loaded only once
    and reused across requests for better performance.

    Returns:
        ModelService: Singleton instance of the model service
    """
    config = get_config()
    return ModelService(
        model_path=config.model_path,
        features_path=config.model_features_path,
        demographics_path=config.demographics_path,
    )


def get_api_metadata() -> Dict[str, str]:
    """
    Get API metadata that can be included in responses.

    Returns:
        Dictionary with API metadata
    """
    return {
        "api_version": "1.0.0",
        "model_type": "KNeighborsRegressor",
        "service": "house-price-prediction",
    }
