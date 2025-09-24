"""Models package for house price prediction API."""

from .prediction import (
    HouseFeatures,
    MinimalHouseFeatures,
    PredictionMetadata,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
)

__all__ = [
    "HouseFeatures",
    "MinimalHouseFeatures", 
    "PredictionMetadata",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ErrorResponse",
]