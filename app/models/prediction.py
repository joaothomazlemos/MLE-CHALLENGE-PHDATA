"""Pydantic models for house price prediction API."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from app.services.model_service import ModelService

class HouseFeatures(BaseModel):
    """Input schema for house prediction - matches future_unseen_examples.csv format."""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2500,
                "sqft_lot": 8000,
                "floors": 2.0,
                "waterfront": 0,
                "view": 0,
                "condition": 4,
                "grade": 8,
                "sqft_above": 2000,
                "sqft_basement": 500,
                "yr_built": 1990,
                "yr_renovated": 0,
                "zipcode": "98115",
                "lat": 47.6974,
                "long": -122.313,
                "sqft_living15": 2200,
                "sqft_lot15": 7500
            }
        }
    }
    
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=15000, description="Square feet of living space")
    sqft_lot: int = Field(..., ge=500, le=200000, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=5, description="Number of floors")
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0=No, 1=Yes)")
    view: int = Field(..., ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Condition rating (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Grade rating (1-13)")
    sqft_above: int = Field(..., ge=100, le=15000, description="Square feet above ground")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Square feet of basement")
    yr_built: int = Field(..., ge=1800, le=2025, description="Year built")
    yr_renovated: int = Field(..., ge=0, le=2025, description="Year renovated (0 if never)")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Longitude")
    sqft_living15: int = Field(..., ge=100, le=15000, description="Living space of nearest 15 neighbors")
    sqft_lot15: int = Field(..., ge=500, le=200000, description="Lot size of nearest 15 neighbors")


class MinimalHouseFeatures(BaseModel):
    """Minimal input schema for house prediction - essential features only."""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.0,
                "sqft_living": 1800,
                "sqft_lot": 6000,
                "floors": 1.5,
                "sqft_above": 1200,
                "sqft_basement": 600,
                "zipcode": "98115"
            }
        }
    }
    
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=15000, description="Square feet of living space")
    sqft_lot: int = Field(..., ge=500, le=200000, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=5, description="Number of floors")
    sqft_above: int = Field(..., ge=100, le=15000, description="Square feet above ground")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Square feet of basement")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")

    waterfront: int = Field(0, ge=0, le=1, description="Waterfront property (0=No, 1=Yes)")
    view: int = Field(0, ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(3, ge=1, le=5, description="Condition rating (1-5)")
    grade: int = Field(7, ge=1, le=13, description="Grade rating (1-13)")
    yr_built: int = Field(1980, ge=1800, le=2025, description="Year built")
    yr_renovated: int = Field(0, ge=0, le=2025, description="Year renovated (0 if never)")
    lat: Optional[float] = Field(None, ge=47.0, le=48.0, description="Latitude (auto-filled from zipcode)")
    long: Optional[float] = Field(None, ge=-123.0, le=-121.0, description="Longitude (auto-filled from zipcode)")
    sqft_living15: Optional[int] = Field(None, ge=100, le=15000, description="Living space of nearest 15 neighbors")
    sqft_lot15: Optional[int] = Field(None, ge=500, le=200000, description="Lot size of nearest 15 neighbors")


class PredictionMetadata(BaseModel):
    """Metadata for prediction response."""
    
    model_config = {"protected_namespaces": ()}

    model_metadata: dict = Field(..., description="Metadata of the ML model used, from ModelService")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")
    zipcode_demographics_available: bool = Field(..., description="Whether demographics data was found for zipcode")
    features_used: int = Field(..., description="Number of features used in prediction")


class PredictionResponse(BaseModel):
    """Response schema for house price prediction."""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_price": 450000.0,
                "metadata": {
                    "model_metadata": ModelService.get_model_metadata,
                    "prediction_timestamp": datetime.now().isoformat(),
                    "zipcode_demographics_available": True,
                    "features_used": 25
                }
            }
        }
    }
    
    predicted_price: float = Field(..., ge=0, description="Predicted house price in USD")
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "houses": [
                    {
                        "bedrooms": 4,
                        "bathrooms": 2.5,
                        "sqft_living": 2500,
                        "sqft_lot": 8000,
                        "floors": 2.0,
                        "waterfront": 0,
                        "view": 0,
                        "condition": 4,
                        "grade": 8,
                        "sqft_above": 2000,
                        "sqft_basement": 500,
                        "yr_built": 1990,
                        "yr_renovated": 0,
                        "zipcode": "98115",
                        "lat": 47.6974,
                        "long": -122.313,
                        "sqft_living15": 2200,
                        "sqft_lot15": 7500
                    }
                ]
            }
        }
    }
    
    houses: list[HouseFeatures] = Field(..., min_items=1, max_items=100, description="List of houses to predict")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "predicted_price": 450000.0,
                        "metadata": {
                            "model_metadata": {ModelService.get_model_metadata},
                            "prediction_timestamp": datetime.now().isoformat(),
                            "zipcode_demographics_available": True,
                            "features_used": 25
                        }
                    }
                ],
                "batch_metadata": {
                    "total_predictions": 1,
                    "processing_time_ms": 150,
                    "batch_id": "batch_20250920_103000"
                }
            }
        }
    }
    
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    batch_metadata: dict = Field(..., description="Batch processing metadata")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid zipcode provided",
                "details": {
                    "field": "zipcode",
                    "provided_value": "1234",
                    "expected": "5-digit zipcode"
                },
                "timestamp": "2025-09-20T10:30:00Z"
            }
        }
    }
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="ISO timestamp of error")