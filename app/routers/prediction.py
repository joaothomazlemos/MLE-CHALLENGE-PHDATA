"""Prediction endpoints for house price estimation."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status

from ..models.prediction import (
    HouseFeatures,
    MinimalHouseFeatures,
    PredictionResponse,
    PredictionMetadata,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
)
from ..services.model_service import ModelService
from ..dependencies import get_model_service
from ..internal.exceptions import (
    ModelNotFoundError,
    PredictionError,
    DataProcessingError,
)

router = APIRouter(
    prefix="/prediction",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


def create_error_response(error_type: str, message: str, details: Optional[dict] = None) -> ErrorResponse:
    """Create standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def create_prediction_metadata(
    model_service: ModelService,
    zipcode_demographics_available: bool,
    features_used: int,
) -> PredictionMetadata:
    """Create prediction metadata."""
    return PredictionMetadata(
        model_metadata={"model_version": "1.0.0"},  # Fixed: dict instead of direct field
        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
        zipcode_demographics_available=zipcode_demographics_available,
        features_used=features_used
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict house price",
    description="Predict house price using full feature set from future_unseen_examples.csv format",
    response_description="Prediction with metadata",
)
async def predict_house_price(
    house_data: HouseFeatures,
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """
    Predict house price using complete feature set.
    
    This endpoint accepts all features as provided in the future_unseen_examples.csv format,
    joins with zipcode demographics data, and returns a price prediction with metadata.
    
    Args:
        house_data: Complete house features including all columns from future_unseen_examples.csv
        model_service: Injected model service dependency
        
    Returns:
        PredictionResponse with predicted price and metadata
        
    Raises:
        HTTPException: For various error conditions (validation, model, data processing)
    """
    try:
        house_dict = house_data.model_dump()
        
        zipcode = house_dict.pop('zipcode')
        
        result = model_service.predict(house_dict, zipcode)
        predicted_price = result["predicted_price"]
        
        zipcode_demographics_available = result.get("has_demographics", True)
        features_used = result.get("features_used", len(house_dict))
        
        metadata = create_prediction_metadata(
            model_service=model_service,
            zipcode_demographics_available=zipcode_demographics_available,
            features_used=features_used,
        )
        
        return PredictionResponse(
            predicted_price=predicted_price,
            metadata=metadata,
        )
        
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(
                error_type="ModelNotFoundError",
                message=str(e),
                details={"zipcode": house_data.zipcode},
            ).model_dump(),
        )
    except DataProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=create_error_response(
                error_type="DataProcessingError",
                message=str(e),
                details={"zipcode": house_data.zipcode},
            ).model_dump(),
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="PredictionError",
                message=str(e),
                details={"zipcode": house_data.zipcode},
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="InternalServerError",
                message="An unexpected error occurred during prediction",
                details={"original_error": str(e)},
            ).model_dump(),
        )


@router.post(
    "/predict/minimal",
    response_model=PredictionResponse,
    summary="Predict house price with minimal features",
    description="Predict house price using only essential features with smart defaults for optional ones",
    response_description="Prediction with metadata",
)
async def predict_house_price_minimal(
    house_data: MinimalHouseFeatures,
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """
    Predict house price using minimal feature set with smart defaults.
    
    This endpoint accepts only essential features and applies smart defaults for optional ones.
    Missing geographic and neighborhood features are estimated based on zipcode when possible.
    
    Args:
        house_data: Essential house features with optional advanced features
        model_service: Injected model service dependency
        
    Returns:
        PredictionResponse with predicted price and metadata
        
    Raises:
        HTTPException: For various error conditions (validation, model, data processing)
    """
    try:
        house_dict = house_data.model_dump()
        
        if house_dict.get("lat") is None:
            house_dict["lat"] = 0
        if house_dict.get("long") is None:
            house_dict["long"] = 0
        if house_dict.get("sqft_living15") is None:
            house_dict["sqft_living15"] = house_dict["sqft_living"]
        if house_dict.get("sqft_lot15") is None:
            house_dict["sqft_lot15"] = house_dict["sqft_lot"]
            
        try:
            result = model_service.predict_minimal(house_dict)
            predicted_price = result["predicted_price"]
            actual_features_used = result.get("features_used", len(house_dict))
        except AttributeError:
            zipcode = house_dict.get('zipcode', '98115')
            if 'zipcode' in house_dict:
                house_dict_copy = house_dict.copy()
                zipcode = house_dict_copy.pop('zipcode')
                result = model_service.predict(house_dict_copy, zipcode)
                predicted_price = result["predicted_price"]
                actual_features_used = result.get("features_used", len(house_dict))
            else:
                result = model_service.predict_minimal(house_dict)
                predicted_price = result["predicted_price"]
                actual_features_used = result.get("features_used", len(house_dict))
        
        zipcode_demographics_available = result.get("has_demographics", False)
        features_used = actual_features_used
        
        metadata = create_prediction_metadata(
            model_service=model_service,
            zipcode_demographics_available=zipcode_demographics_available,
            features_used=features_used,
        )
        
        return PredictionResponse(
            predicted_price=predicted_price,
            metadata=metadata,
        )
        
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(
                error_type="ModelNotFoundError",
                message=str(e),
                details={"zipcode": house_data.zipcode},
            ).model_dump(),
        )
    except DataProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=create_error_response(
                error_type="DataProcessingError",
                message=str(e),
                details={"zipcode": house_data.zipcode},
            ).model_dump(),
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="PredictionError",
                message=str(e),
                details={"zipcode": house_data.zipcode},
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="InternalServerError",
                message="An unexpected error occurred during minimal prediction",
                details={"original_error": str(e)},
            ).model_dump(),
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict multiple house prices",
    description="Predict prices for multiple houses in a single request (1-100 houses)",
    response_description="Batch predictions with metadata",
)
async def predict_batch_house_prices(
    batch_request: BatchPredictionRequest,
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    """
    Predict house prices for multiple houses in a single batch request.
    
    This endpoint processes multiple house predictions efficiently and returns
    all predictions with batch-level metadata for performance tracking.
    
    Args:
        batch_request: List of houses to predict (1-100 houses)
        model_service: Injected model service dependency
        
    Returns:
        BatchPredictionResponse with all predictions and batch metadata
        
    Raises:
        HTTPException: For various error conditions (validation, model, data processing)
    """
    start_time = datetime.now(timezone.utc)
    batch_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    try:
        predictions: List[PredictionResponse] = []
        
        for i, house_data in enumerate(batch_request.houses):
            try:
                house_dict = house_data.model_dump()
                
                zipcode = house_dict.pop('zipcode')
                
                result = model_service.predict(house_dict, zipcode)
                predicted_price = result["predicted_price"]
                
                metadata = create_prediction_metadata(
                    model_service=model_service,
                    zipcode_demographics_available=True,
                    features_used=len(house_dict) + 26,
                )
                
                predictions.append(PredictionResponse(
                    predicted_price=predicted_price,
                    metadata=metadata,
                ))
                
            except Exception as e:
                raise PredictionError(f"Failed to predict house {i+1}: {str(e)}")
        
        end_time = datetime.now(timezone.utc)
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        batch_metadata = {
            "total_predictions": len(predictions),
            "processing_time_ms": processing_time_ms,
            "batch_id": batch_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_metadata=batch_metadata,
        )
        
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(
                error_type="ModelNotFoundError",
                message=str(e),
                details={"batch_id": batch_id},
            ).model_dump(),
        )
    except DataProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=create_error_response(
                error_type="DataProcessingError",
                message=str(e),
                details={"batch_id": batch_id},
            ).model_dump(),
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="PredictionError",
                message=str(e),
                details={"batch_id": batch_id},
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="InternalServerError",
                message="An unexpected error occurred during batch prediction",
                details={"batch_id": batch_id, "original_error": str(e)},
            ).model_dump(),
        )


@router.get(
    "/health",
    summary="Check prediction service health",
    description="Health check endpoint for prediction service and model availability",
    response_description="Service health status",
)
async def health_check(
    model_service: ModelService = Depends(get_model_service),
) -> dict:
    """
    Check health of prediction service and model availability.
    
    Returns basic health information about the prediction service,
    including model loading status and service readiness.
    
    Args:
        model_service: Injected model service dependency
        
    Returns:
        Dictionary with health status information
    """
    try:
        service_health = model_service.health_check()
        
        health_status = {
            "service": "prediction",
            "status": "healthy" if service_health.get("healthy", False) else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_loaded": service_health.get("model_loaded", False),
            "demographics_loaded": service_health.get("demographics_loaded", False),
            "version": "1.0.0",
            "features_count": service_health.get("features_count", 0),
            "zipcodes_count": service_health.get("zipcodes_count", 0),
        }
        
        return health_status
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=create_error_response(
                error_type="ServiceUnavailable",
                message="Prediction service is not healthy",
                details={"error": str(e)},
            ).model_dump(),
        )