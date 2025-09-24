"""
Utility functions for data processing.

Simple helper functions for data transformation used by the model service.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def ensure_numeric_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all numeric fields are properly typed.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Data with proper numeric types
    """
    processed = {}
    
    for key, value in data.items():
        if key == 'zipcode':
            processed[key] = str(value).zfill(5)
        elif isinstance(value, (int, float)):
            processed[key] = float(value)
        else:
            processed[key] = value
    
    return processed


def fill_missing_demographics(data: Dict[str, Any], required_features: List[str]) -> Dict[str, Any]:
    """
    Fill missing demographic features with default values.
    
    Args:
        data: Input data dictionary
        required_features: List of required feature names
        
    Returns:
        Data with missing demographic features filled with defaults
    """
    filled_data = data.copy()
    
    for feature in required_features:
        if feature not in filled_data:
            filled_data[feature] = 0.0
            logger.warning(f"Using default value 0.0 for missing feature: {feature}")
    
    return filled_data


def log_prediction_request(zipcode: str, has_demographics: bool) -> None:
    """
    Log information about a prediction request.
    
    Args:
        zipcode: The zipcode for the prediction
        has_demographics: Whether demographic data was available
    """
    logger.info(
        f"Prediction request - Zipcode: {zipcode}, "
        f"Demographics: {'Available' if has_demographics else 'Missing'}"
    )