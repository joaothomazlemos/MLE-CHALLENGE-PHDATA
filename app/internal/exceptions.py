"""
Custom exceptions for the house price prediction API.
"""


class ModelServiceError(Exception):
    """Base exception for model service errors."""

    pass


class ModelLoadError(ModelServiceError):
    """Raised when model loading fails."""

    pass


class PredictionError(ModelServiceError):
    """Raised when prediction computation fails."""

    pass


class ModelNotFoundError(ModelServiceError):
    """Raised when model or model components are not found."""

    pass


class DataProcessingError(ModelServiceError):
    """Raised when data processing fails."""

    pass
