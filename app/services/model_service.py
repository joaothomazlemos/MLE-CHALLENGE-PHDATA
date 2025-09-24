"""
Model service for house price prediction.

This module handles loading the trained model, demographics data, and provides
prediction functionality.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from app.internal.exceptions import ModelLoadError, PredictionError
from app.internal.data_utils import ensure_numeric_types, fill_missing_demographics, log_prediction_request

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and using the house price prediction model."""
    
    def __init__(
        self,
        model_path: str,
        features_path: str,
        demographics_path: str,
        model_metadata: Optional[Dict] = {"model_name": "KNeighborsRegressor",
            "version": "1.0.0"}
    ):
        """
        Initialize the model service.
        
        Args:
            model_path: Path to the pickled model file
            features_path: Path to the JSON file containing model features
            demographics_path: Path to the demographics CSV file
        """
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.demographics_path = Path(demographics_path)
        self._model_metadata = model_metadata
        
        self._model: Optional[Pipeline] = None
        self._features: Optional[List[str]] = None
        self._demographics: Optional[pd.DataFrame] = None
        
        self._load_components()
        
        logger.info("ModelService initialized successfully")
    
    def _load_components(self) -> None:
        """Load all model components."""
        self._load_model()
        self._load_features()
        self._load_demographics()
    
    def _load_model(self) -> None:
        """Load the trained model from pickle file."""
        try:
            if not self.model_path.exists():
                raise ModelLoadError(f"Model file not found: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")
    
    def _load_features(self) -> None:
        """Load the model features from JSON file."""
        try:
            if not self.features_path.exists():
                raise ModelLoadError(f"Features file not found: {self.features_path}")
            
            with open(self.features_path, 'r') as f:
                self._features = json.load(f)
            
            logger.info(f"Model features loaded: {len(self._features)} features")
            
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            raise ModelLoadError(f"Features loading failed: {e}")
    
    def _load_demographics(self) -> None:
        """Load demographics data from CSV file."""
        try:
            if not self.demographics_path.exists():
                raise ModelLoadError(f"Demographics file not found: {self.demographics_path}")
            
            self._demographics = pd.read_csv(
                self.demographics_path,
                dtype={'zipcode': str}
            )
            
            logger.info(f"Demographics data loaded: {len(self._demographics)} zipcodes")
            
        except Exception as e:
            logger.error(f"Failed to load demographics: {e}")
            raise ModelLoadError(f"Demographics loading failed: {e}")
    
    def get_demographics_for_zipcode(self, zipcode: str) -> Optional[Dict]:
        """
        Get demographic data for a specific zipcode.
        
        Args:
            zipcode: The zipcode to look up
            
        Returns:
            Dictionary with demographic data or None if not found
        """
        if self._demographics is None:
            raise ModelLoadError("Demographics data not loaded")
        
        zipcode = str(zipcode).zfill(5)
        
        demographic_row = self._demographics[
            self._demographics['zipcode'] == zipcode
        ]
        
        if demographic_row.empty:
            logger.warning(f"No demographic data found for zipcode: {zipcode}")
            return None
        
        demo_dict = demographic_row.iloc[0].to_dict()
        demo_dict.pop('zipcode', None)
        
        return demo_dict
    
    def predict(self, house_data: Dict, zipcode: str) -> Dict:
        """
        Make a price prediction for a house.
        
        Args:
            house_data: Dictionary containing house features
            zipcode: The zipcode for demographic lookup
            
        Returns:
            Dictionary containing prediction and metadata
        """
        if self._model is None or self._features is None:
            raise ModelLoadError("Model components not loaded")
        
        try:
            processed_data = ensure_numeric_types(house_data)
            
            demographics = self.get_demographics_for_zipcode(zipcode)
            has_demographics = demographics is not None
            
            combined_data = processed_data.copy()
            if demographics:
                combined_data.update(demographics)
            
            combined_data = fill_missing_demographics(combined_data, self._features)
            
            prediction_df = pd.DataFrame([combined_data])
            prediction_df = prediction_df[self._features]
            
            prediction = self._model.predict(prediction_df)[0]
            
            log_prediction_request(zipcode, has_demographics)
            
            result = {
                "predicted_price": float(prediction),
                "zipcode": zipcode,
                "has_demographics": has_demographics,
                "confidence": "high" if has_demographics else "medium",
                "features_used": len(self._features)
            }
            
            logger.info(f"Prediction made for zipcode {zipcode}: ${prediction:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def predict_minimal(self, minimal_data: Dict) -> Dict:
        """
        Make a prediction using only essential features.
        
        Args:
            minimal_data: Dictionary with minimal required features
            
        Returns:
            Dictionary containing prediction and metadata
        """
        if self._model is None or self._features is None:
            raise ModelLoadError("Model components not loaded")
        
        try:
            processed_data = ensure_numeric_types(minimal_data)
            
            full_data = fill_missing_demographics(processed_data, self._features)
            
            prediction_df = pd.DataFrame([full_data])
            prediction_df = prediction_df[self._features]
            
            prediction = self._model.predict(prediction_df)[0]
            
            result = {
                "predicted_price": float(prediction),
                "has_demographics": False,
                "confidence": "low",
                "features_used": len(self._features),
                "note": "Prediction made without demographic data"
            }
            
            logger.info(f"Minimal prediction made: ${prediction:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Minimal prediction failed: {e}")
            raise PredictionError(f"Minimal prediction failed: {e}")
    
    @property
    def get_model_metadata(self) -> Dict:
        """Get metadata about the model."""
        return self._model_metadata if self._model_metadata is not None else {}

    @property
    def model_features(self) -> List[str]:
        """Get the list of model features."""
        if self._features is None:
            raise ModelLoadError("Features not loaded")
        return self._features.copy()
    
    @property
    def available_zipcodes(self) -> List[str]:
        """Get list of available zipcodes in demographics data."""
        if self._demographics is None:
            raise ModelLoadError("Demographics data not loaded")
        return self._demographics['zipcode'].tolist()
    
    def health_check(self) -> Dict:
        """
        Perform a health check on the model service.
        
        Returns:
            Dictionary with health status information
        """
        try:
            model_loaded = self._model is not None
            features_loaded = self._features is not None and len(self._features) > 0
            demographics_loaded = self._demographics is not None and not self._demographics.empty
            
            test_data = {
                'bedrooms': 3, 'bathrooms': 2.0, 'sqft_living': 1500,
                'sqft_lot': 5000, 'floors': 1.0, 'sqft_above': 1500,
                'sqft_basement': 0
            }
            
            try:
                self.predict_minimal(test_data)
                prediction_works = True
            except Exception:
                prediction_works = False
            
            healthy = all([model_loaded, features_loaded, demographics_loaded, prediction_works])
            
            return {
                "healthy": healthy,
                "model_loaded": model_loaded,
                "features_loaded": features_loaded,
                "demographics_loaded": demographics_loaded,
                "prediction_works": prediction_works,
                "features_count": len(self._features) if self._features else 0,
                "zipcodes_count": len(self._demographics) if self._demographics is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}