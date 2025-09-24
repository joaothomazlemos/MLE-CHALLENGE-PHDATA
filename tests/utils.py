"""
Test utilities and data generators.

Shared utility functions and classes for creating test data across all test types.
"""

import pandas as pd
import numpy as np
from unittest.mock import Mock
from typing import Dict, Any, List


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_house_data(**overrides):
        """Create house data with optional overrides."""
        base_data = {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 1800,
            "sqft_lot": 7500,
            "floors": 1.0,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "grade": 7,
            "sqft_above": 1800,
            "sqft_basement": 0,
            "yr_built": 1995,
            "yr_renovated": 0,
            "zipcode": "98103",
            "lat": 47.669,
            "long": -122.346
        }
        base_data.update(overrides)
        return base_data
    
    @staticmethod
    def create_batch_data(num_houses=2, **overrides):
        """Create batch data with specified number of houses."""
        houses = []
        for i in range(num_houses):
            house_overrides = {k: v for k, v in overrides.items() if not k.startswith('house_')}
            house_data = TestDataGenerator.create_house_data(**house_overrides)
            
            for key, value in overrides.items():
                if key.startswith(f'house_{i}_'):
                    field_name = key.replace(f'house_{i}_', '')
                    house_data[field_name] = value
            
            houses.append(house_data)
        
        return {"houses": houses}
    
    @staticmethod
    def create_edge_case_house():
        """Create house data with edge case values."""
        return {
            "bedrooms": 0,
            "bathrooms": 0.0,
            "sqft_living": 500,
            "sqft_lot": 1000,
            "floors": 1.0,
            "waterfront": 1,
            "view": 4,
            "condition": 1,
            "grade": 13,
            "sqft_above": 500,
            "sqft_basement": 0,
            "yr_built": 1900,
            "yr_renovated": 2023,
            "zipcode": "98001",
            "lat": 47.0,
            "long": -122.0
        }
    
    @staticmethod
    def create_demographics_dataframe():
        """Create sample demographics DataFrame for testing."""
        return pd.DataFrame({
            'zipcode': ['98103', '98102', '98101'],
            'median_income': [75000, 85000, 95000],
            'population': [45000, 55000, 35000],
            'education_bachelor_or_higher': [0.65, 0.72, 0.78]
        })


class MockFactories:
    """Factory methods for creating mock objects."""
    
    @staticmethod
    def create_mock_model():
        """Create a mock machine learning model."""
        model = Mock()
        model.predict.return_value = np.array([500000.0])
        model.feature_names_in_ = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long'
        ]
        return model
    
    @staticmethod
    def create_mock_demographics_service():
        """Create a mock demographics service."""
        service = Mock()
        service.get_demographics.return_value = {
            'median_income': 75000,
            'population': 45000,
            'education_bachelor_or_higher': 0.65
        }
        service.zipcode_exists.return_value = True
        return service


class MockHTTPResponse:
    """Mock HTTP response for testing external API calls."""
    
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def skip_if_no_docker():
    """Skip test if Docker is not available."""
    import subprocess
    try:
        subprocess.run(['docker', '--version'], 
                      check=True, 
                      capture_output=True, 
                      text=True)
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


VALID_ZIPCODES = ['98103', '98102', '98101', '98109', '98112']
INVALID_ZIPCODES = ['00000', '99999', 'invalid']

FEATURE_RANGES = {
    'bedrooms': (0, 10),
    'bathrooms': (0.0, 8.0),
    'sqft_living': (300, 10000),
    'sqft_lot': (500, 50000),
    'floors': (1.0, 3.5),
    'waterfront': (0, 1),
    'view': (0, 4),
    'condition': (1, 5),
    'grade': (1, 13),
    'yr_built': (1900, 2023),
    'yr_renovated': (0, 2023)
}