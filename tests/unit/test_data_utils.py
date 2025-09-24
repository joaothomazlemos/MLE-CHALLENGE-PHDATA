"""
Unit tests for data utility functions.

Tests data processing, validation, and transformation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from app.internal.data_utils import (
    ensure_numeric_types,
    fill_missing_demographics,
    log_prediction_request
)
from tests.utils import TestDataGenerator, VALID_ZIPCODES


class TestDataUtils:
    """Test cases for data utility functions."""

    def test_ensure_numeric_types_valid_data(self):
        """Test numeric type conversion with valid data."""
        input_data = {
            'bedrooms': 3,
            'bathrooms': 2.5,
            'sqft_living': 1800,
            'floors': 1.0,
            'zipcode': '98103'
        }
        
        result = ensure_numeric_types(input_data)
        
        assert result['bedrooms'] == 3.0
        assert result['bathrooms'] == 2.5
        assert result['sqft_living'] == 1800.0
        assert result['floors'] == 1.0
        assert result['zipcode'] == '98103'

    def test_ensure_numeric_types_invalid_data(self):
        """Test numeric type conversion with invalid data."""
        input_data = {
            'bedrooms': 'invalid',
            'bathrooms': 2.5,
            'zipcode': '98103'
        }
        
        result = ensure_numeric_types(input_data)
        
        assert result['bedrooms'] == 'invalid'
        assert result['bathrooms'] == 2.5
        assert result['zipcode'] == '98103'

    def test_ensure_numeric_types_none_values(self):
        """Test numeric type conversion with None values."""
        input_data = {
            'bedrooms': None,
            'bathrooms': 2.5,
            'sqft_living': 1800
        }
        
        result = ensure_numeric_types(input_data)
        
        assert result['bedrooms'] is None
        assert result['bathrooms'] == 2.5
        assert result['sqft_living'] == 1800.0

    def test_fill_missing_demographics_all_present(self):
        """Test filling missing demographics when all features are present."""
        input_data = {
            'bedrooms': 3,
            'bathrooms': 2.0,
            'sqft_living': 1800,
            'median_income': 75000,
            'population': 45000
        }
        
        required_features = ['bedrooms', 'bathrooms', 'sqft_living', 'median_income', 'population']
        
        result = fill_missing_demographics(input_data, required_features)
        
        assert result == input_data

    def test_fill_missing_demographics_missing_features(self):
        """Test filling missing demographics when some features are missing."""
        input_data = {
            'bedrooms': 3,
            'bathrooms': 2.0,
            'sqft_living': 1800
        }
        
        required_features = [
            'bedrooms', 'bathrooms', 'sqft_living', 
            'median_income', 'population', 'education_bachelor_or_higher'
        ]
        
        result = fill_missing_demographics(input_data, required_features)
        
        assert all(feature in result for feature in required_features)
        
        assert result['bedrooms'] == 3
        assert result['bathrooms'] == 2.0
        assert result['sqft_living'] == 1800
        
        assert 'median_income' in result
        assert 'population' in result
        assert 'education_bachelor_or_higher' in result

    def test_fill_missing_demographics_empty_input(self):
        """Test filling missing demographics with empty input."""
        input_data = {}
        required_features = ['bedrooms', 'bathrooms', 'sqft_living', 'median_income']
        
        result = fill_missing_demographics(input_data, required_features)
        
        assert len(result) == len(required_features)
        assert all(feature in result for feature in required_features)

    @patch('app.internal.data_utils.logger')
    def test_log_prediction_request_with_demographics(self, mock_logger):
        """Test logging prediction request with demographics."""
        log_prediction_request('98103', True)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert '98103' in call_args
        assert 'available' in call_args.lower()

    @patch('app.internal.data_utils.logger')
    def test_log_prediction_request_without_demographics(self, mock_logger):
        """Test logging prediction request without demographics."""
        log_prediction_request('99999', False)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert '99999' in call_args
        assert 'missing' in call_args.lower()

    def test_fill_missing_demographics_numeric_defaults(self):
        """Test that missing demographics get reasonable numeric defaults."""
        input_data = {'bedrooms': 3}
        required_features = ['bedrooms', 'median_income', 'population', 'education_bachelor_or_higher']
        
        result = fill_missing_demographics(input_data, required_features)
        
        assert isinstance(result['median_income'], (int, float))
        assert isinstance(result['population'], (int, float))
        assert isinstance(result['education_bachelor_or_higher'], (int, float))
        
        assert result['median_income'] == 0.0
        assert result['population'] == 0.0  
        assert result['education_bachelor_or_higher'] == 0.0

    def test_fill_missing_demographics_preserves_types(self):
        """Test that fill_missing_demographics preserves original data types."""
        input_data = {
            'bedrooms': 3,
            'bathrooms': 2.5,
            'sqft_living': np.int64(1800),
            'zipcode': '98103'
        }
        
        required_features = list(input_data.keys()) + ['median_income']
        
        result = fill_missing_demographics(input_data, required_features)
        
        assert isinstance(result['bedrooms'], int)
        assert isinstance(result['bathrooms'], float)
        assert isinstance(result['sqft_living'], (int, np.integer))
        assert isinstance(result['zipcode'], str)

    def test_ensure_numeric_types_with_scientific_notation(self):
        """Test numeric conversion with scientific notation."""
        input_data = {
            'sqft_living': 1.8e3,
            'price': 4.5e5,
        }
        
        result = ensure_numeric_types(input_data)
        
        assert result['sqft_living'] == 1800.0
        assert result['price'] == 450000.0

    def test_ensure_numeric_types_with_negative_values(self):
        """Test numeric conversion with negative values."""
        input_data = {
            'yr_renovated': 0,
            'sqft_basement': -1,
        }
        
        result = ensure_numeric_types(input_data)
        
        assert result['yr_renovated'] == 0.0
        assert result['sqft_basement'] == -1.0
