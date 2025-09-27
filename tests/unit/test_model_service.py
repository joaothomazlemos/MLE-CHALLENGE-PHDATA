"""
Unit tests for the ModelService class.

Tests the core model loading, prediction, and data processing functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open

from app.services.model_service import ModelService
from app.internal.exceptions import ModelLoadError


class TestModelService:
    """Test cases for ModelService class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock ML model."""
        model = Mock()
        model.predict.return_value = [450000.0]
        return model

    @pytest.fixture
    def mock_features(self):
        """Create mock model features list."""
        return [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
            'sqft_living15', 'sqft_lot15'
        ]

    @pytest.fixture
    def mock_demographics(self):
        """Create mock demographics data."""
        return pd.DataFrame({
            'zipcode': ['98103', '98115', '98117'],
            'median_income': [75000, 85000, 95000],
            'population': [45000, 50000, 55000],
            'education_bachelor_or_higher': [0.65, 0.75, 0.85]
        })

    @pytest.fixture
    def sample_house_data(self):
        """Sample house data for testing."""
        return {
            'bedrooms': 3,
            'bathrooms': 2.0,
            'sqft_living': 1800,
            'sqft_lot': 6000,
            'floors': 1.5,
            'waterfront': 0,
            'view': 0,
            'condition': 4,
            'grade': 8,
            'sqft_above': 1200,
            'sqft_basement': 600,
            'yr_built': 1990,
            'yr_renovated': 0,
            'lat': 47.6759,
            'long': -122.3289,
            'sqft_living15': 1600,
            'sqft_lot15': 8000
        }

    @patch('app.services.model_service.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.services.model_service.Path.exists')
    def test_load_model_success(self, mock_exists, mock_file, mock_pickle, mock_model):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_pickle.return_value = mock_model
        
        with patch('app.services.model_service.json.load') as mock_json, \
            patch('app.services.model_service.pd.read_csv') as mock_csv:
            mock_json.return_value = ['feature1', 'feature2']
            mock_csv.return_value = pd.DataFrame({'zipcode': ['98103']})
            
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            assert service._model == mock_model

    @patch('app.services.model_service.Path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(ModelLoadError, match="Model file not found"):
            ModelService('test.pkl', 'features.json', 'demographics.csv')

    @patch('app.services.model_service.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.services.model_service.Path.exists')
    def test_load_model_pickle_error(self, mock_exists, mock_file, mock_pickle):
        """Test model loading when pickle fails."""
        mock_exists.return_value = True
        mock_pickle.side_effect = Exception("Pickle error")
        
        with pytest.raises(ModelLoadError, match="Model loading failed"):
            ModelService('test.pkl', 'features.json', 'demographics.csv')

    def test_get_demographics_for_zipcode_found(self, mock_demographics):
        """Test getting demographics for existing zipcode."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._demographics = mock_demographics
            
            result = service.get_demographics_for_zipcode('98103')
            
            assert result is not None
            assert result['median_income'] == 75000
            assert 'zipcode' not in result

    def test_get_demographics_for_zipcode_not_found(self, mock_demographics):
        """Test getting demographics for non-existing zipcode."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._demographics = mock_demographics
            
            result = service.get_demographics_for_zipcode('99999')
            
            assert result is None

    def test_predict_success_with_demographics(self, mock_model, mock_features, mock_demographics, sample_house_data):
        """Test successful prediction with demographics."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._model = mock_model
            service._features = mock_features
            service._demographics = mock_demographics
            
            result = service.predict(sample_house_data, '98103')
            
            assert result['predicted_price'] == 450000.0
            assert result['zipcode'] == '98103'
            assert result['has_demographics'] is True
            assert result['confidence'] == 'high'

    def test_predict_success_without_demographics(self, mock_model, mock_features, mock_demographics, sample_house_data):
        """Test successful prediction without demographics."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._model = mock_model
            service._features = mock_features
            service._demographics = mock_demographics
            
            result = service.predict(sample_house_data, '99999')
            
            assert result['predicted_price'] == 450000.0
            assert result['zipcode'] == '99999'
            assert result['has_demographics'] is False
            assert result['confidence'] == 'medium'

    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._model = None
            service._features = None
            
            with pytest.raises(ModelLoadError, match="Model components not loaded"):
                service.predict({}, '98103')

    def test_predict_minimal_success(self, mock_model, mock_features):
        """Test minimal prediction success."""
        minimal_data = {
            'bedrooms': 3,
            'bathrooms': 2.0,
            'sqft_living': 1800,
            'sqft_lot': 6000,
            'floors': 1.5,
            'sqft_above': 1200,
            'sqft_basement': 600,
            'zipcode': '98103'
        }
        
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._model = mock_model
            service._features = mock_features
            
            result = service.predict_minimal(minimal_data)
            
            assert result['predicted_price'] == 450000.0
            assert result['has_demographics'] is False
            assert result['confidence'] == 'low'

    def test_health_check_healthy(self, mock_model, mock_features, mock_demographics):
        """Test health check when all components are healthy."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._model = mock_model
            service._features = mock_features
            service._demographics = mock_demographics
            
            with patch.object(service, 'predict_minimal', return_value={'predicted_price': 450000.0}):
                result = service.health_check()
                
                assert result['healthy'] is True
                assert result['model_loaded'] is True
                assert result['features_loaded'] is True
                assert result['demographics_loaded'] is True
                assert result['prediction_works'] is True

    def test_health_check_unhealthy(self):
        """Test health check when components are missing."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._model = None
            service._features = None
            service._demographics = None
            
            result = service.health_check()
            
            assert result['healthy'] is False
            assert result['model_loaded'] is False
            assert result['features_loaded'] is False
            assert result['demographics_loaded'] is False

    def test_model_features_property(self, mock_features):
        """Test model_features property."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._features = mock_features
            
            result = service.model_features
            
            assert result == mock_features
            assert result is not mock_features

    def test_available_zipcodes_property(self, mock_demographics):
        """Test available_zipcodes property."""
        with patch.object(ModelService, '_load_components'):
            service = ModelService('test.pkl', 'features.json', 'demographics.csv')
            service._demographics = mock_demographics
            
            result = service.available_zipcodes
            
            expected = ['98103', '98115', '98117']
            assert result == expected