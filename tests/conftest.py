"""
Test configuration and shared fixtures.

Shared fixtures and pytest configuration for all tests.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from tests.utils import TestDataGenerator, MockFactories


@pytest.fixture
def sample_house_data():
    """Sample house data for testing."""
    return TestDataGenerator.create_house_data()


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    return TestDataGenerator.create_batch_data()


@pytest.fixture
def edge_case_house_data():
    """Sample house data with edge case values."""
    return TestDataGenerator.create_edge_case_house()


@pytest.fixture
def mock_model():
    """Mock machine learning model for testing."""
    return MockFactories.create_mock_model()


@pytest.fixture
def mock_demographics_service():
    """Mock demographics service for testing."""
    return MockFactories.create_mock_demographics_service()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "docker: mark test as requiring Docker"
    )