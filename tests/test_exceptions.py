"""
Simple test for exceptions module to validate basic functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.exceptions import (
    EpistemicFactKGError,
    ConfigurationError,
    DataSourceError,
    BuildError,
    ValidationError,
)


def test_base_exception():
    """Test base exception works."""
    error = EpistemicFactKGError("Test error")
    assert str(error) == "Test error"


def test_configuration_error():
    """Test configuration error."""
    error = ConfigurationError("Config failed")
    assert str(error) == "Config failed"
    assert isinstance(error, EpistemicFactKGError)


def test_data_source_error():
    """Test data source error."""
    error = DataSourceError("Data failed")
    assert str(error) == "Data failed"
    assert isinstance(error, EpistemicFactKGError)


def test_build_error():
    """Test build error."""
    error = BuildError("Build failed")
    assert str(error) == "Build failed"
    assert isinstance(error, EpistemicFactKGError)


def test_validation_error():
    """Test validation error."""
    error = ValidationError("Validation failed")
    assert str(error) == "Validation failed"
    assert isinstance(error, EpistemicFactKGError)
