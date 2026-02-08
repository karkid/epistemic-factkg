"""
Lightweight Test Configuration - No Real AI2THOR Scenes

Simple mocks that don't load actual scenes for fast testing.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def simple_config():
    """Create a simple test configuration."""
    return {
        "ai2thor": {
            "environment": "TestScene",
            "grid_size": 0.25
        },
        "knowledge_graph": {
            "base_uri": "http://test.example.org/"
        }
    }


@pytest.fixture
def lightweight_mock_data_source():
    """Create a lightweight mock data source - no real scenes loaded."""
    mock_source = Mock()
    
    # Simple test data - completely mocked
    mock_source.get_scene_data.return_value = {
        "scene": "MockScene",
        "objects": [{"objectId": "Apple_1", "objectType": "Apple"}]
    }
    mock_source.scene_names = ["MockScene"]  # Only one mock scene
    mock_source.validate.return_value = True
    
    return mock_source


@pytest.fixture
def mock_controller():
    """Create a mock controller that never loads real scenes."""
    mock = Mock()
    mock.start.return_value = None
    mock.stop.return_value = None
    mock.step.return_value = {"success": True, "objects": []}
    return mock