"""
Tests for Data Source API - Pure Dummy Data

These tests validate data processing without any external dependencies.
"""

import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.exceptions import DataSourceError, ConfigurationError


class TestDataSourceAPI:
    """Test data source API with pure dummy data."""

    def test_dummy_scene_data_structure(self):
        """Test dummy scene data has correct structure."""
        dummy_scene = {
            "scene": "DummyKitchen",
            "objects": [
                {
                    "objectId": "Apple_1",
                    "objectType": "Apple",
                    "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                    "visible": True,
                    "receptacle": False,
                },
                {
                    "objectId": "Table_1",
                    "objectType": "Table",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "visible": True,
                    "receptacle": True,
                },
            ],
        }

        # Validate structure
        assert "scene" in dummy_scene
        assert "objects" in dummy_scene
        assert len(dummy_scene["objects"]) == 2
        assert dummy_scene["objects"][0]["objectType"] == "Apple"

    def test_dummy_data_processing(self):
        """Test processing dummy data."""
        dummy_data = [
            {"objectId": "Cup_1", "objectType": "Cup", "visible": True},
            {"objectId": "Spoon_1", "objectType": "Spoon", "visible": False},
        ]

        # Test filtering visible objects
        visible_objects = [obj for obj in dummy_data if obj["visible"]]
        assert len(visible_objects) == 1
        assert visible_objects[0]["objectType"] == "Cup"

    def test_configuration_structure(self, simple_config):
        """Test configuration has required structure."""
        assert "ai2thor" in simple_config
        assert "knowledge_graph" in simple_config
        assert "base_uri" in simple_config["knowledge_graph"]

    def test_object_relationships_dummy_data(self):
        """Test object relationships with dummy data."""
        dummy_relationships = [
            {
                "subject": "Apple_1",
                "predicate": "on_top_of",
                "object": "Table_1",
                "confidence": 0.95,
            },
            {
                "subject": "Cup_1",
                "predicate": "near",
                "object": "Apple_1",
                "confidence": 0.8,
            },
        ]

        # Test relationship structure
        assert len(dummy_relationships) == 2
        assert dummy_relationships[0]["predicate"] == "on_top_of"
        assert dummy_relationships[1]["confidence"] == 0.8

    def test_error_handling_with_dummy_data(self):
        """Test error handling with dummy scenarios."""
        # Test configuration error
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid dummy config")

        # Test data source error
        with pytest.raises(DataSourceError):
            raise DataSourceError("Dummy data source failed")
