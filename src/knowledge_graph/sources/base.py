"""
Clean data source interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Iterator, Optional


@dataclass
class Relationship:
    """Relationship between two objects."""
    subject_id: str      # Object that has the relationship
    predicate: str       # Type of relationship (e.g., "isOn", "contains", "nextTo")  
    object_id: str       # Object being related to
    confidence: float = 1.0  # Optional confidence score


@dataclass
class ObjectMetadata:
    """Metadata for a single object in a scene."""
    object_id: str
    object_type: str
    properties: Dict[str, Any]          # Object properties (color, material, etc.)
    position: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None


@dataclass 
class SceneData:
    """Complete data for a scene."""
    scene_id: str
    objects: List[ObjectMetadata]
    relationships: List[Relationship] = None    # Object-to-object relationships
    metadata: Dict[str, Any] | None = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []


class DataSource(ABC):
    """Abstract interface for any data source."""
    
    @abstractmethod
    def get_scenes(self) -> Iterator[SceneData]:
        """Yield scene data one by one."""
        pass
    
    @abstractmethod
    def get_scene_by_id(self, scene_id: str) -> SceneData:
        """Get specific scene by ID."""
        pass

    @abstractmethod
    def get_available_scenes(self) -> List[str]:
        """List all available scene IDs."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources held by the data source."""
        pass