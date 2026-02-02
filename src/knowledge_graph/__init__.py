"""
Fresh, modular knowledge graph construction library.

Core Components:
- DataSource: Interface for any data source
- BaseOntology: Configurable property mappings
- KnowledgeGraphBuilder: Main graph construction engine
"""

from .sources.base import DataSource, SceneData, ObjectMetadata, Relationship
from .ontology.base import BaseOntology, PredicateMapping, RelationType
from .core.builder import KnowledgeGraphBuilder, BuildResult
from .core.namespaces import create_entity_uri, create_relation_uri, create_scene_uri

__all__ = [
    "DataSource", "SceneData", "ObjectMetadata", "Relationship",
    "BaseOntology", "PredicateMapping", "RelationType",
    "KnowledgeGraphBuilder", "BuildResult",
    "create_entity_uri", "create_relation_uri", "create_scene_uri"
]