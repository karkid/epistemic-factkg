"""
Fresh, modular knowledge graph construction library.

Core Components:
- DataSource: Interface for any data source
- BaseOntology: Configurable property mappings
- KnowledgeGraphBuilder: Main graph construction engine
- NamespaceManager: Configurable namespace management
"""

from .sources.base import DataSource, SceneData, ObjectMetadata, Relationship
from .ontology.base import BaseOntology, PredicateMapping, RelationType
from .core.knowledge_graph_builder import KnowledgeGraphBuilder, BuildResult
from .core.namespace_manager import (
    NamespaceManager,
    NamespaceConfig,
    create_entity_uri,
    create_relation_uri,
    create_scene_uri,
    create_attribute_uri,
    create_value_uri,
)

__all__ = [
    "DataSource",
    "SceneData",
    "ObjectMetadata",
    "Relationship",
    "BaseOntology",
    "PredicateMapping",
    "RelationType",
    "KnowledgeGraphBuilder",
    "BuildResult",
    "NamespaceManager",
    "NamespaceConfig",
    "create_entity_uri",
    "create_relation_uri",
    "create_scene_uri",
    "create_attribute_uri",
    "create_value_uri",
]
