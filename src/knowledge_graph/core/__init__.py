"""Core knowledge graph functionality."""

from .builder import KnowledgeGraphBuilder, BuildResult
from .namespaces import (
    NamespaceManager,
    NamespaceConfig,
    create_entity_uri,
    create_relation_uri,
    create_scene_uri,
    create_attribute_uri,
    create_value_uri,
)

__all__ = [
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
