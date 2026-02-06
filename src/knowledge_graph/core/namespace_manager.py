"""
Configurable namespace management for RDF generation.

This module provides configurable namespace management to support
different knowledge graph schemas and URI patterns.
"""

from dataclasses import dataclass
from typing import Optional
from rdflib import Namespace, URIRef, RDF
from urllib.parse import quote


@dataclass(frozen=True)
class NamespaceConfig:
    """Configuration for RDF namespaces used in knowledge graph generation."""

    base_iri: str = "http://epistemicfactkg.org/"
    entities_suffix: str = "entities/"
    relations_suffix: str = "relations/"
    attributes_suffix: str = "attributes/"
    scenes_suffix: str = "scenes/"
    values_suffix: str = "values/"

    def __post_init__(self):
        """Ensure base IRI ends with slash."""
        if not self.base_iri.endswith("/"):
            object.__setattr__(self, "base_iri", self.base_iri + "/")

    @property
    def base(self) -> Namespace:
        """Base namespace."""
        return Namespace(self.base_iri)

    @property
    def entities(self) -> Namespace:
        """Entities namespace."""
        return Namespace(self.base_iri + self.entities_suffix)

    @property
    def relations(self) -> Namespace:
        """Relations namespace."""
        return Namespace(self.base_iri + self.relations_suffix)

    @property
    def attributes(self) -> Namespace:
        """Attributes namespace."""
        return Namespace(self.base_iri + self.attributes_suffix)

    @property
    def scenes(self) -> Namespace:
        """Scenes namespace."""
        return Namespace(self.base_iri + self.scenes_suffix)

    @property
    def values(self) -> Namespace:
        """Values namespace."""
        return Namespace(self.base_iri + self.values_suffix)

    @property
    def type_predicate(self) -> URIRef:
        """RDF type predicate."""
        return RDF.type


def safe_fragment(text: str) -> str:
    """
    Convert arbitrary strings to URI-safe fragments by URL encoding.

    Args:
        text: The text to make URI-safe

    Returns:
        URI-safe fragment
    """
    if hasattr(text, "value"):  # Handle Enum values
        text = text.value
    return quote(str(text), safe="-_.~")


def create_uri(identifier: str, namespace: Namespace) -> URIRef:
    """
    Create a URIRef in the given namespace with a safe fragment.

    Args:
        identifier: The identifier to create a URI for
        namespace: The namespace to use

    Returns:
        A properly formed URIRef
    """
    safe_id = safe_fragment(identifier)
    return namespace[safe_id]


class NamespaceManager:
    """Manages namespaces for RDF graph generation with configurable settings."""

    def __init__(self, config: Optional[NamespaceConfig] = None):
        """
        Initialize namespace manager.

        Args:
            config: Optional namespace configuration. Uses defaults if None.
        """
        self.config = config or NamespaceConfig()

    def create_entity_uri(self, identifier: str) -> URIRef:
        """Create URI for an entity (object, scene, etc.)."""
        return create_uri(identifier, self.config.entities)

    def create_relation_uri(self, identifier: str) -> URIRef:
        """Create URI for a relation/predicate."""
        return create_uri(identifier, self.config.relations)

    def create_attribute_uri(self, identifier: str) -> URIRef:
        """Create URI for an attribute."""
        return create_uri(identifier, self.config.attributes)

    def create_scene_uri(self, identifier: str) -> URIRef:
        """Create URI for a scene."""
        return create_uri(identifier, self.config.scenes)

    def create_value_uri(self, identifier: str) -> URIRef:
        """Create URI for a value."""
        return create_uri(identifier, self.config.values)

    def get_type_predicate(self) -> URIRef:
        """Get the RDF type predicate."""
        return self.config.type_predicate


# Default instance for easy access
DEFAULT_NAMESPACE_CONFIG = NamespaceConfig()
DEFAULT_NAMESPACE_MANAGER = NamespaceManager(DEFAULT_NAMESPACE_CONFIG)


# Main functions using default manager (primary interface)
def create_entity_uri(identifier: str) -> URIRef:
    """Create URI for an entity using default configuration."""
    return DEFAULT_NAMESPACE_MANAGER.create_entity_uri(identifier)


def create_relation_uri(identifier: str) -> URIRef:
    """Create URI for a relation using default configuration."""
    return DEFAULT_NAMESPACE_MANAGER.create_relation_uri(identifier)


def create_scene_uri(identifier: str) -> URIRef:
    """Create URI for a scene using default configuration."""
    return DEFAULT_NAMESPACE_MANAGER.create_scene_uri(identifier)


def create_attribute_uri(identifier: str) -> URIRef:
    """Create URI for an attribute using default configuration."""
    return DEFAULT_NAMESPACE_MANAGER.create_attribute_uri(identifier)


def create_value_uri(identifier: str) -> URIRef:
    """Create URI for a value using default configuration."""
    return DEFAULT_NAMESPACE_MANAGER.create_value_uri(identifier)
