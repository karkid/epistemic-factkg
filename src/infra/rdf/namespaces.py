"""
Configurable namespace management for RDF generation.

This module provides configurable namespace management to support
different knowledge graph schemas and URI patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

from rdflib import Namespace, URIRef, RDF


@dataclass(frozen=True)
class NamespaceConfig:
    """Configuration for RDF namespaces used in knowledge graph generation."""

    base_iri: str = "http://epistemicfactkg.org/"
    entities_suffix: str = "entities/"
    relations_suffix: str = "relations/"
    attributes_suffix: str = "attributes/"
    contexts_suffix: str = "contexts/"
    values_suffix: str = "values/"

    def __post_init__(self) -> None:
        if not self.base_iri.endswith("/"):
            object.__setattr__(self, "base_iri", self.base_iri + "/")

    @property
    def base(self) -> Namespace:
        return Namespace(self.base_iri)

    @property
    def entities(self) -> Namespace:
        return Namespace(self.base_iri + self.entities_suffix)

    @property
    def relations(self) -> Namespace:
        return Namespace(self.base_iri + self.relations_suffix)

    @property
    def attributes(self) -> Namespace:
        return Namespace(self.base_iri + self.attributes_suffix)

    @property
    def contexts(self) -> Namespace:
        return Namespace(self.base_iri + self.contexts_suffix)

    @property
    def values(self) -> Namespace:
        return Namespace(self.base_iri + self.values_suffix)

    @property
    def type_predicate(self) -> URIRef:
        return RDF.type


def safe_fragment(text: object) -> str:
    """Convert arbitrary values to URI-safe fragments by URL encoding."""
    if hasattr(text, "value"):  # Enum-like
        text = getattr(text, "value")
    return quote(str(text), safe="-_.~")


def create_uri(identifier: str, namespace: Namespace) -> URIRef:
    """Create a URIRef in the given namespace with a safe fragment."""
    return namespace[safe_fragment(identifier)]


class NamespaceManager:
    """Manages namespaces for RDF graph generation with configurable settings."""

    def __init__(self, config: Optional[NamespaceConfig] = None):
        self.config = config or NamespaceConfig()

    def entity_uri(self, identifier: str) -> URIRef:
        return create_uri(identifier, self.config.entities)

    def relation_uri(self, identifier: str) -> URIRef:
        return create_uri(identifier, self.config.relations)

    def attribute_uri(self, identifier: str) -> URIRef:
        return create_uri(identifier, self.config.attributes)

    def context_uri(self, identifier: str) -> URIRef:
        return create_uri(identifier, self.config.contexts)

    def value_uri(self, identifier: str) -> URIRef:
        return create_uri(identifier, self.config.values)

    @property
    def type_predicate(self) -> URIRef:
        return self.config.type_predicate


# Convenience factory (no globals)
def get_namespace_manager(config: Optional[NamespaceConfig] = None) -> NamespaceManager:
    """Create a NamespaceManager using the provided config (or defaults)."""
    return NamespaceManager(config)
