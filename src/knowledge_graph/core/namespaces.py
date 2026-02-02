"""
Clean RDF namespace definitions.
"""

from dataclasses import dataclass
from urllib.parse import quote
from rdflib import Namespace, URIRef, RDF

# Base namespace
BASE_IRI = "http://epistemicfactkg.org/"
BASE = Namespace(BASE_IRI)

# Core namespaces
ENTITIES = Namespace(BASE + "entities/")
RELATIONS = Namespace(BASE + "relations/")
ATTRIBUTES = Namespace(BASE + "attributes/")  
SCENES = Namespace(BASE + "scenes/")
VALUES = Namespace(BASE + "values/")
TYPE: URIRef = RDF.type

# ----------------------------
# Data Classes for Namespaces
# ----------------------------
@dataclass(frozen=True)
class Namespaces:
    """Data class to hold namespace definitions."""

    base: Namespace = BASE
    entities: Namespace = ENTITIES
    attributes: Namespace = ATTRIBUTES
    relations: Namespace = RELATIONS
    scenes: Namespace = SCENES
    values: Namespace = VALUES
    type: URIRef = TYPE
# Instance of Namespaces for easy access
NAMESPACES = Namespaces()

def safe_fragment(text: str) -> str:
    """
    Convert arbitrary strings to URI-safe fragments by URL encoding.
    """
    return quote(text, safe='-_.~')

def create_uri(identifier: str, namespace: Namespace) -> URIRef:
    """Create a URIRef in the given namespace with a safe fragment."""
    safe_id = safe_fragment(identifier)
    return namespace[safe_id]

def create_entity_uri(identifier: str) -> URIRef:
    """Create URI for an entity (object, scene, etc.)."""
    return create_uri(identifier, ENTITIES)

def create_relation_uri(relation_name: str) -> URIRef:
    """Create URI for a relation/property."""
    return create_uri(relation_name, RELATIONS)

def create_scene_uri(scene_id: str) -> URIRef:
    """Create URI for a scene."""
    return create_uri(scene_id, SCENES)
