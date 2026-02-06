"""
Fresh ontology implementation - flexible and extensible.
"""

from dataclasses import dataclass
from typing import Dict, Any, NamedTuple, Set, Callable, Optional, List
from enum import Enum


class RelationType(Enum):
    """Types of relations that can be mapped."""

    OBJECT_RELATION = "object_relation"  # Object-to-object (mug isOn table)
    DATA_RELATION = "data_relation"  # Object-to-literal (mug hasColor "red")
    SPATIAL = "spatial"  # Spatial relations
    STATE = "state"  # State relations


class CorePredicates(Enum):
    """Core predicates common to any knowledge graph."""

    HAS_OBJECT = "hasObject"
    IN_SCENE = "inScene"
    POSITION = "position"
    ROTATION = "rotation"
    OBJECT_TYPE = "rdf:type"
    OBJECT_ID = "hasID"
    LOCATED_AT = "locatedAt"


class PredicateInfo(NamedTuple):
    name: str
    type: RelationType


@dataclass
class PredicateMapping:
    """Maps a source field to RDF predicate."""

    source_field: str
    predicate_uri: str
    relation_type: RelationType
    transform_func: Optional[Callable[[Any], Any]] = None


CorePredicatesInfo: List[PredicateInfo] = [
    PredicateInfo(CorePredicates.HAS_OBJECT.value, RelationType.OBJECT_RELATION),
    PredicateInfo(CorePredicates.IN_SCENE.value, RelationType.OBJECT_RELATION),
    PredicateInfo(CorePredicates.POSITION.value, RelationType.SPATIAL),
    PredicateInfo(CorePredicates.ROTATION.value, RelationType.SPATIAL),
    PredicateInfo(CorePredicates.OBJECT_TYPE.value, RelationType.DATA_RELATION),
    PredicateInfo(CorePredicates.OBJECT_ID.value, RelationType.DATA_RELATION),
    PredicateInfo(CorePredicates.LOCATED_AT.value, RelationType.SPATIAL),
]


class BaseOntology:
    """
    Fresh, flexible ontology that can adapt to any data source.

    Key principles:
    - Start minimal, extend as needed
    - Property mappings are configurable
    - Transform functions handle data conversion
    - No assumptions about specific domains
    """

    def __init__(self):
        self.predicate_mappings: Dict[str, PredicateMapping] = {}
        self.object_type_mappings: Dict[str, str] = {}

        # Register basic structural relations that are generic for any KG
        self._register_core_predicates()

    def _register_core_predicates(self):
        """Register core structural predicates that are common to any knowledge graph."""
        for predicate_info in CorePredicatesInfo:
            self.register_predicate(
                source_field=predicate_info.name,
                predicate_uri=predicate_info.name,
                relation_type=predicate_info.type,
                transform_func=None,
            )

    def register_predicate(
        self,
        source_field: str,
        predicate_uri: str,
        relation_type: RelationType = RelationType.DATA_RELATION,
        transform_func: Optional[Callable[[Any], Any]] = None,
    ):
        """Register a predicate mapping."""
        self.predicate_mappings[source_field] = PredicateMapping(
            source_field=source_field,
            predicate_uri=predicate_uri,
            relation_type=relation_type,
            transform_func=transform_func,
        )

    def register_object_type(self, source_type: str, rdf_class: str):
        """Map source object types to RDF classes."""
        self.object_type_mappings[source_type] = rdf_class

    def get_predicate_mapping(self, field_name: str) -> Optional[PredicateMapping]:
        """Get mapping for a field, or None if not found."""
        return self.predicate_mappings.get(field_name)

    def get_object_class(self, source_type: str) -> Optional[str]:
        """Get RDF class for object type."""
        return self.object_type_mappings.get(source_type)

    def get_all_mapped_fields(self) -> Set[str]:
        """Get all fields that have mappings."""
        return set(self.predicate_mappings.keys())
    
    def get_state_predicates(self) -> Set[str]:
        """Get all predicates that are of type STATE."""
        return {
            mapping.source_field
            for mapping in self.predicate_mappings.values()
            if mapping.relation_type == RelationType.STATE
        }
    
    def get_spatial_predicates(self) -> Set[str]:
        """Get all predicates that are of type SPATIAL."""
        return {
            mapping.source_field
            for mapping in self.predicate_mappings.values()
            if mapping.relation_type == RelationType.SPATIAL
        }
    
    def get_data_predicates(self) -> Set[str]:
        """Get all predicates that are of type DATA_RELATION."""
        return {
            mapping.source_field
            for mapping in self.predicate_mappings.values()
            if mapping.relation_type == RelationType.DATA_RELATION
        }
    
    def get_object_relation_predicates(self) -> Set[str]:
        """Get all predicates that are of type OBJECT_RELATION."""
        return {
            mapping.source_field
            for mapping in self.predicate_mappings.values()
            if mapping.relation_type == RelationType.OBJECT_RELATION
        }
    
    def get_all_predicates(self) -> Set[str]:
        """Get all predicate URIs."""
        return {mapping.predicate_uri for mapping in self.predicate_mappings.values()}
    
    def is_state_predicate(self, predicate_uri: str) -> bool:
        """Check if a predicate URI is of type STATE."""
        mapping = next(
            (m for m in self.predicate_mappings.values() if m.predicate_uri == predicate_uri),
            None,
        )
        return mapping is not None and mapping.relation_type == RelationType.STATE
    
    def is_spatial_predicate(self, predicate_uri: str) -> bool:
        """Check if a predicate URI is of type SPATIAL."""
        mapping = next(
            (m for m in self.predicate_mappings.values() if m.predicate_uri == predicate_uri),
            None,
        )
        return mapping is not None and mapping.relation_type == RelationType.SPATIAL
    
    def is_data_predicate(self, predicate_uri: str) -> bool:
        """Check if a predicate URI is of type DATA_RELATION."""
        mapping = next(
            (m for m in self.predicate_mappings.values() if m.predicate_uri == predicate_uri),
            None,
        )
        return mapping is not None and mapping.relation_type == RelationType.DATA_RELATION
    
    def is_object_relation_predicate(self, predicate_uri: str) -> bool:
        """Check if a predicate URI is of type OBJECT_RELATION."""
        mapping = next(
            (m for m in self.predicate_mappings.values() if m.predicate_uri == predicate_uri),
            None,
        )
        return mapping is not None and mapping.relation_type == RelationType.OBJECT_RELATION
    
    def get_predicates_by_type(self, relation_type: RelationType) -> Set[str]:
        """Get all predicate URIs of a specific relation type."""
        return {
            mapping.predicate_uri
            for mapping in self.predicate_mappings.values()
            if mapping.relation_type == relation_type
        }
    
    def has_predicate(self, predicate_uri: str) -> bool:
        """Check if a predicate URI is registered."""
        return any(
            mapping.predicate_uri == predicate_uri
            for mapping in self.predicate_mappings.values()
        )
    
    def has_object_type(self, source_type: str) -> bool:
        """Check if an object type is registered."""
        return source_type in self.object_type_mappings
    
    def get_all_object_types(self) -> Set[str]:
        """Get all registered object types."""
        return set(self.object_type_mappings.keys())
    
    def get_all_rdf_classes(self) -> Set[str]:
        """Get all registered RDF classes."""
        return set(self.object_type_mappings.values())
