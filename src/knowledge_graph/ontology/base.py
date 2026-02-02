"""
Fresh ontology implementation - flexible and extensible.
"""

from dataclasses import dataclass
from typing import Dict, Any, Set, Callable, Optional
from rdflib import Graph, URIRef, Literal
from enum import Enum


class RelationType(Enum):
    """Types of relations that can be mapped."""
    OBJECT_RELATION = "object_relation"      # Object-to-object (mug isOn table)
    DATA_RELATION = "data_relation"          # Object-to-literal (mug hasColor "red")
    SPATIAL = "spatial"                      # Spatial relations
    STATE = "state"                          # State relations


@dataclass
class PredicateMapping:
    """Maps a source field to RDF predicate."""
    source_field: str
    predicate_uri: str
    relation_type: RelationType
    transform_func: Optional[Callable[[Any], Any]] = None


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
        core_predicates = [
            ("hasObject", "hasObject", RelationType.OBJECT_RELATION),    # Scene has objects
            ("inScene", "inScene", RelationType.OBJECT_RELATION),        # Object is in scene
            ("position", "position", RelationType.SPATIAL),              # Object position
            ("rotation", "rotation", RelationType.SPATIAL),              # Object rotation
        ]
        
        for source_field, predicate_uri, relation_type in core_predicates:
            self.predicate_mappings[source_field] = PredicateMapping(
                source_field=source_field,
                predicate_uri=predicate_uri,
                relation_type=relation_type,
                transform_func=None
            )
        
    def register_predicate(self, 
                          source_field: str, 
                          predicate_uri: str,
                          relation_type: RelationType = RelationType.DATA_RELATION,
                          transform_func: Optional[Callable[[Any], Any]] = None):
        """Register a predicate mapping."""
        self.predicate_mappings[source_field] = PredicateMapping(
            source_field=source_field,
            predicate_uri=predicate_uri, 
            relation_type=relation_type,
            transform_func=transform_func
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