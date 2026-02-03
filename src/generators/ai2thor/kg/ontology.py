"""
AI2-THOR specific ontology configuration.

Configures the base BaseOntology with AI2-THOR property mappings.
"""

from knowledge_graph.ontology.base import BaseOntology, RelationType
from .object_types import AI2THOR_MATERIAL_TYPES, AI2THOR_OBJECT_TYPES
from .relation_types import AI2THOR_ATTRIBUTE_TYPES, AI2THOR_STATE_RELATION_TYPES, AI2THOR_VALUE_RELATION_CUSTOM_MAPPING, AI2THOR_VALUE_RELATION_TYPES, AI2THOR_SPATIAL_RELATION_TYPES
from utils import to_namespace

def create_ai2thor_ontology() -> BaseOntology:
    """
    Configure BaseOntology with comprehensive AI2-THOR mappings.
    
    Includes spatial relationships extracted from parentReceptacles.
    """
    
    ontology = BaseOntology()  # Gets core relations (hasObject, inScene, position, rotation)
    
    # Register AI2-THOR spatial relations (from parentReceptacles)
    spatial_relations = to_namespace(AI2THOR_SPATIAL_RELATION_TYPES)
    
    for relation in spatial_relations:
        ontology.register_predicate(
            source_field=relation,
            predicate_uri=relation,
            relation_type=RelationType.SPATIAL
        )
    
    # Register AI2-THOR state predicates (both attributes and state values)
    # Capabilities/attributes (openable, pickupable, etc.)
    capability_fields = to_namespace(AI2THOR_ATTRIBUTE_TYPES)
    
    for field in capability_fields:
        ontology.register_predicate(
            source_field=field,
            predicate_uri=field,
            relation_type=RelationType.STATE
        )
    
    # States (isOpen, isMoving, etc.)
    state_fields = to_namespace(AI2THOR_STATE_RELATION_TYPES)
    
    for field in state_fields:
        ontology.register_predicate(
            source_field=field,
            predicate_uri=field,
            relation_type=RelationType.STATE
        )
        
    # Register AI2-THOR value relation properties with custom mappings
    for source, predicate in AI2THOR_VALUE_RELATION_CUSTOM_MAPPING.items():
        ontology.register_predicate(
            source_field=source,
            predicate_uri=predicate,
            relation_type=RelationType.DATA_RELATION,
        )
    
    # Register object types
    objects_types = to_namespace(AI2THOR_OBJECT_TYPES)
    for obj_type in objects_types:
        ontology.register_object_type(obj_type, obj_type)

    # Register material types
    material_types = to_namespace(AI2THOR_MATERIAL_TYPES)
    for material_type in material_types:
        ontology.register_object_type(material_type, material_type)

    return ontology
