"""
AI2-THOR specific ontology configuration.

Configures the base BaseOntology with AI2-THOR property mappings.
"""

from knowledge_graph.ontology.base import BaseOntology, RelationType
from .object_types import AI2THOR_MATERIAL_TYPES, AI2THOR_OBJECT_TYPES
from .relation_types import AI2THOR_STATE_RELATION_TYPES, AI2THOR_VALUE_RELATION_TYPES, AI2THOR_SPATIAL_RELATION_TYPES

def create_ai2thor_ontology() -> BaseOntology:
    """
    Configure BaseOntology with comprehensive AI2-THOR mappings.
    
    Includes spatial relationships extracted from parentReceptacles.
    """
    
    ontology = BaseOntology()  # Gets core relations (hasObject, inScene, position, rotation)
    
    # Register AI2-THOR spatial relations (from parentReceptacles)
    spatial_relations = ['inside', 'onTopOf', 'hanging', 'near']
    
    for relation in spatial_relations:
        ontology.register_predicate(
            source_field=relation,
            predicate_uri=relation,
            relation_type=RelationType.SPATIAL
        )
    
    # Register AI2-THOR state predicates (both attributes and state values)
    # Capabilities/attributes (openable, pickupable, etc.)
    capability_fields = ['openable', 'togglable', 'pickupable', 'moveable', 'receptacle', 
                        'cookable', 'sliceable', 'breakable', 'dirtyable', 'canFillWithLiquid', 
                        'canBeUsedUp']
    
    for field in capability_fields:
        ontology.register_predicate(
            source_field=field,
            predicate_uri=field,
            relation_type=RelationType.STATE
        )
    
    # States (isOpen, isMoving, etc.)
    state_fields = ['isOpen', 'isToggled', 'isMoving', 'isPickedUp', 'isFilledWithLiquid',
                   'isCooked', 'isSliced', 'isBroken', 'isDirty', 'isUsedUp']
    
    for field in state_fields:
        ontology.register_predicate(
            source_field=field,
            predicate_uri=field,
            relation_type=RelationType.STATE
        )
        
    # Register AI2-THOR value relation properties with custom mappings
    # Temperature: source_field="temperature" -> predicate="hasTemperature" 
    ontology.register_predicate(
        source_field="temperature",
        predicate_uri="hasTemperature", 
        relation_type=RelationType.DATA_RELATION,
    )
    
    # Mass: source_field="mass" -> predicate="hasMass"
    ontology.register_predicate(
        source_field="mass",
        predicate_uri="hasMass",
        relation_type=RelationType.DATA_RELATION,
        transform_func=lambda mass: round(float(mass), 3) if mass is not None else None
    )
    
    # SalientMaterials: source_field="salientMaterials" -> predicate="hasMaterial"
    ontology.register_predicate(
        source_field="salientMaterials", 
        predicate_uri="hasMaterial",
        relation_type=RelationType.DATA_RELATION,
    )
    
    # Register object types
    for obj_type in AI2THOR_OBJECT_TYPES:
        ontology.register_object_type(obj_type, obj_type)

    return ontology
