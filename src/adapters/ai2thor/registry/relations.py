# src/adapters/ai2thor/registry/relations.py
from __future__ import annotations


from src.core.registry.relation import RelationInfo, RelationRegistry, RelationType
from src.adapters.ai2thor.ids.predicates import (
    PredicateId,
    ATTRIBUTE_RELATIONS,
    STATE_RELATIONS,
    SPATIAL_RELATIONS,
    DATA_RELATIONS,
)


def create_relation_registry(registry: RelationRegistry) -> None:
    """
    Populate a core RelationRegistry with AI2-THOR relation vocabulary.
    """

    for r in PredicateId:
        if r in ATTRIBUTE_RELATIONS:
            relation_type = RelationType.ATTRIBUTE
        elif r in STATE_RELATIONS:
            relation_type = RelationType.STATE
        elif r in SPATIAL_RELATIONS:
            relation_type = RelationType.SPATIAL
        elif r in DATA_RELATIONS:
            relation_type = RelationType.DATA
        else:
            relation_type = RelationType.OBJECT

        registry.register(
            rid=r.value,
            info=RelationInfo(
                name=r.value,
                type=relation_type,
            ),
        )
