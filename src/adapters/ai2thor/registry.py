from __future__ import annotations

from src.core.registry import (
    EntityInfo,
    EntityRegistry,
    SpatialRole,
    RelationInfo,
    RelationRegistry,
    RelationType,
)
from src.adapters.ai2thor.ids.object_types import (
    ObjectType,
    CONTAINER_ROLES,
    SURFACE_ROLES,
    HANGING_ROLES,
)
from src.adapters.ai2thor.ids.predicates import (
    PredicateId,
    ATTRIBUTE_RELATIONS,
    STATE_RELATIONS,
    SPATIAL_RELATIONS,
    DATA_RELATIONS,
)


def create_entity_registry(registry: EntityRegistry) -> None:
    for obj in ObjectType:
        if obj in CONTAINER_ROLES:
            role = SpatialRole.CONTAINER
        elif obj in SURFACE_ROLES:
            role = SpatialRole.SURFACE
        elif obj in HANGING_ROLES:
            role = SpatialRole.HANGING
        else:
            role = None

        registry.register(
            obj.value,
            EntityInfo(name=obj.value, type="object", spatial_roles=role),
        )


def create_relation_registry(registry: RelationRegistry) -> None:
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
            info=RelationInfo(name=r.value, type=relation_type),
        )
