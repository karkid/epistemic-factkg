from __future__ import annotations

from enum import StrEnum
from typing import FrozenSet

from src.core.registry.entity import EntityInfo, EntityRegistry, SpatialRole
from src.adapters.ai2thor.ids.object_types import ObjectType, CONTAINER_ROLES, SURFACE_ROLES, HANGING_ROLES


def create_entity_registry(registry: EntityRegistry) -> None:
    """
    Populate a core EntityRegistry with AI2-THOR object vocabulary.
    """
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
            EntityInfo(
                name=obj.value,
                type="object",
                spatial_roles=role,
            ),
        )
