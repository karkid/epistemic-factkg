# src/adapters/ai2thor/registry/relations.py
from __future__ import annotations

from enum import StrEnum
from typing import FrozenSet

from src.core.registry.relation import RelationInfo, RelationRegistry, RelationType


class Relations(StrEnum):
    # Attribute relation types
    openable = "openable"
    togglable = "togglable"
    pickable = "pickable"
    movable = "movable"
    receptacle = "receptacle"
    cookable = "cookable"
    sliceable = "sliceable"
    breakable = "breakable"
    dirtyable = "dirtyable"
    canFillWithLiquid = "canFillWithLiquid"
    canBeUsedUp = "canBeUsedUp"

    # State relation types
    isOpen = "isOpen"
    isToggled = "isToggled"
    isMoving = "isMoving"
    isPickedUp = "isPickedUp"
    isFilledWithLiquid = "isFilledWithLiquid"
    isCooked = "isCooked"
    isSliced = "isSliced"
    isBroken = "isBroken"
    isDirty = "isDirty"
    isUsedUp = "isUsedUp"

    # Data relations (values)
    temperature = "hasTemperature"
    mass = "hasMass"
    material = "hasMaterial"

    # Spatial relations
    inside = "inside"
    onTopOf = "onTopOf"
    hanging = "hanging"
    near = "near"


VALUE_RELATIONS: FrozenSet[Relations] = frozenset(
    [
        Relations.temperature,
        Relations.mass,
        Relations.material,
    ]
)

ATTRIBUTE_RELATIONS: FrozenSet[Relations] = frozenset(
    [
        Relations.openable,
        Relations.togglable,
        Relations.pickable,
        Relations.movable,
        Relations.receptacle,
        Relations.cookable,
        Relations.sliceable,
        Relations.breakable,
        Relations.dirtyable,
        Relations.canFillWithLiquid,
        Relations.canBeUsedUp,
    ]
)

STATE_RELATIONS: FrozenSet[Relations] = frozenset(
    [
        Relations.isOpen,
        Relations.isToggled,
        Relations.isMoving,
        Relations.isPickedUp,
        Relations.isFilledWithLiquid,
        Relations.isCooked,
        Relations.isSliced,
        Relations.isBroken,
        Relations.isDirty,
        Relations.isUsedUp,
    ]
)

SPATIAL_RELATIONS: FrozenSet[Relations] = frozenset(
    [
        Relations.inside,
        Relations.onTopOf,
        Relations.hanging,
        Relations.near,
    ]
)

def _register_common_relation_info(registory: RelationRegistry) -> None:

    registory.register("hasObject", RelationInfo(name="hasObject", type=RelationType.OBJECT))
    registory.register("inScene", RelationInfo(name="inScene", type=RelationType.OBJECT))

    registory.register("hasAttribute", RelationInfo(name="hasAttribute", type=RelationType.ATTRIBUTE))

    registory.register("type", RelationInfo(name="type", type=RelationType.DATA))
    registory.register("hasID", RelationInfo(name="hasID", type=RelationType.DATA))

    registory.register("locatedAt", RelationInfo(name="locatedAt", type=RelationType.SPATIAL))
    registory.register("position", RelationInfo(name="position", type=RelationType.SPATIAL))
    registory.register("rotation", RelationInfo(name="rotation", type=RelationType.SPATIAL))


def create_relation_registry(registry: RelationRegistry) -> None:
    """
    Populate a core RelationRegistry with AI2-THOR relation vocabulary.
    """
    _register_common_relation_info(registry)
    
    for r in Relations:
        if r in ATTRIBUTE_RELATIONS:
            relation_type = RelationType.ATTRIBUTE
        elif r in STATE_RELATIONS:
            relation_type = RelationType.STATE
        elif r in SPATIAL_RELATIONS:
            relation_type = RelationType.SPATIAL
        elif r in VALUE_RELATIONS:
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
