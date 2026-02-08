from enum import StrEnum
from typing import FrozenSet

class PredicateId(StrEnum):
    # attributes
    openable = "openable"
    toggleable = "toggleable"
    pickable = "pickable"
    movable = "movable"
    receptacle = "receptacle"
    cookable = "cookable"
    sliceable = "sliceable"
    breakable = "breakable"
    dirtyable = "dirtyable"
    canFillWithLiquid = "canFillWithLiquid"
    canBeUsedUp = "canBeUsedUp"

    # state
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

    # values
    temperature = "temperature"
    mass = "mass"
    material = "material"

    # spatial
    inside = "inside"
    onTopOf = "onTopOf"
    hanging = "hanging"
    near = "near"
    locatedAt = "locatedAt"
    position = "position"
    rotation = "rotation"

    # core membership
    hasObject = "hasObject"
    inScene = "inScene"
    hasAttribute = "hasAttribute"
    type = "type"
    hasID = "hasID"


DATA_RELATIONS: FrozenSet[PredicateId] = frozenset(
    [
        PredicateId.temperature,
        PredicateId.mass,
        PredicateId.material,
        PredicateId.type,
        PredicateId.hasID,
    ]
)

ATTRIBUTE_RELATIONS: FrozenSet[PredicateId] = frozenset(
    [
        PredicateId.openable,
        PredicateId.toggleable,
        PredicateId.pickable,
        PredicateId.movable,
        PredicateId.receptacle,
        PredicateId.cookable,
        PredicateId.sliceable,
        PredicateId.breakable,
        PredicateId.dirtyable,
        PredicateId.canFillWithLiquid,
        PredicateId.canBeUsedUp,
        PredicateId.hasAttribute,
    ]
)

STATE_RELATIONS: FrozenSet[PredicateId] = frozenset(
    [
        PredicateId.isOpen,
        PredicateId.isToggled,
        PredicateId.isMoving,
        PredicateId.isPickedUp,
        PredicateId.isFilledWithLiquid,
        PredicateId.isCooked,
        PredicateId.isSliced,
        PredicateId.isBroken,
        PredicateId.isDirty,
        PredicateId.isUsedUp,
    ]
)

STATE_ATTRIBUTE_MAP: dict[PredicateId, PredicateId] = {
    PredicateId.isOpen: PredicateId.openable,
    PredicateId.isToggled: PredicateId.toggleable,
    PredicateId.isMoving: PredicateId.movable,
    PredicateId.isPickedUp: PredicateId.pickable,
    PredicateId.isFilledWithLiquid: PredicateId.canFillWithLiquid,
    PredicateId.isCooked: PredicateId.cookable,
    PredicateId.isSliced: PredicateId.sliceable,
    PredicateId.isBroken: PredicateId.breakable,
    PredicateId.isDirty: PredicateId.dirtyable,
    PredicateId.isUsedUp: PredicateId.canBeUsedUp,
}

ATTRIBUTE_STATE_MAP: dict[PredicateId, PredicateId] = {
    PredicateId.openable: PredicateId.isOpen,
    PredicateId.toggleable: PredicateId.isToggled,
    PredicateId.movable: PredicateId.isMoving,
    PredicateId.pickable: PredicateId.isPickedUp,
    PredicateId.canFillWithLiquid: PredicateId.isFilledWithLiquid,
    PredicateId.cookable: PredicateId.isCooked,
    PredicateId.sliceable: PredicateId.isSliced,
    PredicateId.breakable: PredicateId.isBroken,
    PredicateId.dirtyable: PredicateId.isDirty,
    PredicateId.canBeUsedUp: PredicateId.isUsedUp,
}

SPATIAL_RELATIONS: FrozenSet[PredicateId] = frozenset(
    [
        PredicateId.inside,
        PredicateId.onTopOf,
        PredicateId.hanging,
        PredicateId.near,
        PredicateId.locatedAt,
        PredicateId.position,
        PredicateId.rotation,
    ]
)