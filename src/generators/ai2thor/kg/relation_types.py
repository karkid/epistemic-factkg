from enum import Enum
from typing import FrozenSet

class RelationType(Enum):
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


    # State relation types specific to AI2-THOR
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

    # Value relation types can be added here as needed
    temperature = "hasTemperature"
    mass = "hasMass"
    material = "hasMaterial"

    # Spatial relation types specific to AI2-THOR
    inside = "inside"
    onTopOf = "onTopOf"
    hanging = "hanging"
    near = "near"

# AI2-THOR specific attribute type sets
AI2THOR_ATTRIBUTE_TYPES: FrozenSet[RelationType] = frozenset([
    RelationType.openable,
    RelationType.togglable,
    RelationType.pickable,
    RelationType.movable,
    RelationType.receptacle,
    RelationType.cookable,
    RelationType.sliceable,
    RelationType.breakable,
    RelationType.dirtyable,
    RelationType.canFillWithLiquid,
    RelationType.canBeUsedUp,
])



# AI2-THOR specific state relation names
AI2THOR_STATE_RELATION_TYPES: FrozenSet[RelationType] = frozenset([
    RelationType.isOpen,
    RelationType.isToggled,
    RelationType.isMoving,
    RelationType.isPickedUp,
    RelationType.isFilledWithLiquid,
    RelationType.isCooked,
    RelationType.isSliced,
    RelationType.isBroken,
    RelationType.isDirty,
    RelationType.isUsedUp,
])

AI2THOR_ATTRIBUTE_STATE_MAPPING = {
    RelationType.openable: RelationType.isOpen,
    RelationType.togglable: RelationType.isToggled,
    RelationType.pickable: RelationType.isPickedUp,
    RelationType.movable: RelationType.isMoving,
    RelationType.canFillWithLiquid: RelationType.isFilledWithLiquid,
    RelationType.cookable: RelationType.isCooked,
    RelationType.sliceable: RelationType.isSliced,
    RelationType.breakable: RelationType.isBroken,
    RelationType.dirtyable: RelationType.isDirty,
    RelationType.canBeUsedUp: RelationType.isUsedUp,
}

# AI2-THOR specific value relation names
AI2THOR_VALUE_RELATION_TYPES: FrozenSet[RelationType] = frozenset([
    RelationType.temperature,
    RelationType.mass,
    RelationType.material,
])

AI2THOR_VALUE_RELATION_CUSTOM_MAPPING = {
    "temperature": RelationType.temperature.value,
    "mass": RelationType.mass.value,
    "salientMaterials": RelationType.material.value,
}

# AI2-THOR specific spatial relation names
AI2THOR_SPATIAL_RELATION_TYPES: FrozenSet[RelationType] = frozenset([
    RelationType.inside,
    RelationType.onTopOf,
    #RelationType.hanging,
    #RelationType.near,
])