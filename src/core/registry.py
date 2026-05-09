from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Set


class SpatialRole(Enum):
    SURFACE = auto()
    CONTAINER = auto()
    HANGING = auto()


@dataclass(frozen=True)
class EntityInfo:
    name: str
    type: str | None = None
    spatial_roles: Set[SpatialRole] | None = None


class EntityRegistry:
    def __init__(self):
        self._entities: Dict[str, EntityInfo] = {}

    def register(self, eid: str, info: EntityInfo):
        self._entities[eid] = info

    def get(self, eid: str) -> EntityInfo | None:
        return self._entities.get(eid)

    def get_all(self) -> Dict[str, EntityInfo]:
        return self._entities


class RelationType(Enum):
    OBJECT = "object"
    DATA = "data"
    SPATIAL = "spatial"
    STATE = "state"
    ATTRIBUTE = "attribute"


@dataclass(frozen=True)
class RelationInfo:
    name: str
    type: RelationType


class RelationRegistry:
    def __init__(self):
        self._relations: Dict[str, RelationInfo] = {}

    def register(self, rid: str, info: RelationInfo):
        self._relations[rid] = info

    def get(self, rid: str) -> RelationInfo | None:
        return self._relations.get(rid)

    def get_all(self) -> Dict[str, RelationInfo]:
        return self._relations
