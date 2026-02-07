from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set
from enum import auto

from enum import Enum, auto
class SpatialRole(Enum):
    SURFACE = auto()
    CONTAINER = auto()
    HANGING = auto()


@dataclass(frozen=True)
class EntityInfo:

    name: str
    type: str | None = None
    spatial_roles: Set[SpatialRole] | None = None
    is_countable: bool = True

class EntityRegistry:

    def __init__(self):
        self._entities: Dict[str, EntityInfo] = {}

    def register(self, eid: str, info: EntityInfo):
        self._entities[eid] = info

    def get(self, eid: str) -> EntityInfo | None:
        return self._entities.get(eid)

    def get_all(self) -> Dict[str, EntityInfo]:
        return self._entities
