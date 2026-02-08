
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from enum import Enum

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