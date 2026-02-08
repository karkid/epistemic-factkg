from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.core.registry.relation import RelationInfo
from src.core.registry.entity import EntityInfo

@dataclass(frozen=True)
class PredicateMapping:
    info: RelationInfo
    uri: str
    transform: Optional[Callable[[Any], Any]] = None

@dataclass(frozen=True)
class EntityMapping:
    info: EntityInfo
    uri: str
    transform: Optional[Callable[[Any], Any]] = None