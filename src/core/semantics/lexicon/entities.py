from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable


@dataclass(frozen=True)
class EntityLexeme:
    label: str  # e.g. "TV stand", "paper towel roll"
    is_countable: bool = True
    proper: bool = False              # True => no article
    mass_noun: bool = False


class EntityLexicon:
    def __init__(self) -> None:
        self._by_eid: Dict[str, EntityLexeme] = {}

    def register(self, eid: str, lexeme: EntityLexeme) -> None:
        self._by_eid[eid] = lexeme

    def get(self, eid: str) -> Optional[EntityLexeme]:
        return self._by_eid.get(eid)

    def all(self) -> Dict[str, EntityLexeme]:
        return dict(self._by_eid)
