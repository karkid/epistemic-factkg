from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, Optional


class PredicateForm(StrEnum):
    
    PREP = "prep"   # on, inside, under
    ADJ = "adj"     # open, dirty, broken
    ATTR = "attr"   # temperature, mass
    VERB = "verb"   # contains, includes
    PROP = "prop"   # for unary predicates (e.g. isReceptacle) that we want to verbalize as "is a receptacle"

@dataclass(frozen=True)
class PredicateLexeme:
    """
    Language surface form for a predicate.
    kind:
      - "prep": spatial predicate realized as preposition ("on", "inside")
      - "adj":  state predicate realized as adjective phrase ("open", "switched on")
      - "attr": data predicate realized as attribute label ("temperature")
      - "verb": object predicate realized as verb ("contains")
      - "prop": unary predicate realized as property phrase ("is a receptacle")
    """

    kind: PredicateForm
    label: str


class PredicateLexicon:
    def __init__(self) -> None:
        self._by_pid: Dict[str, PredicateLexeme] = {}

    def register(self, pid: str, lexeme: PredicateLexeme) -> None:
        self._by_pid[pid] = lexeme

    def get(self, pid: str) -> Optional[PredicateLexeme]:
        return self._by_pid.get(pid)

    def all(self) -> Dict[str, PredicateLexeme]:
        return dict(self._by_pid)
