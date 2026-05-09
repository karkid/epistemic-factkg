from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, Optional


class PredicateForm(StrEnum):
    PREP = "prep"  # on, inside, under
    ADJ = "adj"  # open, dirty, broken
    ATTR = "attr"  # temperature, mass
    VERB = "verb"  # contains, includes
    PROP_STATE = "prop_state"  # for state predicates that we want to verbalize as "is a receptacle" instead of "is receptacle"
    PROP_VALUE = "prop_value"  # for value predicates (e.g. temperature) that we want to verbalize as "is hot" instead of "has temperature hot"


@dataclass(frozen=True)
class PredicateLexeme:
    """
    Language surface form for a predicate.
    kind:
      - "prep": spatial predicate realized as preposition ("on", "inside")
      - "adj":  state predicate realized as adjective phrase ("open", "switched on")
      - "attr": data predicate realized as attribute label ("temperature")
      - "verb": object predicate realized as verb ("contains")
      - "prop_state": unary predicate realized as property phrase ("is a receptacle")
      - "prop_value": unary predicate realized as property phrase with value ("is hot")

    template_mode:
      - "predicate": Use predicate label in template (e.g., "The fridge is openable")
      - "value": Use object value in template (e.g., "The cabinet is hot")
    """

    kind: PredicateForm
    label: str
    template_mode: str = "predicate"  # "predicate" or "value"


class PredicateLexicon:
    def __init__(self) -> None:
        self._by_pid: Dict[str, PredicateLexeme] = {}

    def register(self, pid: str, lexeme: PredicateLexeme) -> None:
        self._by_pid[pid] = lexeme

    def get(self, pid: str) -> Optional[PredicateLexeme]:
        return self._by_pid.get(pid)

    def all(self) -> Dict[str, PredicateLexeme]:
        return dict(self._by_pid)
