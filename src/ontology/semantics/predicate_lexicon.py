"""Predicate lexicon — maps predicate IDs to surface-form labels and syntactic kinds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, Optional


class PredicateForm(StrEnum):
    PREP = "prep"  # on, inside, under
    ADJ = "adj"  # open, dirty, broken
    ATTR = "attr"  # temperature, mass
    VERB = "verb"  # contains, includes
    PROP_STATE = "prop_state"  # "is a receptacle"
    PROP_VALUE = "prop_value"  # "is hot"


@dataclass(frozen=True)
class PredicateLexeme:
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
