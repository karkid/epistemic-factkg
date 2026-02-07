from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PredicateLexeme:
    """
    Language surface form for a predicate.
    kind:
      - "prep": spatial predicate realized as preposition ("on", "inside")
      - "adj":  state predicate realized as adjective phrase ("open", "switched on")
      - "attr": data predicate realized as attribute label ("temperature")
      - "verb": object predicate realized as verb ("contains")
    """
    kind: str
    text: str


class PredicateLexicon:
    def __init__(self) -> None:
        self._by_pid: Dict[str, PredicateLexeme] = {}

    def register(self, pid: str, lexeme: PredicateLexeme) -> None:
        self._by_pid[pid] = lexeme

    def get(self, pid: str) -> Optional[PredicateLexeme]:
        return self._by_pid.get(pid)

    def all(self) -> Dict[str, PredicateLexeme]:
        return dict(self._by_pid)
