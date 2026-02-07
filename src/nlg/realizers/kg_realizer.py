# src/nlg/realizers/kg_realizer.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.core.graph.types import Triple
from src.nlg.entities.formatter import EntityFormatter
from src.nlg.templates.sentence_template import SentenceTemplate


class KGRealizer:
    """
    Realizes a Triple into a sentence using predicate -> SentenceTemplate mapping.
    """

    def __init__(
        self,
        *,
        templates_by_predicate: Mapping[str, SentenceTemplate],
        formatter: EntityFormatter,
        predicate_to_prep: Optional[Dict[str, str]] = None,
    ):
        self.templates_by_predicate = dict(templates_by_predicate)
        self.formatter = formatter
        self.predicate_to_prep = predicate_to_prep or {}

    def realize(self, triple: Triple) -> str:
        s, p, o = triple

        template = self.templates_by_predicate.get(p)
        if template is None:
            raise ValueError(f"No template registered for predicate: {p}")

        slots = self._build_slots(s, p, o)
        return template.verbalize(**slots)

    def _build_slots(self, s: Any, p: str, o: Any) -> Dict[str, str]:
        return {
            "s": self.formatter.format_term(s),
            "o": self.formatter.format_term(o),
            "p": p,
            "prep": self._predicate_to_prep(p),
        }

    def _predicate_to_prep(self, p: str) -> str:
        return self.predicate_to_prep.get(p, p)
