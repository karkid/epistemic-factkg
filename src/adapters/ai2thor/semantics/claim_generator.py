from __future__ import annotations

from src.core.graph.types import Triple
from src.core.ontology.base import BaseOntology
from src.nlg.realizers.kg_realizer import KGRealizer
from src.semantics.claims.base import BaseClaimGenerator


class AI2ThorClaimGenerator(BaseClaimGenerator):
    def __init__(self, *, ontology: BaseOntology, realizer: KGRealizer):
        self.ontology = ontology
        self.realizer = realizer

    def verbalize(self, triple: Triple) -> str:
        # If you want predicate filtering, keep it here; otherwise just realize.
        return self.realizer.realize(triple)
