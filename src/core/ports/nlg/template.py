from __future__ import annotations
from abc import ABC, abstractmethod

from src.core.semantics.lexicon.predicates import PredicateForm
from src.core.graph.types import Triple


class BaseTemplate(ABC):
    @abstractmethod
    def render(self, triple: Triple, kind: PredicateForm) -> str:
        """Generate a sentence from the template, filling in the provided fields."""
        raise NotImplementedError

    @abstractmethod
    def render_conjunction(
        self,
        t1: Triple,
        k1: PredicateForm,
        t2: Triple,
        k2: PredicateForm,
        conj: str = "and",
    ) -> str:
        """Generate a sentence expressing the conjunction of two triples."""
        raise NotImplementedError

    @abstractmethod
    def render_negation(self, triple: Triple, kind: PredicateForm) -> str:
        """Generate a sentence expressing the negation of a triple."""
        raise NotImplementedError
