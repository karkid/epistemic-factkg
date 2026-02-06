from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Set, Tuple, List
from utils.typing import Triple, TripleSet, TripleList


class BaseClaimGenerator(ABC):
    """
    Generate textual claims from RDF triples.
    """

    @abstractmethod
    def verbalize(self, triple: Triple) -> str:
        """
        Convert an RDF triple into a human-readable claim.
        """
        return ""


class BaseClaimCorruptor(ABC):
    """
    Corrupt textual claims for negative sampling.
    """

    @abstractmethod
    def corrupt(
        triple: Triple, all_triples: TripleList, triple_set: TripleSet
    ) -> Triple:
        """
        Corrupt a given claim.
        """
        return triple


class BaseClaimValidator(ABC):
    """
    Validate textual claims.
    """

    @abstractmethod
    def validate(self, triple: Triple) -> bool:
        """
        Validate a given claim.
        """
        return True
