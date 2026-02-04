from __future__ import annotations

from typing import Set, Tuple, List

Triple = Tuple[str, str, str]


class BaseClaimGenerator:
    """
    Generate textual claims from RDF triples.
    """

    def verbalize(self, triple: Triple) -> str:
        """
        Convert an RDF triple into a human-readable claim.
        """
        return ""


class BaseClaimCorruptor:
    """
    Corrupt textual claims for negative sampling.
    """

    def corrupt(
        triple: Triple, all_triples: List[Triple], triple_set: Set[Triple]
    ) -> Triple:
        """
        Corrupt a given claim.
        """
        return triple


class BaseClaimValidator:
    """
    Validate textual claims.
    """

    def validate(self, triple: Triple) -> bool:
        """
        Validate a given claim.
        """
        return True
