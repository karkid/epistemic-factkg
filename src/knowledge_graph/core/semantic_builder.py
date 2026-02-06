from typing import Dict, Any
from knowledge_graph.semantics.source.base import ClaimCorpus, SemanticDataSource
from dataclasses import dataclass


@dataclass
class SemanticBuildResult:
    """Result of building a knowledge graph."""

    num_claims: int
    num_refuted_claims: int
    num_supported_claims: int


class SemanticBuilder:
    def __init__(self):
        self.claims_corpus = ClaimCorpus()
        self.stats = {"claims": 0, "refuted": 0, "supported": 0}

    def build_from_source(self, data_source: SemanticDataSource) -> SemanticBuildResult:
        """Build knowledge graph from any data source."""
        self.claims_corpus = ClaimCorpus()
        self.stats = {"claims": 0, "refuted": 0, "supported": 0}

        # Process each claim
        for claim_data in data_source.get_available_claims():
            self._add_claim(claim_data)

        return SemanticBuildResult(
            num_claims=self.stats["claims"],
            num_refuted_claims=self.stats["refuted"],
            num_supported_claims=self.stats["supported"],
        )

    def build_from_corpus(self, corpus: ClaimCorpus) -> SemanticBuildResult:
        """Build knowledge graph from a given claim corpus."""
        self.claims_corpus = ClaimCorpus()
        self.stats = {"claims": 0, "refuted": 0, "supported": 0}

        for claim_instance in corpus:
            self._add_claim(claim_instance)

        return SemanticBuildResult(
            num_claims=self.stats["claims"],
            num_refuted_claims=self.stats["refuted"],
            num_supported_claims=self.stats["supported"],
        )

    def export_corpus_schema(self) -> Dict[str, Any]:
        """Export the built claim corpus."""
        return self.claims_corpus.get_schema_layout()

    def _add_claim(self, claim_instance):
        """Add a claim instance to the corpus and update stats."""
        self.claims_corpus.add_claim(claim_instance)
        self.stats["claims"] += 1
        if claim_instance.claim.label == "refuted":
            self.stats["refuted"] += 1
        elif claim_instance.claim.label == "supported":
            self.stats["supported"] += 1
