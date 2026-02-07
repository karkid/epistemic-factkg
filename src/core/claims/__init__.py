from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Iterator, Optional, Sequence, Tuple
from src.core.graph.types import Triple

@dataclass
class Evidence:
    """Evidence supporting a claim."""

    evidence_triples: Sequence[Triple]
    evidence_source: str
    evidence_source_type: str
    evidence_urls: Optional[List[str]] = None


@dataclass
class Context:
    """Contextual information for a claim."""

    scene_id: Optional[str] = None
    generator: str = "agent"
    split: Optional[str] = None


@dataclass
class Meta:
    """Metadata for a claim record."""

    created_utc: str
    notes: Optional[str] = None


@dataclass
class Reasoning:
    """Reasoning behind a claim."""

    structural: str


@dataclass
class Claim:
    text: str
    claim_triples: TripleList


@dataclass
class ClaimInstance:
    """Structured record for a claim in the dataset."""

    rec_id: str
    claim: Claim
    label: str
    reasoning: Reasoning
    evidence: Evidence
    context: Context
    meta: Meta

    def get_schema_layout(self) -> Dict[str, Any]:
        """Return the schema layout of the claim instance."""
        return {
            "id": self.rec_id,
            "label": self.label,
            "claim": self.claim.text,
            "claim_triples": self.claim.claim_triples,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "context": self.context,
            "meta": self.meta,
        }

    @staticmethod
    def make_instance(
        *,
        rec_id: str,
        claim: str,
        label: str,
        claim_triples: Sequence[Triple],
        structural_reasoning: str,
        evidence_triples: Sequence[Triple],
        evidence_source: str,
        evidence_source_type: str,
        evidence_urls: Optional[List[str]] = None,
        scene_id: Optional[str] = None,
        generator: str = "agent",
        split: Optional[str] = None,
        notes: Optional[str] = None,
        created_utc: Optional[str] = None,
    ) -> "ClaimInstance":
        """Factory method to create a ClaimInstance."""

        if evidence_urls is None:
            evidence_urls = []

        if created_utc is None:
            from schema.schema_defs import utc_now_iso

            created_utc = utc_now_iso()

        claim = Claim(claim=claim, claim_triples=list(claim_triples))
        reasoning = Reasoning(structural=structural_reasoning)
        evidence = Evidence(
            evidence_triples=list(evidence_triples),
            evidence_source=evidence_source,
            evidence_source_type=evidence_source_type,
            evidence_urls=evidence_urls,
        )
        context = Context(scene_id=scene_id, generator=generator, split=split)
        meta = Meta(created_utc=created_utc, notes=notes)

        return ClaimInstance(
            rec_id=rec_id,
            claim=claim,
            label=label,
            reasoning=reasoning,
            evidence=evidence,
            context=context,
            meta=meta,
        )


class ClaimCorpus:
    """A corpus of claims."""

    def __init__(self, claims: List[ClaimInstance]):
        self.claims: List[ClaimInstance] = []

    def add_claim(self, claim_instance: ClaimInstance) -> None:
        """Add a claim instance to the corpus."""
        self.claims.append(claim_instance)

    def add_claims(self, claim_instances: List[ClaimInstance]) -> None:
        """Add multiple claim instances to the corpus."""
        self.claims.extend(claim_instances)

    def get_all_claims(self) -> List[ClaimInstance]:
        """Retrieve all claim instances in the corpus."""
        return self.claims

    def filter_by_label(self, label: str) -> List[ClaimInstance]:
        """Filter claims by their label."""
        return [ci for ci in self.claims if ci.claim.label == label]

    def count_by_label(self) -> Dict[str, int]:
        """Count claims by their labels."""
        label_count: Dict[str, int] = {}
        for ci in self.claims:
            label_count[ci.claim.label] = label_count.get(ci.claim.label, 0) + 1
        return label_count

    def __len__(self) -> int:
        return len(self.claims)

    def __iter__(self) -> Iterator[ClaimInstance]:
        return iter(self.claims)

    def __getitem__(self, index: int) -> ClaimInstance:
        return self.claims[index]

    def __contains__(self, claim_instance: ClaimInstance) -> bool:
        return claim_instance in self.claims

    def __repr__(self) -> str:
        return f"ClaimCorpus(num_claims={len(self.claims)})"

    def __str__(self) -> str:
        return f"ClaimCorpus with {len(self.claims)} claims"

    def clear(self) -> None:
        """Clear all claims from the corpus."""
        self.claims.clear()

    def extend(self, other: "ClaimCorpus") -> None:
        """Extend the corpus with claims from another corpus."""
        self.claims.extend(other.claims)

    def get_claim_by_id(self, rec_id: str) -> Optional[ClaimInstance]:
        """Retrieve a claim instance by its record ID."""
        for ci in self.claims:
            if ci.claim.rec_id == rec_id:
                return ci
        return None

    def get_claims_by_ids(self, rec_ids: List[str]) -> List[ClaimInstance]:
        """Retrieve multiple claim instances by their record IDs."""
        id_set = set(rec_ids)
        return [ci for ci in self.claims if ci.claim.rec_id in id_set]

    def unique_labels(self) -> List[str]:
        """Get a list of unique labels in the corpus."""
        return list(set(ci.claim.label for ci in self.claims))

    def split_by_label(self) -> Dict[str, "ClaimCorpus"]:
        """Split the corpus into multiple corpora by label."""
        split_corpora: Dict[str, ClaimCorpus] = {}
        for ci in self.claims:
            if ci.claim.label not in split_corpora:
                split_corpora[ci.claim.label] = ClaimCorpus([])
            split_corpora[ci.claim.label].add_claim(ci)
        return split_corpora

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert the corpus to a list of dictionaries."""
        return [ci.__dict__ for ci in self.claims]

    def get_schema_layout(self) -> List[Dict[str, Any]]:
        """Get the schema layout for all claim instances in the corpus."""
        return [ci.get_schema_layout() for ci in self.claims]
