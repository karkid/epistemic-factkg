# src/core/claims/types.py
from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from src.core.graph.types import Triple, TripleList
from src.core.claims.labels import EvidenceStance, Pramana, Verdict
from src.utils.io import write_jsonl
from src.utils.time import utc_now_iso


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_triples: Sequence[Triple]
    evidence_source: str
    evidence_source_type: str
    evidence_urls: List[str]  # default to empty list at construction
    evidence_extract: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Context:
    context_id: Optional[str] = None
    context_type: Optional[str] = None
    generator: str = "agent"
    split: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Meta:
    created_utc: str
    notes: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Reasoning:
    structural: str


@dataclass(frozen=True, slots=True)
class Claim:
    """
    A claim is text + triples expressing it.

    `text` is a good key name (common, unambiguous).
    Avoid `claim` as a field name because it conflicts with the concept/type name.
    """

    text: str
    claim_triples: TripleList


@dataclass(frozen=True, slots=True)
class ClaimInstance:
    """
    One labeled example in your dataset: claim + label + reasoning + evidence + context + meta.
    """

    rec_id: str
    claim: Claim
    label: str
    reasoning: Reasoning
    evidence: Evidence
    context: Context
    meta: Meta

    def get_schema_layout(self) -> Dict[str, Any]:
        from src.core.claims.labels import CONFIDENCE_WEIGHTS

        pramana = self.evidence.evidence_source_type
        try:
            weight = CONFIDENCE_WEIGHTS.get(Pramana(pramana), 0.70)
        except ValueError:
            weight = 0.70

        structural = self.reasoning.structural
        structural_v2 = structural.replace("-", "_") if structural else None

        evidence_triples = [(t.s, t.p, t.o) for t in self.evidence.evidence_triples]
        claim_triples = [(t.s, t.p, t.o) for t in self.claim.claim_triples]

        return {
            "schema_version": "2.0",
            "id": self.rec_id,
            "claim": self.claim.text,
            "verdict": {
                "label": self.label,
                "justification": self.evidence.evidence_extract,
            },
            "epistemic": {
                "pramana_primary": pramana,
                "pramana_all": [pramana],
                "confidence_weight": weight,
                "assignment_method": "rule_based",
            },
            "claim_triples": claim_triples if claim_triples else None,
            "reasoning": {
                "structural": structural_v2,
                "strategy": None,
            }
            if structural_v2
            else None,
            "evidence": [
                {
                    "evidence_id": f"{self.rec_id}-e0",
                    "text": self.evidence.evidence_extract,
                    "triples": evidence_triples if evidence_triples else [],
                    "triple_source": "ground_truth",
                    "modality": "simulation_state",
                    "stance": (
                        EvidenceStance.ABSENT.value
                        if pramana == Pramana.NON_APPREHENSION.value
                        and self.label == Verdict.SUPPORTED.value
                        else EvidenceStance.SUPPORTS.value
                        if self.label == Verdict.SUPPORTED.value
                        else EvidenceStance.REFUTES.value
                    ),
                    "source_url": self.evidence.evidence_urls[0]
                    if self.evidence.evidence_urls
                    else None,
                }
            ],
            "provenance": {
                "dataset": "ai2thor",
                "split": self.context.split,
                "context_id": self.context.context_id,
            },
            "meta": {
                "schema_version": "2.0",
                "created_utc": self.meta.created_utc,
            },
        }

    def get_schema_layout_json(self) -> str:

        return json.dumps(self.get_schema_layout(), indent=2)

    @staticmethod
    def make_instance(
        *,
        rec_id: str,
        claim_text: str,
        label: str,
        claim_triples: Sequence[Triple],
        structural_reasoning: str,
        evidence_triples: Sequence[Triple],
        evidence_source: str,
        evidence_source_type: str,
        evidence_urls: Optional[List[str]] = None,
        context_id: Optional[str] = None,
        generator: str = "agent",
        context_type: Optional[str] = None,
        split: Optional[str] = None,
        notes: Optional[str] = None,
        created_utc: Optional[str] = None,
        evidence_extract: Optional[str] = None,
    ) -> "ClaimInstance":
        # defaults
        if evidence_urls is None:
            evidence_urls = []

        if created_utc is None:
            created_utc = utc_now_iso()

        claim = Claim(text=claim_text, claim_triples=list(claim_triples))
        reasoning = Reasoning(structural=structural_reasoning)
        evidence = Evidence(
            evidence_triples=list(evidence_triples),
            evidence_source=evidence_source,
            evidence_source_type=evidence_source_type,
            evidence_urls=evidence_urls,
            evidence_extract=evidence_extract,
        )
        context = Context(
            context_id=context_id,
            context_type=context_type,
            generator=generator,
            split=split,
        )
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
    """In-memory collection of ClaimInstance objects."""

    def __init__(self, claims: Optional[Iterable[ClaimInstance]] = None):

        self.claims: List[ClaimInstance] = list(claims) if claims else []

    def add(self, claim_instance: ClaimInstance) -> None:

        self.claims.append(claim_instance)

    def extend(self, claim_instances: Iterable[ClaimInstance]) -> None:

        self.claims.extend(claim_instances)

    def all(self) -> List[ClaimInstance]:

        return self.claims

    def filter_by_label(self, label: str) -> List[ClaimInstance]:

        return [ci for ci in self.claims if ci.label == label]

    def count_by_label(self) -> Dict[str, int]:

        out: Dict[str, int] = {}

        for ci in self.claims:
            out[ci.label] = out.get(ci.label, 0) + 1

        return out

    def unique_labels(self) -> List[str]:

        return sorted({ci.label for ci in self.claims})

    def get_by_id(self, rec_id: str) -> Optional[ClaimInstance]:

        for ci in self.claims:
            if ci.rec_id == rec_id:
                return ci

        return None

    def get_by_ids(self, rec_ids: Sequence[str]) -> List[ClaimInstance]:

        wanted = set(rec_ids)
        return [ci for ci in self.claims if ci.rec_id in wanted]

    def split_by_label(self) -> Dict[str, "ClaimCorpus"]:

        out: Dict[str, ClaimCorpus] = {}

        for ci in self.claims:
            out.setdefault(ci.label, ClaimCorpus()).add(ci)

        return out

    def get_schema_layout(self) -> List[Dict[str, Any]]:

        return [ci.get_schema_layout() for ci in self.claims]

    def get_schema_layout_json(self) -> str:

        return json.dumps(self.get_schema_layout(), indent=2)

    def clear(self) -> None:

        self.claims.clear()

    def remove_duplicates(self) -> None:

        seen = set()
        unique_claims = []
        for ci in self.claims:
            layout_json = ci.get_schema_layout_json()
            if layout_json not in seen:
                seen.add(layout_json)
                unique_claims.append(ci)
        self.claims = unique_claims

    def save_to_jsonl(self, file_path: str) -> None:
        write_jsonl(file_path, [ci.get_schema_layout() for ci in self.claims])

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
