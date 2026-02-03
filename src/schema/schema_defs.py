# src/schema/schema_defs.py

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

Triple = Tuple[str, str, str]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(
    *,
    rec_id: str,
    claim: str,
    label: str,
    structural_reasoning: str,
    evidence_type: str,
    evidence_triples: Sequence[Triple],
    evidence_source: str,
    evidence_source_type: str,
    evidence_urls: Optional[List[str]] = None,
    scene_id: Optional[str] = None,
    generator: str = "ai2thor",
    split: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a dataset record in Epistemic-FactKG schema.
    """

    if evidence_urls is None:
        evidence_urls = []

    return {
        "id": rec_id,
        "claim": claim,
        "label": label,
        "structural_reasoning": structural_reasoning,
        "evidence_type": evidence_type,
        "evidence": {
            "triples": [[s, p, o] for (s, p, o) in evidence_triples],
            "source": evidence_source,
            "source_type": evidence_source_type,
            "urls": evidence_urls,
        },
        "context": {
            "scene_id": scene_id,
            "generator": generator,
            "split": split,
        },
        "meta": {
            "created_utc": utc_now_iso(),
            "notes": notes,
        },
    }
