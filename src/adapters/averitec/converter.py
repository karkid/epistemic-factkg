from __future__ import annotations

import re

from src.core.ports.dataset.converter import DatasetConverter
from src.core.claims.labels import CONFIDENCE_WEIGHTS, PramanaLabel
from src.utils.time import utc_now_iso


_LABEL_MAP = {
    "supported": "supported",
    "refuted": "refuted",
    "not enough evidence": "not_enough_evidence",
    "not_enough_evidence": "not_enough_evidence",
    "conflicting evidence/cherrypicking": "conflicting_evidence",
    "conflicting evidence": "conflicting_evidence",
    "cherrypicking": "conflicting_evidence",
}

_ANSWER_TYPE_MAP = {
    "boolean": "boolean",
    "extractive": "extractive",
    "abstractive": "abstractive",
    "unanswerable": "unanswerable",
}

_PERCEPTUAL = {"image", "video", "audio"}
_TEXTUAL = {"web_text", "pdf", "web_table", "other"}

_NUMERIC_CUES = re.compile(
    r"\b(%|percent|percentage|largest|smallest|rank|gdp|million|billion|trillion)\b",
    re.IGNORECASE,
)

_PRIMARY_ORDER = [
    "non_apprehension",
    "perception",
    "comparison_analogy",
    "inference",
    "testimony",
    "postulation_derivation",
]


def _normalize_label(label) -> str:
    if not label:
        return "not_enough_evidence"
    return _LABEL_MAP.get(
        str(label).strip().lower(), str(label).strip().lower().replace(" ", "_")
    )


def _medium_to_modality(source_medium) -> str:
    if not source_medium:
        return "other"
    sm = str(source_medium).strip().lower().replace(" ", "_")
    for key, mod in (
        ("web_table", "web_table"),
        ("web_text", "web_text"),
        ("pdf", "pdf"),
        ("video", "video"),
        ("youtube", "video"),
        ("image", "image"),
        ("jpeg", "image"),
        ("png", "image"),
        ("audio", "audio"),
    ):
        if key in sm:
            return mod
    return "other"


def _infer_pramana(
    label: str,
    modalities: set[str],
    src_urls: set[str],
    answer_types: list[str],
    answers_text: str,
) -> tuple[str, list[str], float]:
    proof_types: set[str] = set()

    if label == "not_enough_evidence":
        proof_types.add("non_apprehension")

    if modalities & _PERCEPTUAL:
        proof_types.add("perception")

    if modalities & _TEXTUAL:
        proof_types.add("testimony")

    if _NUMERIC_CUES.search(answers_text):
        proof_types.add("comparison_analogy")

    # Fixed inference saturation: require >=2 abstractive answers AND >=2 distinct sources
    n_abstractive = sum(1 for t in answer_types if t == "abstractive")
    if n_abstractive >= 2 and len(src_urls) >= 2:
        proof_types.add("inference")

    if not proof_types:
        proof_types.add("testimony")

    sorted_types = sorted(
        proof_types,
        key=lambda p: _PRIMARY_ORDER.index(p) if p in _PRIMARY_ORDER else 99,
    )
    primary = sorted_types[0]
    weight = CONFIDENCE_WEIGHTS.get(PramanaLabel(primary), 0.70)
    return primary, sorted_types, weight


class AveritecConverter(DatasetConverter):
    """Converts AVeriTeC JSON records to unified v2.0 JSONL."""

    @property
    def dataset_name(self) -> str:
        return "averitec"

    def iter_records(self, in_path: str):
        import json

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list at top-level in {in_path}")
        yield from enumerate(data, start=1)

    def infer_pramana(self, raw_record: dict) -> tuple[str, list[str], float]:
        label = _normalize_label(raw_record.get("label"))
        evidence_items = self._build_evidence(raw_record, "placeholder")
        modalities = {e["modality"] for e in evidence_items}
        src_urls = {e["source_url"] for e in evidence_items if e.get("source_url")}
        answer_types = [e["_answer_type"] for e in evidence_items]
        answers_text = " ".join(e["_answer_text"] for e in evidence_items)
        return _infer_pramana(label, modalities, src_urls, answer_types, answers_text)

    def _build_evidence(self, rec: dict, rec_id: str) -> list[dict]:
        """Build evidence dicts (with private _answer_* keys for pramana inference)."""
        label = _normalize_label(rec.get("label"))
        if label == "supported":
            stance = "supports"
        elif label == "refuted":
            stance = "refutes"
        else:
            stance = "unknown"

        items = []
        for qi, q in enumerate(rec.get("questions") or [], start=1):
            qtext = (q.get("question") or "").strip()
            for ai, a in enumerate(q.get("answers") or [], start=1):
                evidence_id = f"{rec_id}-q{qi}-a{ai}"
                ans_text = str(a.get("answer") or "").strip()
                source_medium = a.get("source_medium")
                modality = _medium_to_modality(source_medium)
                ans_type_key = str(a.get("answer_type") or "").strip().lower()
                ans_type = _ANSWER_TYPE_MAP.get(ans_type_key, "unanswerable")
                text = f"{qtext} {ans_text}".strip() if qtext else ans_text

                items.append(
                    {
                        "evidence_id": evidence_id,
                        "text": text or None,
                        "triples": [],
                        "triple_source": None,
                        "modality": modality,
                        "stance": stance,
                        "source_url": a.get("source_url"),
                        # Private keys stripped before output
                        "_answer_type": ans_type,
                        "_answer_text": ans_text,
                    }
                )
        return items

    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        oid = str(raw_record.get("id") or rec_id)
        claim_text = (raw_record.get("claim") or "").strip()
        label = _normalize_label(raw_record.get("label"))

        evidence_raw = self._build_evidence(raw_record, oid)

        modalities = {e["modality"] for e in evidence_raw}
        src_urls = {e["source_url"] for e in evidence_raw if e.get("source_url")}
        answer_types = [e["_answer_type"] for e in evidence_raw]
        answers_text = " ".join(e["_answer_text"] for e in evidence_raw)

        pramana_primary, pramana_all, confidence_weight = _infer_pramana(
            label, modalities, src_urls, answer_types, answers_text
        )

        evidence_out = [
            {k: v for k, v in e.items() if not k.startswith("_")} for e in evidence_raw
        ]

        if not evidence_out:
            evidence_out = [
                {
                    "evidence_id": f"{oid}-e0",
                    "text": None,
                    "triples": [],
                    "triple_source": None,
                    "modality": "other",
                    "stance": "unknown",
                    "source_url": None,
                }
            ]

        return {
            "schema_version": "2.0",
            "id": oid,
            "claim": claim_text,
            "verdict": {
                "label": label,
                "justification": raw_record.get("justification"),
            },
            "epistemic": {
                "pramana_primary": pramana_primary,
                "pramana_all": pramana_all,
                "confidence_weight": confidence_weight,
                "assignment_method": "rule_based",
            },
            "claim_triples": None,
            "reasoning": None,
            "evidence": evidence_out,
            "provenance": {
                "dataset": "averitec",
                "split": None,
                "context_id": "schlichtkrull2023averitec",
            },
            "meta": {
                "schema_version": "2.0",
                "created_utc": utc_now_iso(),
            },
        }
