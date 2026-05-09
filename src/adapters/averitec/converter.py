from __future__ import annotations

import re

from src.core.ports.dataset.converter import DatasetConverter
from src.core.claims.labels import CONFIDENCE_WEIGHTS, EvidenceStance, Pramana, Verdict
from src.utils.time import utc_now_iso


_LABEL_MAP: dict[str, Verdict] = {
    "supported": Verdict.SUPPORTED,
    "refuted": Verdict.REFUTED,
    "not enough evidence": Verdict.NOT_ENOUGH_EVIDENCE,
    "not_enough_evidence": Verdict.NOT_ENOUGH_EVIDENCE,
    "conflicting evidence/cherrypicking": Verdict.CONFLICTING_EVIDENCE,
    "conflicting evidence": Verdict.CONFLICTING_EVIDENCE,
    "cherrypicking": Verdict.CONFLICTING_EVIDENCE,
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

# Descending priority: first match becomes pramana_primary
_PRIMARY_ORDER = [
    Pramana.PERCEPTION,
    Pramana.COMPARISON_ANALOGY,
    Pramana.INFERENCE,
    Pramana.TESTIMONY,
    Pramana.POSTULATION_DERIVATION,
]


def _normalize_label(label) -> Verdict:
    if not label:
        return Verdict.NOT_ENOUGH_EVIDENCE
    return _LABEL_MAP.get(
        str(label).strip().lower(),
        Verdict.NOT_ENOUGH_EVIDENCE,
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
    modalities: set[str],
    src_urls: set[str],
    answer_types: list[str],
    answers_text: str,
) -> tuple[str, list[str], float]:
    proof_types: set[Pramana] = set()

    if modalities & _PERCEPTUAL:
        proof_types.add(Pramana.PERCEPTION)

    if modalities & _TEXTUAL:
        proof_types.add(Pramana.TESTIMONY)

    if _NUMERIC_CUES.search(answers_text):
        proof_types.add(Pramana.COMPARISON_ANALOGY)

    # Require >=2 abstractive answers AND >=2 distinct sources to avoid saturation
    n_abstractive = sum(1 for t in answer_types if t == "abstractive")
    if n_abstractive >= 2 and len(src_urls) >= 2:
        proof_types.add(Pramana.INFERENCE)

    if not proof_types:
        proof_types.add(Pramana.TESTIMONY)

    sorted_types = sorted(
        proof_types,
        key=lambda p: _PRIMARY_ORDER.index(p) if p in _PRIMARY_ORDER else 99,
    )
    primary = sorted_types[0]
    weight = CONFIDENCE_WEIGHTS.get(primary, 0.70)
    return primary.value, [p.value for p in sorted_types], weight


def _verdict_to_stance(label: Verdict) -> EvidenceStance | None:
    """
    Map verdict to per-evidence stance.

    For supported/refuted we know all evidence points one way — AVeriTeC fact-checkers
    cite only the decisive evidence. For conflicting/not_enough we cannot assign a
    per-item stance without per-answer annotation, so we leave it null.
    """
    if label == Verdict.SUPPORTED:
        return EvidenceStance.SUPPORTS
    if label == Verdict.REFUTED:
        return EvidenceStance.REFUTES
    return None


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
        evidence_items = self._build_evidence(raw_record, "placeholder")
        modalities = {e["modality"] for e in evidence_items}
        src_urls = {e["source_url"] for e in evidence_items if e.get("source_url")}
        answer_types = [e["_answer_type"] for e in evidence_items]
        answers_text = " ".join(e["_answer_text"] for e in evidence_items)
        return _infer_pramana(modalities, src_urls, answer_types, answers_text)

    def _build_evidence(self, rec: dict, rec_id: str) -> list[dict]:
        """Build evidence dicts (with private _answer_* keys for pramana inference)."""
        label = _normalize_label(rec.get("label"))
        stance = _verdict_to_stance(label)

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
                        "stance": stance.value if stance else None,
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
            modalities, src_urls, answer_types, answers_text
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
                    "stance": None,
                    "source_url": None,
                }
            ]

        return {
            "schema_version": "2.0",
            "id": oid,
            "claim": claim_text,
            "verdict": {
                "label": label.value,
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
