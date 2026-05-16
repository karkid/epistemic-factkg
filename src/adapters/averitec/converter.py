from __future__ import annotations

import re
from urllib.parse import urlparse

from src.ports.converter import DatasetConverter
from src.epistemic.enums import (
    EvidenceStance,
    EvidenceType,
    ReasoningStrategy,
    Verdict,
)
from src.epistemic.registry import get_source_trust, resolve_source_id
from src.utils.time import utc_now_iso


_STRATEGY_MAP: dict[str, str] = {
    "numerical comparison": ReasoningStrategy.SPATIAL_COMPARISON,
    "written evidence": ReasoningStrategy.TESTIMONIAL_LOOKUP,
    "consultation": ReasoningStrategy.MULTI_HOP_INFERENCE,
    "expert consultation": ReasoningStrategy.MULTI_HOP_INFERENCE,
    "fact-checker reference": ReasoningStrategy.MULTI_HOP_INFERENCE,
    "satirical source identification": ReasoningStrategy.TESTIMONIAL_LOOKUP,
}

# Priority order: more informative strategies win when multiple are present.
_STRATEGY_PRIORITY: list[str] = [
    ReasoningStrategy.MULTI_HOP_INFERENCE,
    ReasoningStrategy.SPATIAL_COMPARISON,
    ReasoningStrategy.TESTIMONIAL_LOOKUP,
]


def _to_strategy(strategies: list[str] | None) -> str:
    """Map AVeriTeC fact_checking_strategies list to a single canonical value.

    When multiple strategies are present, the most informative one wins
    according to _STRATEGY_PRIORITY.
    """
    if not strategies:
        return ReasoningStrategy.TESTIMONIAL_LOOKUP
    mapped = {
        _STRATEGY_MAP.get(s.strip().lower(), ReasoningStrategy.TESTIMONIAL_LOOKUP)
        for s in strategies
    }
    for candidate in _STRATEGY_PRIORITY:
        if candidate in mapped:
            return candidate
    return ReasoningStrategy.TESTIMONIAL_LOOKUP


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

# Perceptual modalities → evidence_type = ["perception"]
# No inference added by default — perceptual evidence is direct observation.
# Inference is added by the abstractive multi-source post-pass if warranted.
_PERCEPTUAL = {"image", "video", "audio"}

# Textual modalities → testimony base (web_table also gets comparison_analogy)
_TEXTUAL = {"web_text", "pdf", "web_table", "other"}

# fact_checking_strategies → evidence_type enrichment for all textual items
_STRATEGY_EVIDENCE_TYPE_MAP: dict[str, str] = {
    "numerical comparison": EvidenceType.COMPARISON_ANALOGY.value,
    "consultation": EvidenceType.INFERENCE.value,
}

# Numeric/statistical comparison cue — triggers comparison_analogy evidence type
_NUMERIC_CUES = re.compile(
    r"\b(%|percent|percentage|largest|smallest|rank|gdp|million|billion|trillion)\b",
    re.IGNORECASE,
)

# Wayback Machine URL: extract the embedded original URL after the timestamp
_ARCHIVE_RE = re.compile(r"web\.archive\.org/web/\d+[^/]*/(.+)")

# Inference strength per answer_type (IS rubric from ADR-019)
# boolean/extractive → direct lookup (0.8), abstractive → synthesised (0.6),
# unanswerable → no real evidence (0.0)
_IS_FROM_ANSWER_TYPE: dict[str, float] = {
    "boolean": 0.8,
    "extractive": 0.8,
    "abstractive": 0.6,
    "unanswerable": 0.0,
}


def _normalize_label(label) -> Verdict:
    if not label:
        return Verdict.NOT_ENOUGH_EVIDENCE
    return _LABEL_MAP.get(
        str(label).strip().lower(),
        Verdict.NOT_ENOUGH_EVIDENCE,
    )


def _medium_to_modality(
    source_medium,
    source_url: str = "",
    answer_type: str = "",
) -> str:
    if not source_medium:
        return "unanswerable" if answer_type == "unanswerable" else "other"

    sm_raw = str(source_medium).strip()
    sm = sm_raw.lower().replace(" ", "_")

    if sm == "metadata":
        return "annotator_knowledge"

    if sm == "other":
        return "web_text" if source_url else "other"

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


def _verdict_to_stance(label: Verdict) -> EvidenceStance | None:
    if label == Verdict.SUPPORTED:
        return EvidenceStance.SUPPORTS
    if label == Verdict.REFUTED:
        return EvidenceStance.REFUTES
    return None


def _infer_evidence_types_basic(
    modality: str, answer_type: str, answer_text: str
) -> list[str]:
    """Determine base evidence_types for one AVeriTeC evidence item.

    web_table gets comparison_analogy + testimony (tabular data implies numerical
    comparison); all other textual modalities get testimony as base.
    Inference from multi-source synthesis and fact_checking_strategies are added
    as post-processing steps in AveritecConverter._build_evidence.
    """
    if modality in _PERCEPTUAL:
        return [EvidenceType.PERCEPTION.value]
    if answer_type == "unanswerable":
        return []
    if modality == "web_table":
        # Tabular data inherently supports comparison_analogy in addition to testimony
        return [EvidenceType.COMPARISON_ANALOGY.value, EvidenceType.TESTIMONY.value]
    types = [EvidenceType.TESTIMONY.value]
    if _NUMERIC_CUES.search(answer_text):
        types.append(EvidenceType.COMPARISON_ANALOGY.value)
    return types


def _resolve_evidence_source(source_url: str, modality: str, registry: dict) -> str:
    """Parse source_url → domain → registry source_id.

    Wayback Machine URLs embed the original URL in the path; we extract that
    original domain so archived Reuters/BBC/gov pages get their real trust score
    instead of the generic webarchive fallback (ST=0.40).
    """
    if modality == "annotator_knowledge":
        return "annotator_knowledge"
    if not source_url:
        return "unknown_web"
    try:
        # Unwrap Wayback Machine archives before domain resolution
        m = _ARCHIVE_RE.search(source_url)
        if m:
            embedded = m.group(1)
            if not embedded.startswith(("http://", "https://")):
                embedded = "https://" + embedded
            source_url = embedded

        parsed = urlparse(source_url)
        domain = (parsed.netloc or "").lower().removeprefix("www.")
        if not domain:
            return "unknown_web"
        return resolve_source_id(domain, modality, registry)
    except Exception:
        return "unknown_web"


class AveritecConverter(DatasetConverter):
    """Converts AVeriTeC JSON records to unified v3.0 JSONL."""

    def __init__(self, registry: dict | None = None):
        """
        registry: source trust registry dict {source_id: record} for source_id resolution.
                  If None, resolve_source_id falls back to TLD heuristics and 'unknown_web'.
        """
        self._registry: dict = registry or {}

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

    def _build_evidence(self, rec: dict, rec_id: str) -> list[dict]:
        label = _normalize_label(rec.get("label"))
        stance = _verdict_to_stance(label)
        strategies = [
            s.strip().lower() for s in (rec.get("fact_checking_strategies") or [])
        ]

        items = []
        for qi, q in enumerate(rec.get("questions") or [], start=1):
            qtext = (q.get("question") or "").strip()
            for ai, a in enumerate(q.get("answers") or [], start=1):
                evidence_id = f"{rec_id}-q{qi}-a{ai}"
                ans_text = str(a.get("answer") or "").strip()
                source_medium = a.get("source_medium")
                source_url = str(a.get("source_url") or "").strip()
                ans_type_key = str(a.get("answer_type") or "").strip().lower()
                ans_type = _ANSWER_TYPE_MAP.get(ans_type_key, "unanswerable")
                modality = _medium_to_modality(
                    source_medium, source_url=source_url, answer_type=ans_type
                )
                text = f"{qtext} {ans_text}".strip() if qtext else ans_text

                source_id = _resolve_evidence_source(
                    source_url, modality, self._registry
                )
                is_raw = _IS_FROM_ANSWER_TYPE.get(ans_type, 0.6)
                # IS cannot exceed source trust: a Facebook post cannot provide
                # IS=0.8 inference regardless of how direct the answer is.
                st = get_source_trust(source_id, self._registry)
                is_ = is_raw if st >= 0.45 else max(0.10, min(is_raw, st))
                evidence_types = _infer_evidence_types_basic(
                    modality, ans_type, ans_text
                )

                items.append(
                    {
                        "evidence_id": evidence_id,
                        "text": text or None,
                        "triples": [],
                        "triple_source": None,
                        "modality": modality,
                        "stance": stance.value if stance else None,
                        "source_id": source_id,
                        "evidence_types": evidence_types,
                        "inference_strength": is_,
                        "source_url": a.get("source_url"),
                        # Private key stripped before output
                        "_answer_type": ans_type,
                    }
                )

        # Post-pass 1: add inference to abstractive items when claim has multi-source evidence
        src_urls = {e["source_url"] for e in items if e.get("source_url")}
        if len(src_urls) >= 2:
            for item in items:
                if item["_answer_type"] == "abstractive":
                    if EvidenceType.INFERENCE.value not in item["evidence_types"]:
                        item["evidence_types"].append(EvidenceType.INFERENCE.value)

        # Post-pass 2: enrich evidence_types from claim-level fact_checking_strategies.
        # Only applied to textual (non-perceptual, non-unanswerable) items so that
        # strategy signals don't override direct sensory evidence type assignments.
        for strategy in strategies:
            et_to_add = _STRATEGY_EVIDENCE_TYPE_MAP.get(strategy)
            if not et_to_add:
                continue
            for item in items:
                modality = item.get("modality", "")
                if modality not in _PERCEPTUAL and item["evidence_types"]:
                    if et_to_add not in item["evidence_types"]:
                        item["evidence_types"].append(et_to_add)

        return items

    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        oid = str(raw_record.get("id") or rec_id)
        claim_text = (raw_record.get("claim") or "").strip()
        label = _normalize_label(raw_record.get("label"))

        evidence_raw = self._build_evidence(raw_record, oid)

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
                    "source_id": "unknown_web",
                    "evidence_types": [],
                    "inference_strength": 0.0,
                    "source_url": None,
                }
            ]

        evidence_types_all = sorted(
            {t for e in evidence_out for t in e.get("evidence_types", [])}
        )

        return {
            "schema_version": "3.0",
            "id": oid,
            "claim": claim_text,
            "verdict": {
                "label": label.value,
                "justification": raw_record.get("justification"),
                "derivation_method": "annotated",
            },
            "epistemic": {
                "evidence_types_all": evidence_types_all,
                "assignment_method": "rule_based",
            },
            "claim_triples": None,
            "reasoning": {
                "strategy": _to_strategy(
                    raw_record.get("fact_checking_strategies") or []
                )
            },
            "evidence": evidence_out,
            "provenance": {
                "dataset": "averitec",
                "split": None,
                "context_id": "schlichtkrull2023averitec",
            },
            "meta": {
                "schema_version": "3.0",
                "created_utc": utc_now_iso(),
            },
        }
