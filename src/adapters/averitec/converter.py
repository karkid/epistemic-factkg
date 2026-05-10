from __future__ import annotations

import re

from src.core.ports.dataset.converter import DatasetConverter
from src.core.claims.labels import (
    combine_pramana_weights,
    EvidenceStance,
    Pramana,
    Verdict,
)
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

# ── Heuristic modality buckets ────────────────────────────────────────────────
# Pratyaksham (Perception, 0.85–0.95): direct sensory evidence — image/video/audio
# embedded in AVeriTeC answers (rare but possible for media fact-checks).
_PERCEPTUAL = {"image", "video", "audio"}

# Shabda (Testimony, 0.80–0.90): textual sources — web pages, PDFs, tables.
# This is the dominant evidence type in AVeriTeC.
_TEXTUAL = {"web_text", "pdf", "web_table", "other"}

# ── Heuristic signal: Upamanam (Comparison, 0.70–0.80) ───────────────────────
# Triggered when the answer text contains numerical/statistical comparison cues
# (%, rankings, GDP, magnitudes). Kept narrow to avoid false positives.
_NUMERIC_CUES = re.compile(
    r"\b(%|percent|percentage|largest|smallest|rank|gdp|million|billion|trillion)\b",
    re.IGNORECASE,
)

# ── Priority order for pramana_primary selection ──────────────────────────────
# Sorted by specificity/distinctiveness, NOT by confidence weight.
# NON_APPREHENSION is always excluded — confirmed absence cannot be established
# from web text; restricted to AI2THOR closed-world records only.
# Testimony is listed last despite a higher weight (0.80) than
# comparison_analogy (0.65) or inference (0.55) — it is the default fallback,
# and the more specific types are more informative as the primary label.
_DEFAULT_PRIMARY_ORDER: list[Pramana] = [
    Pramana.PERCEPTION,
    Pramana.COMPARISON_ANALOGY,
    Pramana.INFERENCE,
    Pramana.TESTIMONY,
    # POSTULATION_DERIVATION excluded per ADR-011 (no trigger rules; insufficient training samples)
]


def _build_primary_order(custom_order: list[str] | None = None) -> list[Pramana]:
    """Return the pramana priority list for pramana_primary selection.

    custom_order: list of pramana string values from config, highest priority
    first. NON_APPREHENSION is always stripped — not valid for web evidence.
    When custom_order is None the module default is returned.
    """
    if custom_order:
        return [Pramana(p) for p in custom_order if p != Pramana.NON_APPREHENSION.value]
    return list(_DEFAULT_PRIMARY_ORDER)


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
        # Empty medium: unanswerable if the annotator found nothing; else other
        return "unanswerable" if answer_type == "unanswerable" else "other"

    sm_raw = str(source_medium).strip()
    sm = sm_raw.lower().replace(" ", "_")

    if sm == "metadata":
        # Annotator's own knowledge or derived calculation — not a web source
        return "annotator_knowledge"

    if sm == "other":
        # AVeriTeC "Other": calculator/search tool with a real URL → web_text
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


def _infer_pramana(
    modalities: set[str],
    src_urls: set[str],
    answer_types: list[str],
    answers_text: str,
    weights: dict | None = None,
    primary_order: list[Pramana] | None = None,
) -> tuple[str, list[str], float]:
    """
    Heuristic Pramana assignment for AVeriTeC records.

    Rules (from research proposal §5, Pramana table):

    PERCEPTION (Pratyaksham, 0.90):
        Any evidence item uses a perceptual modality (image, video, audio).
        Rare in AVeriTeC but present in media/image fact-checks.

    TESTIMONY (Shabda, 0.85):
        Any evidence item uses a textual source (web_text, pdf, web_table).
        Default for almost all AVeriTeC records — falls back if no other rule fires.

    COMPARISON_ANALOGY (Upamanam, 0.75):
        Answer text contains numerical/statistical comparison cues (%, GDP, rank…).
        Captures claims verified by comparing magnitudes against known benchmarks.

    INFERENCE (Anumanam, 0.70):
        ≥2 abstractive answers AND ≥2 distinct source URLs. Abstractive answers
        require synthesising information across sources — indicative of multi-hop
        reasoning rather than direct lookup.
        Threshold avoids over-assigning inference to simple single-source lookups.

    NON_APPREHENSION (Anupalabdhi): deliberately excluded.
        Confirmed absence cannot be established from web text alone;
        this Pramana is restricted to AI2THOR records where the simulator
        provides a complete, closed-world state.

    POSTULATION_DERIVATION (Arthapatti): no trigger rules yet (status: Limited/Future).

    When multiple types apply, _PRIMARY_ORDER selects the highest-confidence one
    as pramana_primary; all detected types are recorded in pramana_all.
    """
    proof_types: set[Pramana] = set()

    # Rule 1 — Perception: direct sensory/media evidence
    if modalities & _PERCEPTUAL:
        proof_types.add(Pramana.PERCEPTION)

    # Rule 2 — Testimony: any textual source (dominant in AVeriTeC)
    if modalities & _TEXTUAL:
        proof_types.add(Pramana.TESTIMONY)

    # Rule 3 — Comparison: numerical/statistical cues in answer text
    if _NUMERIC_CUES.search(answers_text):
        proof_types.add(Pramana.COMPARISON_ANALOGY)

    # Rule 4 — Inference: multi-source abstractive synthesis
    # Threshold: >=1 abstractive answer AND >=2 distinct URLs.
    # The URL guard prevents single-source lookups from being mis-labelled.
    # Lowered from >=2 abstractive: one synthesised answer across multiple sources
    # is genuine inference — the strict double requirement excluded too many records.
    n_abstractive = sum(1 for t in answer_types if t == "abstractive")
    if n_abstractive >= 1 and len(src_urls) >= 2:
        proof_types.add(Pramana.INFERENCE)

    # Fallback: pure-textual records with no other signal → Testimony
    if not proof_types:
        proof_types.add(Pramana.TESTIMONY)

    _order = primary_order if primary_order is not None else _DEFAULT_PRIMARY_ORDER
    sorted_types = sorted(
        proof_types,
        key=lambda p: _order.index(p) if p in _order else 99,
    )
    primary = sorted_types[0]
    weight = combine_pramana_weights([p.value for p in sorted_types], weights)
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

    def __init__(self, epistemic_config: dict | None = None):
        """
        epistemic_config: optional dict loaded from the `epistemic:` YAML section.
          confidence_weights:    dict[str, float] — override per-pramana weights
          pramana_priority_order: list[str]       — override primary selection order
        Omit (or pass None) to use the defaults from labels.py and _DEFAULT_PRIMARY_ORDER.
        """
        cfg = epistemic_config or {}
        raw_weights = cfg.get("confidence_weights")
        self._weights: dict | None = (
            {Pramana(k): float(v) for k, v in raw_weights.items()}
            if raw_weights
            else None
        )
        self._primary_order: list[Pramana] = _build_primary_order(
            cfg.get("pramana_priority_order")
        )

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
        return _infer_pramana(
            modalities,
            src_urls,
            answer_types,
            answers_text,
            weights=self._weights,
            primary_order=self._primary_order,
        )

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
                source_url = str(a.get("source_url") or "").strip()
                ans_type_key = str(a.get("answer_type") or "").strip().lower()
                ans_type = _ANSWER_TYPE_MAP.get(ans_type_key, "unanswerable")
                modality = _medium_to_modality(
                    source_medium, source_url=source_url, answer_type=ans_type
                )
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
            modalities,
            src_urls,
            answer_types,
            answers_text,
            weights=self._weights,
            primary_order=self._primary_order,
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
