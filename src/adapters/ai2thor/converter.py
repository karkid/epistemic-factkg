from __future__ import annotations

from urllib.parse import unquote

from src.ports.converter import DatasetConverter
from src.epistemic.enums import (
    EvidenceStance,
    EvidenceType,
    Verdict,
)
from src.adapters.ai2thor.claims.strategy import (
    _classify_strategy,
    _infer_evidence_types,
    _label_to_stance,
    _to_strategy,
)
from src.utils.time import utc_now_iso


_LABEL_MAP: dict[str, Verdict] = {
    "support": Verdict.SUPPORTED,
    "supported": Verdict.SUPPORTED,
    "refute": Verdict.REFUTED,
    "refuted": Verdict.REFUTED,
}


def _decode_uri(uri: str) -> str:
    return unquote(str(uri)) if isinstance(uri, str) else str(uri)


def _rel_name(uri: str) -> str:
    return uri.split("/")[-1] if isinstance(uri, str) else "unknown_relation"


def _normalize_label(lbl: str | None) -> Verdict | None:
    if lbl is None:
        return None
    return _LABEL_MAP.get(str(lbl).strip().lower())


def _convert_triples(raw: list) -> list[list[str]]:
    out = []
    for t in raw or []:
        if isinstance(t, (list, tuple)) and len(t) == 3:
            s, p, o = t
            out.append([_decode_uri(str(s)), _rel_name(str(p)), _decode_uri(str(o))])
    return out


def _make_justification(
    label: Verdict | None, predicate: str, claim_triples: list, ev_triples: list
) -> str | None:
    pred = predicate.lower()
    if pred == "temperature":
        observed = ev_triples[0][2] if ev_triples else "unknown"
        claimed = claim_triples[0][2] if claim_triples else "unknown"
        if label == Verdict.SUPPORTED:
            return f"Sensor shows temperature={observed}, matching claim."
        return f"Sensor shows temperature={observed}, contradicting claimed value ({claimed})."
    if pred in {"inside", "ontopof", "near", "in", "on"}:
        verb = "confirms" if label == Verdict.SUPPORTED else "contradicts"
        return f"Sensor {verb} spatial relation ({predicate})."
    if not ev_triples:
        return "No sensor evidence found to support the claim."
    verb = "matches" if label == Verdict.SUPPORTED else "contradicts"
    return f"Sensor observation {verb} the claim."


class AI2ThorConverter(DatasetConverter):
    """Converts AI2THOR JSONL records to unified v3.0 JSONL.

    Handles two source formats:
    - v3.0 list format: evidence already has all v3.0 fields (from updated generator).
      Pass-through: only normalises URIs, labels, and stance values.
    - Legacy dict format: evidence is a dict with evidence_triples, evidence_source, etc.
      Full mapping: derives strategy, evidence_types, stance from scratch.
    """

    @property
    def dataset_name(self) -> str:
        return "ai2thor"

    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        evidence = raw_record.get("evidence") or {}
        if isinstance(evidence, list):
            return self._from_v3(raw_record, rec_id, evidence)
        return self._from_legacy(raw_record, rec_id, evidence)

    def _from_v3(self, raw: dict, rec_id: str, evidence: list) -> dict:
        """Pass-through for records already in v3.0 format — decode URIs and normalise labels."""
        oid = raw.get("id") or rec_id
        label = _normalize_label(
            raw.get("label") or (raw.get("verdict") or {}).get("label")
        )

        claim_triples = _convert_triples(raw.get("claim_triples") or [])
        first = claim_triples[0] if claim_triples else ["", "unknown_relation", ""]
        predicate = first[1] if len(first) == 3 else "unknown_relation"

        ev_out = []
        for ev in evidence:
            decoded_triples = _convert_triples(ev.get("triples") or [])
            raw_stance = ev.get("stance")
            if raw_stance in {s.value for s in EvidenceStance}:
                stance = raw_stance
            elif raw_stance == "absent":
                # Legacy v2.0 used "absent" for confirmed-absence claims; migrate to
                # supports/refutes based on the verdict (ADR-028)
                stance = _label_to_stance(label)
            else:
                stance = _label_to_stance(label)
            # Preserve generator-set evidence_types; fall back to strategy inference
            evidence_types = ev.get("evidence_types") or _infer_evidence_types(
                _classify_strategy(predicate, decoded_triples), bool(decoded_triples)
            )
            ev_out.append(
                {
                    "evidence_id": ev.get("evidence_id", f"{oid}-e0"),
                    "text": ev.get("text") or "",
                    "triples": decoded_triples,
                    "triple_source": ev.get("triple_source", "ground_truth"),
                    "modality": "sensor",
                    "stance": stance,
                    "evidence_types": evidence_types,
                    "source_id": ev.get("source_id", "sensor_perception"),
                    "inference_strength": ev.get("inference_strength", 1.0),
                    "source_url": ev.get("source_url"),
                }
            )

        evidence_types_all = sorted(
            {t for ev in ev_out for t in ev.get("evidence_types", [])}
        )

        reasoning = raw.get("reasoning") or {}
        structural = reasoning.get("structural")
        if structural:
            structural = structural.replace("-", "_")
        raw_strategy = reasoning.get("strategy") or (
            _classify_strategy(predicate, ev_out[0].get("triples", []))
            if ev_out
            else None
        )
        strategy = _to_strategy(raw_strategy)

        verdict = raw.get("verdict") or {}
        justification = verdict.get("justification") or _make_justification(
            label,
            predicate,
            claim_triples,
            ev_out[0].get("triples", []) if ev_out else [],
        )

        provenance = raw.get("provenance") or {}
        meta = raw.get("meta") or {}

        return {
            "schema_version": "3.0",
            "id": oid,
            "claim": raw.get("claim", "").strip(),
            "verdict": {
                "label": label.value if label else None,
                "justification": justification,
                "derivation_method": verdict.get("derivation_method", "annotated"),
            },
            "epistemic": {
                "evidence_types_all": evidence_types_all,
                "assignment_method": "simulator",
            },
            "claim_triples": claim_triples if claim_triples else None,
            "reasoning": {"structural": structural, "strategy": strategy},
            "evidence": ev_out,
            "provenance": {
                "dataset": provenance.get("dataset", "ai2thor"),
                "split": provenance.get("split"),
                "context_id": provenance.get("context_id"),
            },
            "meta": {
                "schema_version": "3.0",
                "created_utc": meta.get("created_utc") or utc_now_iso(),
            },
        }

    def _from_legacy(self, raw: dict, rec_id: str, evidence: dict) -> dict:
        """Convert legacy dict-format evidence (original simulation output)."""
        oid = raw.get("id") or rec_id
        label = _normalize_label(raw.get("label"))

        raw_claim_triples = raw.get("claim_triples") or []
        reasoning = raw.get("reasoning") or {}
        context = raw.get("context") or {}
        meta = raw.get("meta") or {}

        raw_ev_triples = evidence.get("evidence_triples") or []
        evidence_urls = evidence.get("evidence_urls") or []
        evidence_extract = evidence.get("extract") or ""

        claim_triples = _convert_triples(raw_claim_triples)
        ev_triples = _convert_triples(raw_ev_triples)

        first = claim_triples[0] if claim_triples else ["", "unknown_relation", ""]
        predicate = first[1] if len(first) == 3 else "unknown_relation"

        raw_strategy = _classify_strategy(predicate, ev_triples)
        strategy = _to_strategy(raw_strategy)
        evidence_types = _infer_evidence_types(raw_strategy, bool(ev_triples))
        justification = _make_justification(label, predicate, claim_triples, ev_triples)

        structural = reasoning.get("structural")
        if structural:
            structural = structural.replace("-", "_")

        stance = _label_to_stance(label)

        return {
            "schema_version": "3.0",
            "id": oid,
            "claim": raw.get("claim", "").strip(),
            "verdict": {
                "label": label.value if label else None,
                "justification": justification,
                "derivation_method": "annotated",
            },
            "epistemic": {
                "evidence_types_all": evidence_types,
                "assignment_method": "simulator",
            },
            "claim_triples": claim_triples if claim_triples else None,
            "reasoning": {"structural": structural, "strategy": strategy},
            "evidence": [
                {
                    "evidence_id": f"{oid}-e0",
                    "text": evidence_extract,
                    "triples": ev_triples,
                    "triple_source": "ground_truth",
                    "modality": "sensor",
                    "stance": stance,
                    "evidence_types": evidence_types,
                    "source_id": "sensor_perception",
                    "inference_strength": 1.0,
                    "source_url": evidence_urls[0] if evidence_urls else None,
                }
            ],
            "provenance": {
                "dataset": "ai2thor",
                "split": context.get("split"),
                "context_id": context.get("context_id"),
            },
            "meta": {
                "schema_version": "3.0",
                "created_utc": meta.get("created_utc") or utc_now_iso(),
            },
        }
