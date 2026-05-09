from __future__ import annotations

from urllib.parse import unquote

from src.core.ports.dataset.converter import DatasetConverter
from src.core.claims.labels import CONFIDENCE_WEIGHTS, PramanaLabel
from src.utils.time import utc_now_iso


_SPATIAL_PREDS = {"inside", "ontopof", "near", "in", "on"}
_AFFORDANCE_PREDS = {"breakable", "pickupable", "openable", "istoggleable"}

_LABEL_MAP = {
    "support": "supported",
    "supported": "supported",
    "refute": "refuted",
    "refuted": "refuted",
}


def _decode_uri(uri: str) -> str:
    return unquote(str(uri)) if isinstance(uri, str) else str(uri)


def _rel_name(uri: str) -> str:
    return uri.split("/")[-1] if isinstance(uri, str) else "unknown_relation"


def _normalize_label(lbl: str | None) -> str | None:
    if lbl is None:
        return None
    return _LABEL_MAP.get(lbl.strip().lower(), lbl.strip().lower())


def _convert_triples(raw: list) -> list[list[str]]:
    out = []
    for t in (raw or []):
        if isinstance(t, (list, tuple)) and len(t) == 3:
            s, p, o = t
            out.append([_decode_uri(str(s)), _rel_name(str(p)), str(o)])
    return out


def _classify_strategy(predicate: str, ev_triples: list) -> str | None:
    pred = predicate.lower()
    if not ev_triples:
        return "absence_detection"
    if pred in _SPATIAL_PREDS:
        return "spatial_reasoning"
    if pred in _AFFORDANCE_PREDS:
        return "action_testing"
    return "direct_observation"


def _make_justification(label: str, predicate: str, claim_triples: list, ev_triples: list) -> str | None:
    pred = predicate.lower()
    if pred == "temperature":
        observed = ev_triples[0][2] if ev_triples else "unknown"
        claimed = claim_triples[0][2] if claim_triples else "unknown"
        if label == "supported":
            return f"Sensor shows temperature={observed}, matching claim."
        return f"Sensor shows temperature={observed}, contradicting claimed value ({claimed})."
    if pred in _SPATIAL_PREDS:
        verb = "confirms" if label == "supported" else "contradicts"
        return f"Sensor {verb} spatial relation ({predicate})."
    if not ev_triples:
        return "No sensor evidence found to support the claim."
    verb = "matches" if label == "supported" else "contradicts"
    return f"Sensor observation {verb} the claim."


class AI2ThorConverter(DatasetConverter):
    """Converts AI2THOR JSONL records to unified v2.0 JSONL."""

    @property
    def dataset_name(self) -> str:
        return "ai2thor"

    def infer_pramana(self, raw_record: dict) -> tuple[str, list[str], float]:
        ev_triples = (raw_record.get("evidence") or {}).get("evidence_triples") or []
        primary = (
            PramanaLabel.NON_APPREHENSION.value if not ev_triples
            else PramanaLabel.PERCEPTION.value
        )
        weight = CONFIDENCE_WEIGHTS.get(PramanaLabel(primary), 0.70)
        return primary, [primary], weight

    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        oid = raw_record.get("id") or rec_id
        claim_text = raw_record.get("claim", "").strip()
        label = _normalize_label(raw_record.get("label"))

        raw_claim_triples = raw_record.get("claim_triples") or []
        reasoning = raw_record.get("reasoning") or {}
        evidence = raw_record.get("evidence") or {}
        context = raw_record.get("context") or {}
        meta = raw_record.get("meta") or {}

        raw_ev_triples = evidence.get("evidence_triples") or []
        evidence_urls = evidence.get("evidence_urls") or []
        evidence_extract = evidence.get("extract") or None

        claim_triples = _convert_triples(raw_claim_triples)
        ev_triples = _convert_triples(raw_ev_triples)

        first = claim_triples[0] if claim_triples else ["", "unknown_relation", ""]
        predicate = first[1] if len(first) == 3 else "unknown_relation"

        pramana_primary, pramana_all, confidence_weight = self.infer_pramana(raw_record)
        strategy = _classify_strategy(predicate, ev_triples)
        justification = _make_justification(label or "", predicate, claim_triples, ev_triples)

        structural = reasoning.get("structural")
        if structural:
            structural = structural.replace("-", "_")

        created_utc = meta.get("created_utc") or utc_now_iso()
        context_id = context.get("context_id")
        source_url = evidence_urls[0] if evidence_urls else None

        if pramana_primary == PramanaLabel.NON_APPREHENSION.value:
            stance = "absent"
        elif label == "supported":
            stance = "supports"
        elif label == "refuted":
            stance = "refutes"
        else:
            stance = "unknown"

        return {
            "schema_version": "2.0",
            "id": oid,
            "claim": claim_text,
            "verdict": {
                "label": label,
                "justification": justification,
            },
            "epistemic": {
                "pramana_primary": pramana_primary,
                "pramana_all": pramana_all,
                "confidence_weight": confidence_weight,
                "assignment_method": "rule_based",
            },
            "claim_triples": claim_triples if claim_triples else None,
            "reasoning": {
                "structural": structural,
                "strategy": strategy,
            } if structural else None,
            "evidence": [
                {
                    "evidence_id": f"{oid}-e0",
                    "text": evidence_extract,
                    "triples": ev_triples,
                    "triple_source": "ground_truth",
                    "modality": "simulation_state",
                    "stance": stance,
                    "source_url": source_url,
                }
            ],
            "provenance": {
                "dataset": "ai2thor",
                "split": context.get("split"),
                "context_id": context_id,
            },
            "meta": {
                "schema_version": "2.0",
                "created_utc": created_utc,
            },
        }
