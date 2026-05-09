from __future__ import annotations

from urllib.parse import unquote

from src.core.ports.dataset.converter import DatasetConverter
from src.core.claims.labels import (
    combine_pramana_weights,
    EvidenceStance,
    Pramana,
    Verdict,
)
from src.utils.time import utc_now_iso


_SPATIAL_PREDS = {"inside", "ontopof", "near", "in", "on"}
_AFFORDANCE_PREDS = {"breakable", "pickupable", "openable", "istoggleable"}

# Map raw label strings from the AI2THOR claim generator to Verdict enum.
# The generator emits "support"/"supported" for true claims and
# "refute"/"refuted" for corrupted/false claims — both spellings are normalised.
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


def _classify_strategy(predicate: str, ev_triples: list) -> str | None:
    pred = predicate.lower()
    if not ev_triples:
        return "absence_detection"
    if pred in _SPATIAL_PREDS:
        return "spatial_reasoning"
    if pred in _AFFORDANCE_PREDS:
        return "action_testing"
    return "direct_observation"


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
    if pred in _SPATIAL_PREDS:
        verb = "confirms" if label == Verdict.SUPPORTED else "contradicts"
        return f"Sensor {verb} spatial relation ({predicate})."
    if not ev_triples:
        return "No sensor evidence found to support the claim."
    verb = "matches" if label == Verdict.SUPPORTED else "contradicts"
    return f"Sensor observation {verb} the claim."


def _label_to_stance(label: Verdict | None) -> str | None:
    """
    Derive evidence stance from verdict for AI2THOR records.

    Heuristic: AI2THOR claims have a single decisive evidence item whose
    stance mirrors the verdict directly.  For absence claims the stance is
    always 'absent' regardless of the verdict label (absence IS the evidence).
    """
    if label == Verdict.SUPPORTED:
        return EvidenceStance.SUPPORTS.value
    if label == Verdict.REFUTED:
        return EvidenceStance.REFUTES.value
    return None


class AI2ThorConverter(DatasetConverter):
    """Converts AI2THOR JSONL records to unified v2.0 JSONL.

    Handles two source formats:
    - Legacy dict format: evidence is a dict with evidence_triples, evidence_source, etc.
    - v2.0 list format: evidence is already a list of v2.0 evidence objects (from ClaimGenerator).
    """

    @property
    def dataset_name(self) -> str:
        return "ai2thor"

    def infer_pramana(self, raw_record: dict) -> tuple[str, list[str], float]:
        """
        Assign Pramana type for AI2THOR records.

        Heuristic (from research proposal §5, Pramana table):
        - PERCEPTION (0.90): default for all AI2THOR claims — the simulator
          provides direct sensor observations of object properties and spatial
          relations, corresponding to Pratyaksham (direct sensory knowledge).
        - NON_APPREHENSION (0.80): assigned when there are no evidence triples,
          meaning the claim asserts the *absence* of an object or state —
          corresponds to Anupalabdhi (knowledge from confirmed absence).
          Only reliable here because the simulator gives complete world state.
        """
        evidence = raw_record.get("evidence") or {}
        if isinstance(evidence, list):
            has_ev = any(e.get("triples") for e in evidence if isinstance(e, dict))
        else:
            has_ev = bool(evidence.get("evidence_triples"))

        primary = Pramana.PERCEPTION if has_ev else Pramana.NON_APPREHENSION
        weight = combine_pramana_weights([primary.value])
        return primary.value, [primary.value], weight

    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        evidence = raw_record.get("evidence") or {}
        if isinstance(evidence, list):
            return self._from_v2(raw_record, rec_id, evidence)
        return self._from_legacy(raw_record, rec_id, evidence)

    def _from_v2(self, raw: dict, rec_id: str, evidence: list) -> dict:
        """Pass-through for records already in v2.0 format — just decode URIs."""
        oid = raw.get("id") or rec_id
        label = _normalize_label(
            raw.get("label") or (raw.get("verdict") or {}).get("label")
        )

        claim_triples = _convert_triples(raw.get("claim_triples") or [])
        first = claim_triples[0] if claim_triples else ["", "unknown_relation", ""]
        predicate = first[1] if len(first) == 3 else "unknown_relation"

        pramana_primary, pramana_all, confidence_weight = self.infer_pramana(raw)

        ev_out = []
        for ev in evidence:
            decoded_triples = _convert_triples(ev.get("triples") or [])
            strategy = ev.get("strategy") or _classify_strategy(
                predicate, decoded_triples
            )
            # Stance must be a valid EvidenceStance value or None — never "unknown"
            raw_stance = ev.get("stance")
            stance = (
                raw_stance if raw_stance in {s.value for s in EvidenceStance} else None
            )
            ev_out.append(
                {
                    "evidence_id": ev.get("evidence_id", f"{oid}-e0"),
                    "text": ev.get("text"),
                    "triples": decoded_triples,
                    "triple_source": ev.get("triple_source", "ground_truth"),
                    "modality": ev.get("modality", "simulation_state"),
                    "stance": stance,
                    "source_url": ev.get("source_url"),
                }
            )

        reasoning = raw.get("reasoning") or {}
        structural = reasoning.get("structural")
        if structural:
            structural = structural.replace("-", "_")
        strategy = reasoning.get("strategy") or (
            _classify_strategy(predicate, ev_out[0].get("triples", []))
            if ev_out
            else None
        )

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
            "schema_version": "2.0",
            "id": oid,
            "claim": raw.get("claim", "").strip(),
            "verdict": {
                "label": label.value if label else None,
                "justification": justification,
            },
            "epistemic": {
                "pramana_primary": pramana_primary,
                "pramana_all": pramana_all,
                "confidence_weight": confidence_weight,
                "assignment_method": "rule_based",
            },
            "claim_triples": claim_triples if claim_triples else None,
            "reasoning": {"structural": structural, "strategy": strategy}
            if structural
            else None,
            "evidence": ev_out,
            "provenance": {
                "dataset": provenance.get("dataset", "ai2thor"),
                "split": provenance.get("split"),
                "context_id": provenance.get("context_id"),
            },
            "meta": {
                "schema_version": "2.0",
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
        evidence_extract = evidence.get("extract") or None

        claim_triples = _convert_triples(raw_claim_triples)
        ev_triples = _convert_triples(raw_ev_triples)

        first = claim_triples[0] if claim_triples else ["", "unknown_relation", ""]
        predicate = first[1] if len(first) == 3 else "unknown_relation"

        pramana_primary, pramana_all, confidence_weight = self.infer_pramana(raw)
        strategy = _classify_strategy(predicate, ev_triples)
        justification = _make_justification(label, predicate, claim_triples, ev_triples)

        structural = reasoning.get("structural")
        if structural:
            structural = structural.replace("-", "_")

        # Absence claims (non_apprehension) use the ABSENT stance — the absence
        # of evidence triples IS the evidence.  All other stances mirror the verdict.
        if pramana_primary == Pramana.NON_APPREHENSION.value:
            stance = EvidenceStance.ABSENT.value
        else:
            stance = _label_to_stance(label)

        return {
            "schema_version": "2.0",
            "id": oid,
            "claim": raw.get("claim", "").strip(),
            "verdict": {
                "label": label.value if label else None,
                "justification": justification,
            },
            "epistemic": {
                "pramana_primary": pramana_primary,
                "pramana_all": pramana_all,
                "confidence_weight": confidence_weight,
                "assignment_method": "rule_based",
            },
            "claim_triples": claim_triples if claim_triples else None,
            "reasoning": {"structural": structural, "strategy": strategy}
            if structural
            else None,
            "evidence": [
                {
                    "evidence_id": f"{oid}-e0",
                    "text": evidence_extract,
                    "triples": ev_triples,
                    "triple_source": "ground_truth",
                    "modality": "simulation_state",
                    "stance": stance,
                    "source_url": evidence_urls[0] if evidence_urls else None,
                }
            ],
            "provenance": {
                "dataset": "ai2thor",
                "split": context.get("split"),
                "context_id": context.get("context_id"),
            },
            "meta": {
                "schema_version": "2.0",
                "created_utc": meta.get("created_utc") or utc_now_iso(),
            },
        }
