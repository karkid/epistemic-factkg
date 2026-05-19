from __future__ import annotations

from src.ports.validator import DatasetValidator
from src.epistemic.enums import EvidenceStance, EvidenceType

# All evidence types producible by AI2THOR strategy mapping:
#   direct_observation  → [perception]
#   absence_detection   → [perception, non_apprehension]
#   spatial_reasoning   → [perception, comparison_analogy]
#   action_testing      → [perception, inference]
_VALID_AI2THOR_EVIDENCE_TYPES = {
    EvidenceType.PERCEPTION.value,
    EvidenceType.NON_APPREHENSION.value,
    EvidenceType.COMPARISON_ANALOGY.value,
    EvidenceType.INFERENCE.value,
}

_DECISIVE_STANCES = {EvidenceStance.SUPPORTS.value, EvidenceStance.REFUTES.value}


class AI2ThorValidator(DatasetValidator):
    """AI2THOR-specific consistency checks on unified v3.0 records."""

    @property
    def dataset_name(self) -> str:
        return "ai2thor"

    def check(self, record: dict) -> list[str]:
        msgs = []
        evidence = record.get("evidence") or []

        if not evidence:
            msgs.append("AI2THOR record has no evidence items.")
            return msgs

        ev = evidence[0]
        stance = ev.get("stance")
        ev_triples = ev.get("triples") or []
        evidence_types_all = (record.get("epistemic") or {}).get(
            "evidence_types_all", []
        )
        is_absence = EvidenceType.NON_APPREHENSION.value in evidence_types_all

        # Non-apprehension (absence) supported claims must have no evidence triples
        if is_absence and EvidenceStance.SUPPORTS.value == stance and ev_triples:
            msgs.append(
                "AI2THOR supported absence claim has non-empty evidence triples."
            )

        # Non-apprehension claims must carry a decisive stance (supports or refutes)
        if is_absence and stance not in _DECISIVE_STANCES:
            msgs.append(
                f"AI2THOR non_apprehension claim expected supports/refutes stance, got {stance!r}."
            )

        structural = (record.get("reasoning") or {}).get("structural")
        if (
            not is_absence
            and structural != "absence"
            and record.get("claim_triples") is None
        ):
            msgs.append("AI2THOR record missing claim_triples.")

        if record.get("reasoning") is None:
            msgs.append("AI2THOR record missing reasoning block.")

        for et in evidence_types_all:
            if et not in _VALID_AI2THOR_EVIDENCE_TYPES:
                msgs.append(f"AI2THOR record has unexpected evidence_type: {et!r}.")

        # Per-item strict field checks
        assignment_method = (record.get("epistemic") or {}).get("assignment_method")
        if assignment_method != "simulator":
            msgs.append(
                f"AI2THOR record has assignment_method={assignment_method!r}, expected 'simulator'."
            )

        for item in evidence:
            eid = item.get("evidence_id", "?")
            if item.get("modality") != "sensor":
                msgs.append(
                    f"AI2THOR evidence {eid}: modality={item.get('modality')!r}, expected 'sensor'."
                )
            if item.get("source_id") != "sensor_perception":
                msgs.append(
                    f"AI2THOR evidence {eid}: source_id={item.get('source_id')!r}, expected 'sensor_perception'."
                )
            if item.get("inference_strength") != 1.0:
                msgs.append(
                    f"AI2THOR evidence {eid}: inference_strength={item.get('inference_strength')!r}, expected 1.0."
                )

        return msgs
