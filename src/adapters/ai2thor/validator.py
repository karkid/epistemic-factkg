from __future__ import annotations

from src.core.ports.dataset.validator import DatasetValidator
from src.core.claims.labels import EvidenceStance, EvidenceType


_VALID_AI2THOR_EVIDENCE_TYPES = {
    EvidenceType.PERCEPTION.value,
    EvidenceType.NON_APPREHENSION.value,
}


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

        # Absence claims must have no evidence triples — the missing object IS the evidence
        if stance == EvidenceStance.ABSENT.value and ev_triples:
            msgs.append(
                "AI2THOR absence claim (stance=absent) has non-empty evidence triples."
            )

        # non_apprehension evidence type must always pair with stance=absent
        if is_absence and stance != EvidenceStance.ABSENT.value:
            msgs.append(
                f"AI2THOR non_apprehension claim expected stance=absent, got {stance!r}."
            )

        structural = (record.get("reasoning") or {}).get("structural")
        # Absence claims legitimately have no claim_triples
        if (
            not is_absence
            and structural != "absence"
            and record.get("claim_triples") is None
        ):
            msgs.append("AI2THOR record missing claim_triples.")

        if record.get("reasoning") is None:
            msgs.append("AI2THOR record missing reasoning block.")

        # Only perception / non_apprehension are valid for simulator-sourced records
        for et in evidence_types_all:
            if et not in _VALID_AI2THOR_EVIDENCE_TYPES:
                msgs.append(
                    f"AI2THOR record has unexpected evidence_type: {et!r}."
                )

        return msgs
