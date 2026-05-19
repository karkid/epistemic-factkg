from __future__ import annotations

from src.ports.validator import DatasetValidator
from src.epistemic.enums import EvidenceStance, EvidenceType, Verdict

_VALID_AVERITEC_EVIDENCE_TYPES = {
    EvidenceType.PERCEPTION.value,
    EvidenceType.TESTIMONY.value,
    EvidenceType.COMPARISON_ANALOGY.value,
    EvidenceType.INFERENCE.value,
    EvidenceType.POSTULATION_DERIVATION.value,
}


class AveritecValidator(DatasetValidator):
    """AVeriTeC-specific consistency checks on unified v3.0 records."""

    @property
    def dataset_name(self) -> str:
        return "averitec"

    def check(self, record: dict) -> list[str]:
        msgs = []
        verdict_label = (record.get("verdict") or {}).get("label")
        evidence = record.get("evidence") or []
        evidence_types_all = (record.get("epistemic") or {}).get(
            "evidence_types_all", []
        )

        if not evidence:
            msgs.append("AVeriTeC record has no evidence items.")

        if verdict_label is None and evidence:
            msgs.append(
                "verdict.label is null but evidence is present — "
                "check if this is truly a blind-test record."
            )

        if verdict_label is not None and not evidence:
            msgs.append(
                "verdict.label is set but evidence is empty — "
                "cannot verify verdict without evidence."
            )

        if record.get("claim_triples") is not None:
            msgs.append(
                "AVeriTeC record unexpectedly has claim_triples (should be null)."
            )

        # non_apprehension is invalid for AVeriTeC — web text cannot confirm absence
        if EvidenceType.NON_APPREHENSION.value in evidence_types_all:
            msgs.append(
                "AVeriTeC record has non_apprehension evidence type, which is invalid "
                "for web-sourced evidence — confirmed absence requires a closed-world state."
            )

        if verdict_label == Verdict.CONFLICTING_EVIDENCE:
            stances = {e.get("stance") for e in evidence if e.get("stance")}
            if stances and stances <= {EvidenceStance.SUPPORTS.value}:
                msgs.append(
                    "conflicting_evidence verdict but all evidence stances are 'supports' — "
                    "expected mixed stances."
                )
            if stances and stances <= {EvidenceStance.REFUTES.value}:
                msgs.append(
                    "conflicting_evidence verdict but all evidence stances are 'refutes' — "
                    "expected mixed stances."
                )

        if evidence and all(e.get("text") is None for e in evidence):
            msgs.append(
                "AVeriTeC record has no textual evidence content (all evidence.text is null)."
            )

        assignment_method = (record.get("epistemic") or {}).get("assignment_method")
        if assignment_method != "heuristic":
            msgs.append(
                f"AVeriTeC record has assignment_method={assignment_method!r}, expected 'heuristic'."
            )

        _decisive = {EvidenceStance.SUPPORTS.value, EvidenceStance.REFUTES.value}
        for item in evidence:
            eid = item.get("evidence_id", "?")
            # sensor modality is AI2THOR-only — web records must not use it
            if item.get("modality") == "sensor":
                msgs.append(
                    f"AVeriTeC evidence {eid}: modality='sensor' is invalid for web-sourced evidence."
                )
            # items with a decisive stance must have at least one evidence_type
            if item.get("stance") in _decisive and not item.get("evidence_types"):
                msgs.append(
                    f"AVeriTeC evidence {eid}: stance={item.get('stance')!r} but evidence_types is empty."
                )

        return msgs
