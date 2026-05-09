from __future__ import annotations

from src.core.ports.dataset.validator import DatasetValidator
from src.core.claims.labels import EvidenceStance, Verdict

_VALID_AVERITEC_PRAMANAS = {
    "perception",
    "testimony",
    "comparison_analogy",
    "inference",
    "postulation_derivation",
}


class AveritecValidator(DatasetValidator):
    """AVeriTeC-specific consistency checks on unified v2.0 records."""

    @property
    def dataset_name(self) -> str:
        return "averitec"

    def check(self, record: dict) -> list[str]:
        msgs = []
        verdict_label = (record.get("verdict") or {}).get("label")
        evidence = record.get("evidence") or []
        pramana = (record.get("epistemic") or {}).get("pramana_primary")

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
        if pramana == "non_apprehension":
            msgs.append(
                "AVeriTeC record has pramana=non_apprehension, which is invalid "
                "for web-sourced evidence — confirmed absence requires a closed-world state."
            )

        if pramana and pramana not in _VALID_AVERITEC_PRAMANAS:
            msgs.append(f"AVeriTeC record has unexpected pramana_primary: {pramana!r}.")

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

        # Evidence text coverage — all-null text means no textual content for GNN
        if evidence and all(e.get("text") is None for e in evidence):
            msgs.append(
                "AVeriTeC record has no textual evidence content (all evidence.text is null)."
            )

        return msgs
