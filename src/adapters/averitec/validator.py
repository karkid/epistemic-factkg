from __future__ import annotations

from src.core.ports.dataset.validator import DatasetValidator


class AveritecValidator(DatasetValidator):
    """
    AVeriTeC-specific consistency checks on unified v2.0 records.
    Register in the validator registry in validate_unified_dataset.py.
    """

    @property
    def dataset_name(self) -> str:
        return "averitec"

    def check(self, record: dict) -> list[str]:
        msgs = []
        verdict_label = (record.get("verdict") or {}).get("label")
        evidence = record.get("evidence") or []

        if not evidence:
            msgs.append("AVeriTeC record has no evidence items.")

        if verdict_label is None and evidence:
            msgs.append("verdict.label is null but evidence is present — check if this is truly a blind-test record.")

        if verdict_label is not None and not evidence:
            msgs.append("verdict.label is set but evidence is empty — cannot verify verdict without evidence.")

        if record.get("claim_triples") is not None:
            msgs.append("AVeriTeC record unexpectedly has claim_triples (should be null).")

        return msgs
