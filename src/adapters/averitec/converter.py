from __future__ import annotations

from src.core.ports.dataset.converter import DatasetConverter
from src.core.claims.labels import CONFIDENCE_WEIGHTS, PramanaLabel


class AveritecConverter(DatasetConverter):
    """
    Converts AVeriTeC JSON records to unified v2.0 JSONL.
    Implements the DatasetConverter port — plug in by registering in the
    pipeline's converter registry.

    Status: scaffold — full implementation in Phase 3 refactor.
    Current logic lives in src/pipelines/convert_averitec_json.py.
    """

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
        from src.pipelines.convert_averitec_json import infer_proof_types, pick_primary
        label = raw_record.get("label", "")
        claim_types = raw_record.get("claim_types") or []
        strategies = raw_record.get("fact_checking_strategies") or []
        proof_types = infer_proof_types(label, claim_types, strategies, [], [])
        primary = pick_primary(proof_types)
        weight = CONFIDENCE_WEIGHTS.get(PramanaLabel(primary), 0.70)
        return primary, proof_types, weight

    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        from src.pipelines.convert_averitec_json import convert_one
        return convert_one(raw_record, rec_id)
