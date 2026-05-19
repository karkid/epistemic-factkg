"""JSON schema for v3.0 unified records (epistemic_factkg.jsonl).

This schema defines the contract for all records produced by adapters.
"""

from src.epistemic.enums import EvidenceType, Verdict, EvidenceStance, ReasoningStrategy

_EVIDENCE_TYPE_VALUES = [p.value for p in EvidenceType]
_VERDICT_VALUES = [v.value for v in Verdict]
_STANCE_VALUES = [s.value for s in EvidenceStance]
_STRATEGY_VALUES = [s.value for s in ReasoningStrategy] + [None]

CLAIM_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "schema_version",
        "id",
        "claim",
        "verdict",
        "epistemic",
        "evidence",
        "provenance",
        "meta",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "const": "3.0"},
        "id": {"type": "string"},
        "claim": {"type": "string"},
        "verdict": {
            "type": "object",
            "required": ["label", "justification", "derivation_method"],
            "additionalProperties": False,
            "properties": {
                "label": {
                    "type": "string",
                    "enum": _VERDICT_VALUES,
                },
                "justification": {"type": ["string", "null"]},
                "derivation_method": {
                    "type": "string",
                    "enum": ["aggregated_from_evidence", "annotated"],
                },
            },
        },
        "epistemic": {
            "type": "object",
            "required": ["evidence_types_all", "assignment_method"],
            "additionalProperties": False,
            "properties": {
                "evidence_types_all": {
                    "type": "array",
                    "items": {"type": "string", "enum": _EVIDENCE_TYPE_VALUES},
                },
                "assignment_method": {
                    "type": "string",
                    "enum": ["heuristic", "rule_based", "annotated", "llm_generated", "simulator"],
                },
            },
        },
        "claim_triples": {
            "type": ["array", "null"],
            "items": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "string"},
            },
        },
        "reasoning": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "structural": {
                    "type": ["string", "null"],
                    "enum": [
                        "one_hop",
                        "multi_hop",
                        "conjunction",
                        "negation",
                        "absence",
                        None,
                    ],
                },
                "strategy": {
                    "type": ["string", "null"],
                    "enum": _STRATEGY_VALUES,
                },
            },
        },
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "evidence_id",
                    "text",
                    "triples",
                    "triple_source",
                    "modality",
                    "stance",
                    "evidence_types",
                    "source_id",
                    "inference_strength",
                ],
                "additionalProperties": False,
                "properties": {
                    "evidence_id": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "triples": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "string"},
                        },
                    },
                    "triple_source": {
                        "type": ["string", "null"],
                        "enum": ["ground_truth", "extracted", None],
                    },
                    "modality": {
                        "type": "string",
                        "enum": [
                            "sensor",
                            "web_text",
                            "pdf",
                            "web_table",
                            "image",
                            "video",
                            "audio",
                            "annotator_knowledge",
                            "unanswerable",
                            "other",
                        ],
                    },
                    "stance": {
                        "type": "string",
                        "enum": _STANCE_VALUES,
                    },
                    "evidence_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": _EVIDENCE_TYPE_VALUES},
                    },
                    "source_id": {"type": "string", "minLength": 1},
                    "inference_strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "source_url": {"type": ["string", "null"]},
                },
            },
        },
        "provenance": {
            "type": "object",
            "required": ["dataset", "split", "context_id"],
            "additionalProperties": False,
            "properties": {
                "dataset": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                "split": {
                    "type": ["string", "null"],
                    "enum": ["train", "dev", "test", None],
                },
                "context_id": {"type": ["string", "null"]},
            },
        },
        "meta": {
            "type": "object",
            "required": ["schema_version", "created_utc"],
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string", "const": "3.0"},
                "created_utc": {"type": "string"},
                "template_type": {"type": ["string", "null"]},
                "is_shortcut_breaking": {"type": ["boolean", "null"]},
            },
        },
    },
}
