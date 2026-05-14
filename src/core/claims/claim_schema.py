from src.core.claims.labels import EvidenceType, Verdict, EvidenceStance

_EVIDENCE_TYPE_VALUES = [p.value for p in EvidenceType]
_VERDICT_VALUES = [v.value for v in Verdict]
_STANCE_VALUES = [s.value for s in EvidenceStance] + [None]

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
                    "type": ["string", "null"],
                    "enum": _VERDICT_VALUES + [None],
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
                    "enum": ["heuristic", "rule_based", "annotated", "llm_generated"],
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
                    "enum": [
                        "direct_observation",
                        "absence_detection",
                        "spatial_reasoning",
                        "written_evidence",
                        "numerical_comparison",
                        "multi_source_synthesis",
                        "action_testing",
                        None,
                    ],
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
                    "evidence_id": {"type": "string"},
                    "text": {"type": ["string", "null"]},
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
                            "simulation_state",
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
                        "type": ["string", "null"],
                        "enum": _STANCE_VALUES,
                    },
                    "evidence_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": _EVIDENCE_TYPE_VALUES},
                    },
                    "source_id": {"type": "string"},
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
            },
        },
    },
}
