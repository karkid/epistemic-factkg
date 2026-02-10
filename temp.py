import json
from collections import Counter
from datetime import datetime
from jsonschema import Draft7Validator

# ============================================================
# Unified schema (INLINE)
# ============================================================

UNIFIED_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["id", "claim", "verdict", "epistemic", "context", "meta"],
    "additionalProperties": False,

    "properties": {
        "id": {"type": "string"},
        "claim": {"type": "string"},

        "verdict": {
            "type": "object",
            "required": ["label", "justification", "annotator_confidence"],
            "properties": {
                "label": {
                    "type": ["string", "null"],
                    "enum": [
                        "supported",
                        "refuted",
                        "not_enough_evidence",
                        "conflicting_evidence",
                        "cherrypicking",
                        None
                    ]
                },
                "justification": {"type": ["string", "null"]},
                "annotator_confidence": {"type": ["number", "null"]}
            }
        },

        "epistemic": {
            "type": "object",
            "required": ["proof_type", "proof_type_rationale", "proof_confidence"],
            "properties": {
                "proof_type": {
                    "type": "string",
                    "enum": [
                        "perception",
                        "inference",
                        "testimony",
                        "comparison_analogy",
                        "postulation_derivation",
                        "non_apprehension"
                    ]
                },
                "proof_type_rationale": {"type": ["string", "null"]},
                "proof_confidence": {"type": ["number", "null"]}
            }
        },

        "claim_meta": {"type": ["object", "null"]},
        "claim_triples": {"type": ["array", "null"]},
        "reasoning": {"type": ["object", "null"]},
        "qa": {"type": ["array", "null"]},
        "evidence_items": {"type": ["array", "null"]},

        "context": {
            "type": "object",
            "required": ["context_id", "context_type", "generator"],
            "properties": {
                "context_id": {"type": ["string", "null"]},
                "context_type": {"type": "string"},
                "generator": {"type": "string"},
                "split": {"type": ["string", "null"]}
            }
        },

        "meta": {
            "type": "object",
            "required": ["created_utc", "notes"],
            "properties": {
                "created_utc": {"type": "string"},
                "notes": {"type": ["string", "null"]}
            }
        }
    }
}

# ============================================================
# Validator + summary
# ============================================================

def now():
    return datetime.utcnow().isoformat() + "Z"

def validate_file(path):
    validator = Draft7Validator(UNIFIED_SCHEMA)

    counts = Counter()
    label_dist = Counter()
    proof_dist = Counter()
    warnings = Counter()
    errors = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            counts["total"] += 1

            # schema validation
            errs = list(validator.iter_errors(obj))
            if errs:
                counts["schema_invalid"] += 1
                for e in errs:
                    errors[f"{list(e.path)} : {e.message}"] += 1
            else:
                counts["schema_valid"] += 1

            # distributions
            label_dist[obj["verdict"]["label"]] += 1
            proof_dist[obj["epistemic"]["proof_type"]] += 1

            # logic checks
            if obj.get("qa") is None and obj["verdict"]["label"] is not None:
                warnings["Label present but QA missing"] += 1

            if obj.get("evidence_items"):
                for e in obj["evidence_items"]:
                    if e.get("stance") is None:
                        warnings["Evidence missing stance"] += 1

    summary = {
        "file": path,
        "generated_utc": now(),
        "counts": dict(counts),
        "label_distribution": dict(label_dist),
        "epistemic_distribution": dict(proof_dist),
        "schema_errors_top": dict(errors.most_common(5)),
        "warnings_top": dict(warnings.most_common(5))
    }

    return summary


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    for fname in [
        "unified/averitec_train_unified.jsonl",
        "unified/averitec_dev_unified.jsonl"
    ]:
        print("\n===================================")
        s = validate_file(fname)
        print(json.dumps(s, indent=2))
