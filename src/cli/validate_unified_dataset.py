import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from jsonschema import Draft7Validator

from src.utils.time import utc_now_iso

# ----------------------------
# Helpers
# ----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# ----------------------------
# Validation checks
# ----------------------------
def validate_schema(validator: Draft7Validator, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of schema error dicts (empty if none)."""
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    out = []
    for e in errors:
        out.append({
            "path": "/".join([str(x) for x in e.path]),
            "message": e.message
        })
    return out

def check_evidence_links(obj: Dict[str, Any]) -> List[str]:
    """
    Ensure every qa.answers[].evidence_ids[] exists in evidence_items[].evidence_id.
    """
    msgs = []
    qa = obj.get("qa") or []
    eitems = obj.get("evidence_items") or []
    eids = {e.get("evidence_id") for e in eitems if isinstance(e, dict)}

    for qi, q in enumerate(qa, start=1):
        answers = (q or {}).get("answers") or []
        for ai, a in enumerate(answers, start=1):
            for eid in (a or {}).get("evidence_ids") or []:
                if eid not in eids:
                    msgs.append(f"Missing evidence_id link: qa[{qi}].answers[{ai}] -> {eid}")
    return msgs

def check_stance_consistency(obj: Dict[str, Any]) -> List[str]:
    """
    Basic heuristic: if verdict is supported/refuted, evidence stances shouldn't be all opposite.
    (Not a hard error; report as warning.)
    """
    msgs = []
    vlabel = safe_get(obj, ["verdict", "label"])
    eitems = obj.get("evidence_items") or []
    stances = [e.get("stance") for e in eitems if isinstance(e, dict) and e.get("stance")]

    if not stances or vlabel is None:
        return msgs

    if vlabel == "supported":
        if all(s == "refutes" for s in stances):
            msgs.append("All evidence stances are 'refutes' but verdict is 'supported'.")
    if vlabel == "refuted":
        if all(s == "supports" for s in stances):
            msgs.append("All evidence stances are 'supports' but verdict is 'refuted'.")
    return msgs

def check_ai2thor_expected_fields(obj: Dict[str, Any]) -> List[str]:
    """
    If record looks like AI2-THOR (simulation evidence), ensure claim_triples present etc.
    """
    msgs = []
    eitems = obj.get("evidence_items") or []
    has_sim = any((e.get("source_type") == "simulation") for e in eitems if isinstance(e, dict))
    if not has_sim:
        return msgs

    # For AI2-THOR, claim_triples should generally exist
    if obj.get("claim_triples") is None:
        msgs.append("Looks like AI2-THOR (simulation evidence) but claim_triples is null.")
    # reasoning.structural should generally exist
    reasoning = obj.get("reasoning")
    if reasoning is None or not isinstance(reasoning, dict) or "structural" not in reasoning:
        msgs.append("Looks like AI2-THOR (simulation evidence) but reasoning.structural missing.")
    return msgs

def check_averitec_blind_test_shape(obj: Dict[str, Any]) -> List[str]:
    """
    If record looks like blind test (no qa/evidence), verdict.label should often be null.
    Warn if it's non-null but no evidence/qa.
    """
    msgs = []
    qa = obj.get("qa")
    eitems = obj.get("evidence_items")
    vlabel = safe_get(obj, ["verdict", "label"])
    if (qa is None) and (eitems is None):
        if vlabel is not None:
            msgs.append("qa and evidence_items are null but verdict.label is not null (check if this is truly blind test).")
    return msgs

def check_date_formats(obj: Dict[str, Any]) -> List[str]:
    """
    Soft check: claim_meta.claim_date should be YYYY-MM-DD if present.
    """
    msgs = []
    cd = safe_get(obj, ["claim_meta", "claim_date"])
    if cd is None:
        return msgs
    if isinstance(cd, str):
        import re
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cd):
            msgs.append(f"claim_meta.claim_date not ISO YYYY-MM-DD: {cd}")
    return msgs

# ----------------------------
# Summary builder
# ----------------------------
def summarize_file(path: str, schema_validator: Draft7Validator, max_examples: int = 3) -> Dict[str, Any]:
    summary = {
        "file": path,
        "generated_utc": utc_now_iso(),
        "counts": {
            "total_records": 0,
            "schema_valid": 0,
            "schema_invalid": 0,
            "warnings_records": 0
        },
        "distributions": {
            "verdict_label": Counter(),
            "epistemic_proof_type": Counter(),
            "epistemic_proof_types_multi": Counter(),
            "context_type": Counter(),
            "source_type": Counter(),
            "answer_type": Counter()
        },
        "schema_errors_top": Counter(),  # (path|message) frequency
        "logic_warnings_top": Counter(),
        "examples": {
            "schema_invalid": [],  # list of {line, id, errors[]}
            "warnings": []         # list of {line, id, warnings[]}
        }
    }

    for line_no, obj in iter_jsonl(path):
        summary["counts"]["total_records"] += 1

        # distributions — v2.0 field names
        summary["distributions"]["verdict_label"][safe_get(obj, ["verdict", "label"])] += 1
        summary["distributions"]["epistemic_proof_type"][safe_get(obj, ["epistemic", "pramana_primary"])] += 1
        summary["distributions"]["context_type"][safe_get(obj, ["provenance", "dataset"])] += 1

        # all pramana labels (multi-label list)
        for pt in safe_get(obj, ["epistemic", "pramana_all"], []):
            summary["distributions"]["epistemic_proof_types_multi"][pt] += 1

        # source types
        for e in (obj.get("evidence") or []):
            if isinstance(e, dict):
                summary["distributions"]["source_type"][e.get("source_type")] += 1

        # answer types
        for q in (obj.get("qa") or []):
            for a in ((q or {}).get("answers") or []):
                summary["distributions"]["answer_type"][a.get("answer_type")] += 1

        # schema validation
        errs = validate_schema(schema_validator, obj)
        if errs:
            summary["counts"]["schema_invalid"] += 1
            for e in errs:
                key = f"{e['path']} | {e['message']}"
                summary["schema_errors_top"][key] += 1
            if len(summary["examples"]["schema_invalid"]) < max_examples:
                summary["examples"]["schema_invalid"].append({
                    "line": line_no,
                    "id": obj.get("id"),
                    "errors": errs[:5]
                })
        else:
            summary["counts"]["schema_valid"] += 1

        # logic checks (warnings)
        warnings = []
        warnings += check_evidence_links(obj)
        warnings += check_stance_consistency(obj)
        warnings += check_ai2thor_expected_fields(obj)
        warnings += check_averitec_blind_test_shape(obj)
        warnings += check_date_formats(obj)

        if warnings:
            summary["counts"]["warnings_records"] += 1
            for w in warnings:
                summary["logic_warnings_top"][w] += 1
            if len(summary["examples"]["warnings"]) < max_examples:
                summary["examples"]["warnings"].append({
                    "line": line_no,
                    "id": obj.get("id"),
                    "warnings": warnings[:5]
                })

    # convert Counters to dicts for JSON
    for k in list(summary["distributions"].keys()):
        summary["distributions"][k] = dict(summary["distributions"][k])

    summary["schema_errors_top"] = dict(summary["schema_errors_top"].most_common(20))
    summary["logic_warnings_top"] = dict(summary["logic_warnings_top"].most_common(20))

    return summary

def pretty_print_summary(s: Dict[str, Any]):
    c = s["counts"]
    print("\n==============================")
    print(f"FILE: {s['file']}")
    print("==============================")
    print(f"Total records:     {c['total_records']}")
    print(f"Schema valid:      {c['schema_valid']}")
    print(f"Schema invalid:    {c['schema_invalid']}")
    print(f"Records w/warn:    {c['warnings_records']}")

    print("\n-- Verdict label distribution --")
    for k, v in sorted(s["distributions"]["verdict_label"].items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    print("\n-- Epistemic proof_type distribution --")
    for k, v in sorted(s["distributions"]["epistemic_proof_type"].items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    print("\n-- Top schema errors --")
    if not s["schema_errors_top"]:
        print("  (none)")
    else:
        for k, v in s["schema_errors_top"].items():
            print(f"  ({v}) {k}")

    print("\n-- Top logic warnings --")
    if not s["logic_warnings_top"]:
        print("  (none)")
    else:
        for k, v in s["logic_warnings_top"].items():
            print(f"  ({v}) {k}")

    if s["examples"]["schema_invalid"]:
        print("\n-- Examples: schema invalid --")
        for ex in s["examples"]["schema_invalid"]:
            print(f"  line {ex['line']} id={ex['id']}")
            for e in ex["errors"]:
                print(f"    - {e['path']}: {e['message']}")

    if s["examples"]["warnings"]:
        print("\n-- Examples: warnings --")
        for ex in s["examples"]["warnings"]:
            print(f"  line {ex['line']} id={ex['id']}")
            for w in ex["warnings"]:
                print(f"    - {w}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="Path to unified_schema.json")
    ap.add_argument("--files", nargs="+", required=True, help="One or more JSONL files to validate")
    ap.add_argument("--out", default="validation_summary.json", help="Where to write summary JSON")
    ap.add_argument("--max_examples", type=int, default=3, help="Number of failing examples to include")
    args = ap.parse_args()

    schema = load_json(args.schema)
    validator = Draft7Validator(schema)

    all_summaries = []
    for fpath in args.files:
        s = summarize_file(fpath, validator, max_examples=args.max_examples)
        all_summaries.append(s)
        pretty_print_summary(s)

    out_obj = {
        "generated_utc": utc_now_iso(),
        "summaries": all_summaries
    }
    
    # Ensure the output directory exists
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote summary JSON to: {args.out}")

if __name__ == "__main__":
    main()
