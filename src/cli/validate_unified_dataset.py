"""
Validate one or more unified v2.0 JSONL files.

Usage
-----
python -m src.cli.validate_unified_dataset \\
    --files data/processed/ai2thor_unified.jsonl data/processed/averitec_train.jsonl \\
    --out   data/summary/validation.json

--schema is optional; defaults to the built-in CLAIM_SCHEMA (src/core/claims/claim_schema.py).
Pass an external JSON Schema file to override.
"""

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft7Validator

from src.utils.time import utc_now_iso
from src.core.claims.claim_schema import CLAIM_SCHEMA
from src.core.claims.claim_validator import AdvancedClaimValidator
from src.adapters.ai2thor.validator import AI2ThorValidator
from src.adapters.averitec.validator import AveritecValidator

# Registry: dataset name -> DatasetValidator instance
_DATASET_VALIDATORS: Dict[str, Any] = {
    "ai2thor": AI2ThorValidator(),
    "averitec": AveritecValidator(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ---------------------------------------------------------------------------
# Logic checks (v2.0 field names)
# ---------------------------------------------------------------------------


def _check_stance_consistency(obj: dict) -> List[str]:
    msgs = []
    vlabel = _safe_get(obj, ["verdict", "label"])
    stances = [
        e.get("stance")
        for e in (obj.get("evidence") or [])
        if isinstance(e, dict) and e.get("stance")
    ]
    if not stances or vlabel is None:
        return msgs
    if vlabel == "supported" and all(s == "refutes" for s in stances):
        msgs.append("All evidence stances are 'refutes' but verdict is 'supported'.")
    if vlabel == "refuted" and all(s == "supports" for s in stances):
        msgs.append("All evidence stances are 'supports' but verdict is 'refuted'.")
    return msgs


def _check_dataset_specific(obj: dict) -> List[str]:
    dataset = _safe_get(obj, ["provenance", "dataset"])
    validator = _DATASET_VALIDATORS.get(dataset)
    if validator is None:
        return []
    return validator.check(obj)


# ---------------------------------------------------------------------------
# Per-file summary
# ---------------------------------------------------------------------------


def summarize_file(
    path: str,
    schema_validator: Draft7Validator,
    claim_validator: AdvancedClaimValidator,
    max_examples: int = 3,
) -> Dict[str, Any]:
    summary = {
        "file": path,
        "generated_utc": utc_now_iso(),
        "counts": {
            "total_records": 0,
            "schema_valid": 0,
            "schema_invalid": 0,
            "logic_warnings_records": 0,
        },
        "distributions": {
            "verdict_label": Counter(),
            "pramana_primary": Counter(),
            "pramana_all": Counter(),
            "dataset": Counter(),
            "evidence_modality": Counter(),
            "reasoning_structural": Counter(),
        },
        "schema_errors_top": Counter(),
        "logic_warnings_top": Counter(),
        "examples": {
            "schema_invalid": [],
            "warnings": [],
        },
    }

    for line_no, obj in _iter_jsonl(path):
        summary["counts"]["total_records"] += 1

        # Distributions
        summary["distributions"]["verdict_label"][
            _safe_get(obj, ["verdict", "label"])
        ] += 1
        summary["distributions"]["pramana_primary"][
            _safe_get(obj, ["epistemic", "pramana_primary"])
        ] += 1
        summary["distributions"]["dataset"][
            _safe_get(obj, ["provenance", "dataset"])
        ] += 1
        summary["distributions"]["reasoning_structural"][
            _safe_get(obj, ["reasoning", "structural"])
        ] += 1

        for pt in _safe_get(obj, ["epistemic", "pramana_all"], []):
            summary["distributions"]["pramana_all"][pt] += 1

        for e in obj.get("evidence") or []:
            if isinstance(e, dict):
                summary["distributions"]["evidence_modality"][e.get("modality")] += 1

        # JSON Schema validation
        schema_errors = sorted(
            schema_validator.iter_errors(obj), key=lambda e: list(e.path)
        )
        if schema_errors:
            summary["counts"]["schema_invalid"] += 1
            for e in schema_errors:
                key = f"{'/'.join(str(x) for x in e.path)} | {e.message}"
                summary["schema_errors_top"][key] += 1
            if len(summary["examples"]["schema_invalid"]) < max_examples:
                summary["examples"]["schema_invalid"].append(
                    {
                        "line": line_no,
                        "id": obj.get("id"),
                        "errors": [
                            {
                                "path": "/".join(str(x) for x in e.path),
                                "message": e.message,
                            }
                            for e in schema_errors[:5]
                        ],
                    }
                )
        else:
            summary["counts"]["schema_valid"] += 1

        # Logic warnings (v2.0 field names + dataset-specific)
        warnings: List[str] = []
        warnings += _check_stance_consistency(obj)
        warnings += _check_dataset_specific(obj)

        if warnings:
            summary["counts"]["logic_warnings_records"] += 1
            for w in warnings:
                summary["logic_warnings_top"][w] += 1
            if len(summary["examples"]["warnings"]) < max_examples:
                summary["examples"]["warnings"].append(
                    {
                        "line": line_no,
                        "id": obj.get("id"),
                        "warnings": warnings[:5],
                    }
                )

    # Serialize Counters
    for k in list(summary["distributions"]):
        summary["distributions"][k] = dict(summary["distributions"][k])
    summary["schema_errors_top"] = dict(summary["schema_errors_top"].most_common(20))
    summary["logic_warnings_top"] = dict(summary["logic_warnings_top"].most_common(20))

    return summary


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def pretty_print_summary(s: Dict[str, Any]) -> None:
    c = s["counts"]
    print("\n" + "=" * 50)
    print(f"FILE: {s['file']}")
    print("=" * 50)
    print(f"Total records:        {c['total_records']:,}")
    print(f"Schema valid:         {c['schema_valid']:,}")
    print(f"Schema invalid:       {c['schema_invalid']:,}")
    print(f"Records w/ warnings:  {c['logic_warnings_records']:,}")

    print("\n-- Verdict label --")
    for k, v in sorted(
        s["distributions"]["verdict_label"].items(), key=lambda x: str(x[0])
    ):
        print(f"  {k}: {v}")

    print("\n-- Pramana primary --")
    for k, v in sorted(
        s["distributions"]["pramana_primary"].items(), key=lambda x: str(x[0])
    ):
        print(f"  {k}: {v}")

    print("\n-- Dataset --")
    for k, v in sorted(s["distributions"]["dataset"].items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    print("\n-- Evidence modality --")
    for k, v in sorted(
        s["distributions"]["evidence_modality"].items(), key=lambda x: str(x[0])
    ):
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
            print(f"  line {ex['line']}  id={ex['id']}")
            for e in ex["errors"]:
                print(f"    - {e['path']}: {e['message']}")

    if s["examples"]["warnings"]:
        print("\n-- Examples: warnings --")
        for ex in s["examples"]["warnings"]:
            print(f"  line {ex['line']}  id={ex['id']}")
            for w in ex["warnings"]:
                print(f"    - {w}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate unified v2.0 JSONL files against schema + logic checks."
    )
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more unified v2.0 JSONL files to validate.",
    )
    ap.add_argument(
        "--schema",
        default=None,
        help="Path to an external JSON Schema file. Defaults to built-in CLAIM_SCHEMA.",
    )
    ap.add_argument(
        "--out",
        default="data/summary/validation.json",
        help="Where to write the summary JSON.",
    )
    ap.add_argument(
        "--max_examples",
        type=int,
        default=3,
        help="Max failing examples to include in the summary.",
    )
    args = ap.parse_args()

    schema = _load_json(args.schema) if args.schema else CLAIM_SCHEMA
    schema_validator = Draft7Validator(schema)
    claim_validator = AdvancedClaimValidator()

    all_summaries = []
    for fpath in args.files:
        s = summarize_file(fpath, schema_validator, claim_validator, args.max_examples)
        all_summaries.append(s)
        pretty_print_summary(s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_utc": utc_now_iso(), "summaries": all_summaries},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nWrote summary to: {args.out}")


if __name__ == "__main__":
    main()
