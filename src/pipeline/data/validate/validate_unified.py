"""Validate unified v3.0 JSONL files — schema + semantic + dataset-specific checks."""

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft7Validator

from src.epistemic.enums import EvidenceType
from src.epistemic.schema import CLAIM_SCHEMA
from src.epistemic.validator import AdvancedClaimValidator
from src.adapters.ai2thor.validator import AI2ThorValidator
from src.adapters.averitec.validator import AveritecValidator
from src.utils.time import utc_now_iso

_DATASET_VALIDATORS: Dict[str, Any] = {
    "ai2thor": AI2ThorValidator(),
    "averitec": AveritecValidator(),
}


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


def _check_dataset_specific(obj: dict) -> List[str]:
    dataset = _safe_get(obj, ["provenance", "dataset"])
    validator = _DATASET_VALIDATORS.get(dataset)
    if validator is None:
        return []
    return validator.check(obj)


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
        "coverage": {
            "zero_evidence_records": 0,
            "all_evidence_text_null": 0,
            "evidence_count_sum": 0,
            "claim_triples_null": 0,
            "claim_triples_count_sum": 0,
            "evidence_triples_count_sum": 0,
            "refuted_absence_records": 0,
            "refuted_negation_records": 0,
        },
        "distributions": {
            "verdict_label": Counter(),
            "evidence_types_all": Counter(),
            "dataset": Counter(),
            "evidence_modality": Counter(),
            "evidence_stance": Counter(),
            "reasoning_structural": Counter(),
        },
        "schema_errors_top": Counter(),
        "logic_warnings_top": Counter(),
        "examples": {"schema_invalid": [], "warnings": []},
    }

    for line_no, obj in _iter_jsonl(path):
        summary["counts"]["total_records"] += 1
        evidence = obj.get("evidence") or []
        evidence_types_all = _safe_get(obj, ["epistemic", "evidence_types_all"], [])
        is_absence = EvidenceType.NON_APPREHENSION.value in evidence_types_all

        summary["distributions"]["verdict_label"][
            _safe_get(obj, ["verdict", "label"])
        ] += 1
        for et in evidence_types_all:
            summary["distributions"]["evidence_types_all"][et] += 1
        summary["distributions"]["dataset"][
            _safe_get(obj, ["provenance", "dataset"])
        ] += 1
        summary["distributions"]["reasoning_structural"][
            _safe_get(obj, ["reasoning", "structural"])
        ] += 1
        for e in evidence:
            if isinstance(e, dict):
                summary["distributions"]["evidence_modality"][e.get("modality")] += 1
                summary["distributions"]["evidence_stance"][e.get("stance")] += 1

        summary["coverage"]["evidence_count_sum"] += len(evidence)
        if not evidence:
            summary["coverage"]["zero_evidence_records"] += 1
        elif all(e.get("text") is None for e in evidence if isinstance(e, dict)):
            summary["coverage"]["all_evidence_text_null"] += 1

        ct = obj.get("claim_triples")
        structural = _safe_get(obj, ["reasoning", "structural"])
        verdict_label = _safe_get(obj, ["verdict", "label"])
        if ct is None and not is_absence and structural != "absence":
            summary["coverage"]["claim_triples_null"] += 1

        summary["coverage"]["claim_triples_count_sum"] += len(ct or [])
        for e in evidence:
            if isinstance(e, dict):
                summary["coverage"]["evidence_triples_count_sum"] += len(
                    e.get("triples") or []
                )

        if structural == "absence" and verdict_label == "refuted":
            summary["coverage"]["refuted_absence_records"] += 1
        if structural == "negation" and verdict_label == "refuted":
            summary["coverage"]["refuted_negation_records"] += 1

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

        warnings: List[str] = []
        cv_result = claim_validator.validate_claim_advanced(obj)
        for issue in cv_result.issues:
            if issue.category != "schema":
                warnings.append(f"[{issue.category}:{issue.severity}] {issue.message}")
        warnings += _check_dataset_specific(obj)

        if warnings:
            summary["counts"]["logic_warnings_records"] += 1
            for w in warnings:
                summary["logic_warnings_top"][w] += 1
            if len(summary["examples"]["warnings"]) < max_examples:
                summary["examples"]["warnings"].append(
                    {"line": line_no, "id": obj.get("id"), "warnings": warnings[:5]}
                )

    for k in list(summary["distributions"]):
        summary["distributions"][k] = dict(summary["distributions"][k])
    summary["schema_errors_top"] = dict(summary["schema_errors_top"].most_common(20))
    summary["logic_warnings_top"] = dict(summary["logic_warnings_top"].most_common(20))

    total = summary["counts"]["total_records"]
    dataset_warnings = []
    if total > 0:
        for verdict, count in summary["distributions"]["verdict_label"].items():
            pct = count / total * 100
            if pct > 70:
                dataset_warnings.append(
                    f"Label imbalance: '{verdict}' is {pct:.1f}% of records (>70%)"
                )
    summary["dataset_warnings"] = dataset_warnings
    summary["gnn_readiness"] = _compute_gnn_readiness(summary)
    return summary


def _compute_gnn_readiness(summary: dict) -> dict:
    dists = summary.get("distributions", {})
    counts = summary.get("counts", {})
    coverage = summary.get("coverage", {})
    total = counts.get("total_records", 0)
    evidence_type_dist = dists.get("evidence_types_all", {})
    verdict_dist = dists.get("verdict_label", {})
    stance_dist = dists.get("evidence_stance", {})
    absence_count = evidence_type_dist.get(EvidenceType.NON_APPREHENSION.value, 0)
    ev_sum = coverage.get("evidence_count_sum", 0)
    ct_sum = coverage.get("claim_triples_count_sum", 0)
    ev_tri_sum = coverage.get("evidence_triples_count_sum", 0)
    return {
        "total_records": total,
        "verdict_distribution": verdict_dist,
        "evidence_type_distribution": evidence_type_dist,
        "stance_distribution": stance_dist,
        "absence_claims": absence_count,
        "absence_pct": round(absence_count / total * 100, 2) if total else 0,
        "avg_evidence_per_record": round(ev_sum / total, 2) if total else 0,
        "avg_claim_triples_per_record": round(ct_sum / total, 2) if total else 0,
        "avg_evidence_triples_per_record": round(ev_tri_sum / total, 2) if total else 0,
        "zero_evidence_records": coverage.get("zero_evidence_records", 0),
        "all_text_null_records": coverage.get("all_evidence_text_null", 0),
        "claim_triples_null_non_absence": coverage.get("claim_triples_null", 0),
        "refuted_absence_claims": coverage.get("refuted_absence_records", 0),
        "refuted_negation_claims": coverage.get("refuted_negation_records", 0),
        "label_balance_ok": not any(
            (v / total * 100) > 70 for v in verdict_dist.values() if total
        ),
    }


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "0.0%"


def write_validation_report_md(
    summaries: List[Dict[str, Any]], out_path: Path, generated_utc: str
) -> None:
    lines: List[str] = []
    w = lines.append

    w("# Validation Report")
    w(f"\nGenerated: {generated_utc}\n")

    w("## Summary\n")
    w("| File | Total | Schema Valid | Schema Invalid | Records w/ Warnings |")
    w("|------|------:|-------------:|---------------:|--------------------:|")
    for s in summaries:
        c = s["counts"]
        total = c["total_records"]
        fname = Path(s["file"]).name
        w(
            f"| `{fname}` "
            f"| {total:,} "
            f"| {c['schema_valid']:,} ({_pct(c['schema_valid'], total)}) "
            f"| {c['schema_invalid']:,} ({_pct(c['schema_invalid'], total)}) "
            f"| {c['logic_warnings_records']:,} ({_pct(c['logic_warnings_records'], total)}) |"
        )

    for s in summaries:
        fname = Path(s["file"]).name
        c = s["counts"]
        total = c["total_records"]
        w(f"\n---\n\n## File: `{fname}`\n")

        w("### Verdict Distribution\n")
        w("| Verdict | Count | % |")
        w("|---------|------:|--:|")
        for k, v in sorted(
            s["distributions"]["verdict_label"].items(), key=lambda x: str(x[0])
        ):
            w(f"| {k} | {v:,} | {_pct(v, total)} |")

        w("\n### Evidence Type Distribution\n")
        w("| Evidence Type | Count | % |")
        w("|--------------|------:|--:|")
        for k, v in sorted(
            s["distributions"]["evidence_types_all"].items(), key=lambda x: str(x[0])
        ):
            w(f"| {k} | {v:,} | {_pct(v, total)} |")

        w("\n### Evidence Stance Distribution\n")
        w("| Stance | Count | % of evidence items |")
        w("|--------|------:|--------------------:|")
        ev_total = sum(s["distributions"]["evidence_stance"].values())
        for k, v in sorted(
            s["distributions"]["evidence_stance"].items(), key=lambda x: str(x[0])
        ):
            w(f"| {k} | {v:,} | {_pct(v, ev_total)} |")

        w("\n### Evidence Modality Distribution\n")
        w("| Modality | Count | % of evidence items |")
        w("|----------|------:|--------------------:|")
        for k, v in sorted(
            s["distributions"]["evidence_modality"].items(), key=lambda x: str(x[0])
        ):
            w(f"| {k} | {v:,} | {_pct(v, ev_total)} |")

        w("\n### Claim Structural Type\n")
        w("| Structural | Count | % |")
        w("|------------|------:|--:|")
        for k, v in sorted(
            s["distributions"]["reasoning_structural"].items(), key=lambda x: str(x[0])
        ):
            w(f"| {k} | {v:,} | {_pct(v, total)} |")

        w("\n### Schema Errors\n")
        if not s["schema_errors_top"]:
            w("_No schema errors._\n")
        else:
            w("| Count | Error |")
            w("|------:|-------|")
            for msg, cnt in s["schema_errors_top"].items():
                w(f"| {cnt} | {msg} |")

        w("\n### Semantic Rule Violations\n")
        if not s["logic_warnings_top"]:
            w("_No logic warnings._\n")
        else:
            w("| Count | Rule |")
            w("|------:|------|")
            for msg, cnt in s["logic_warnings_top"].items():
                w(f"| {cnt} | {msg} |")

        if s.get("dataset_warnings"):
            w("\n### Dataset-Level Warnings\n")
            for dw in s["dataset_warnings"]:
                w(f"- **{dw}**")
            w("")

        cov = s.get("coverage", {})
        gnn = s.get("gnn_readiness", {})
        w("\n### Coverage Metrics\n")
        w("| Metric | Value |")
        w("|--------|------:|")
        w(f"| Total records | {total:,} |")
        w(
            f"| Avg evidence items / record | {gnn.get('avg_evidence_per_record', 0):.2f} |"
        )
        w(
            f"| Records with zero evidence | {cov.get('zero_evidence_records', 0):,} ({_pct(cov.get('zero_evidence_records', 0), total)}) |"
        )
        w(
            f"| Records with all evidence text=null | {cov.get('all_evidence_text_null', 0):,} ({_pct(cov.get('all_evidence_text_null', 0), total)}) |"
        )
        w(
            f"| Non-absence records missing claim_triples | {cov.get('claim_triples_null', 0):,} ({_pct(cov.get('claim_triples_null', 0), total)}) |"
        )

        w("\n### GNN Readiness\n")
        w("| Signal | Value |")
        w("|--------|------:|")
        w(
            f"| Absence claims (non_apprehension) | {gnn.get('absence_claims', 0):,} ({gnn.get('absence_pct', 0):.1f}%) |"
        )
        w(
            f"| Label balance OK (no class > 70%) | {'Yes' if gnn.get('label_balance_ok') else 'No'} |"
        )
        w(
            f"| Avg evidence items / record | {gnn.get('avg_evidence_per_record', 0):.2f} |"
        )
        w(
            f"| Avg claim triples / record | {gnn.get('avg_claim_triples_per_record', 0):.2f} |"
        )
        w(
            f"| Avg evidence triples / record | {gnn.get('avg_evidence_triples_per_record', 0):.2f} |"
        )
        w(f"| Refuted absence claims | {gnn.get('refuted_absence_claims', 0):,} |")
        w(f"| Refuted negation claims | {gnn.get('refuted_negation_claims', 0):,} |")

        w("\n#### Stance edge-type distribution (GNN edge labels)\n")
        w("| Stance | Count |")
        w("|--------|------:|")
        for k, v in sorted(
            gnn.get("stance_distribution", {}).items(), key=lambda x: str(x[0])
        ):
            w(f"| {k} | {v:,} |")

        if s["examples"]["schema_invalid"]:
            w("\n### Example Schema Errors\n")
            for ex in s["examples"]["schema_invalid"]:
                w(f"- **line {ex['line']}** `{ex['id']}`")
                for e in ex["errors"]:
                    w(f"  - `{e['path']}`: {e['message']}")
            w("")

        if s["examples"]["warnings"]:
            w("\n### Example Logic Warnings\n")
            for ex in s["examples"]["warnings"]:
                w(f"- **line {ex['line']}** `{ex['id']}`")
                for warn in ex["warnings"]:
                    w(f"  - {warn}")
            w("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> int:
    """Called by the validate dispatcher; args must have .files, .out, .schema."""
    schema = _load_json(args.schema) if args.schema else CLAIM_SCHEMA
    schema_validator = Draft7Validator(schema)
    claim_validator = AdvancedClaimValidator()

    all_summaries = []
    for fpath in args.files:
        s = summarize_file(fpath, schema_validator, claim_validator)
        all_summaries.append(s)

    generated_utc = utc_now_iso()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_utc": generated_utc, "summaries": all_summaries},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote JSON summary to: {out_path}")

    md_path = out_path.with_suffix(".md")
    write_validation_report_md(all_summaries, md_path, generated_utc)
    print(f"Wrote Markdown report to: {md_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate unified v3.0 JSONL files.")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--out", default="out/report/validation.json")
    ap.add_argument("--max_examples", type=int, default=3)
    args = ap.parse_args()

    schema = _load_json(args.schema) if args.schema else CLAIM_SCHEMA
    schema_validator = Draft7Validator(schema)
    claim_validator = AdvancedClaimValidator()

    all_summaries = []
    for fpath in args.files:
        s = summarize_file(fpath, schema_validator, claim_validator, args.max_examples)
        all_summaries.append(s)

    generated_utc = utc_now_iso()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_utc": generated_utc, "summaries": all_summaries},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote JSON summary to: {out_path}")

    md_path = out_path.with_suffix(".md")
    write_validation_report_md(all_summaries, md_path, generated_utc)
    print(f"Wrote Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
