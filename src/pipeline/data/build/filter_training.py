#!/usr/bin/env python3
"""Filter unified JSONL to GNN training records.

Excludes:
- postulation_derivation evidence type (ADR-005)
- conflicting_evidence verdict (ADR-007)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.epistemic.enums import Verdict
from src.epistemic.formula import TRAINING_EVIDENCE_TYPES, is_training_record

EXCLUDED_VERDICTS: frozenset[str] = frozenset({Verdict.CONFLICTING_EVIDENCE})


def run(args) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    excluded = 0
    excluded_by_evidence_type: dict[str, int] = defaultdict(int)
    excluded_by_verdict: dict[str, int] = defaultdict(int)

    with (
        open(in_path, "r", encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            verdict = record.get("verdict", {}).get("label", "")
            if verdict in EXCLUDED_VERDICTS:
                excluded_by_verdict[verdict] += 1
                excluded += 1
                continue

            if is_training_record(record):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                for et in record.get("epistemic", {}).get("evidence_types_all", ["unknown"]):
                    excluded_by_evidence_type[et] += 1
                excluded += 1

    total = kept + excluded
    print("=" * 60)
    print("Filter for Training: SUCCESS")
    print(f"Input          : {in_path}")
    print(f"Output         : {out_path}")
    print(f"Total records  : {total:,}")
    print(f"Kept           : {kept:,}  ({kept / total * 100:.1f}%)")
    print(f"Excluded       : {excluded:,}  ({excluded / total * 100:.1f}%)")
    if args.verbose:
        if excluded_by_verdict:
            print("Excluded by verdict (ADR-007):")
            for v, n in sorted(excluded_by_verdict.items()):
                print(f"  {v}: {n:,}")
        if excluded_by_evidence_type:
            print("Excluded by evidence type (ADR-005):")
            for et, n in sorted(excluded_by_evidence_type.items()):
                print(f"  {et}: {n:,}")
    print(f"Training evidence types : {sorted(TRAINING_EVIDENCE_TYPES)}")
    print(f"Excluded verdict types  : {sorted(EXCLUDED_VERDICTS)}")
    print("=" * 60)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Filter unified JSONL to GNN training records (ADR-005).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--verbose", "-v", action="store_true")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
