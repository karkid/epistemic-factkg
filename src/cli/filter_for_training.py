#!/usr/bin/env python3
"""Filter unified JSONL to GNN training records.

Excludes:
- postulation_derivation Pramana (ADR-011)
- conflicting_evidence verdict (ADR-015)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.core.claims.labels import TRAINING_PRAMANA, Verdict, is_training_record

EXCLUDED_VERDICTS: frozenset[str] = frozenset({Verdict.CONFLICTING_EVIDENCE})


def main():
    ap = argparse.ArgumentParser(
        description="Filter unified JSONL to GNN training records (ADR-011).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input", required=True, help="Path to unified epistemic_factkg.jsonl"
    )
    ap.add_argument("--output", required=True, help="Path to write training JSONL")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    excluded = 0
    excluded_by_pramana: dict[str, int] = defaultdict(int)
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
                pramana = record.get("epistemic", {}).get("pramana_primary", "unknown")
                excluded_by_pramana[pramana] += 1
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
            print("Excluded by verdict (ADR-015):")
            for v, n in sorted(excluded_by_verdict.items()):
                print(f"  {v}: {n:,}")
        if excluded_by_pramana:
            print("Excluded by pramana (ADR-011):")
            for p, n in sorted(excluded_by_pramana.items()):
                print(f"  {p}: {n:,}")
    print(f"Training pramana types : {sorted(TRAINING_PRAMANA)}")
    print(f"Excluded verdict types : {sorted(EXCLUDED_VERDICTS)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
