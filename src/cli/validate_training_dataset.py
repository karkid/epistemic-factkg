#!/usr/bin/env python3
"""Validate training JSONL against ADR-012 distribution targets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


_ADR012_TARGETS: dict[str, int] = {
    "perception": 600,
    "non_apprehension": 1000,
    "inference": 1100,
    "testimony": 3000,
    "comparison_analogy": 500,
}


def _load_targets(config_path: str) -> dict[str, int]:
    """Load ADR-012 targets from config if present, else fall back to hardcoded defaults."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        targets = cfg.get("training", {}).get("pramana_targets", {})
        if targets:
            return {k: int(v) for k, v in targets.items()}
    except Exception:
        pass
    return dict(_ADR012_TARGETS)


def main():
    ap = argparse.ArgumentParser(
        description="Validate training JSONL against ADR-012 targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path to training JSONL")
    ap.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file (optional training.pramana_targets override)",
    )
    ap.add_argument(
        "--out", required=True, help="Path to write training_validation.json"
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    targets = _load_targets(args.config)

    pramana_counts: dict[str, int] = defaultdict(int)
    verdict_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    total = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            pramana = record.get("epistemic", {}).get("pramana_primary", "unknown")
            verdict = record.get("verdict", {}).get("label", "unknown")
            dataset = record.get("provenance", {}).get("dataset", "unknown")
            pramana_counts[pramana] += 1
            verdict_counts[verdict] += 1
            source_counts[dataset] += 1

    # Check for zero postulation_derivation
    postulation_count = pramana_counts.get("postulation_derivation", 0)

    # Comparison vs targets
    pramana_vs_targets = {}
    for pramana, target in targets.items():
        actual = pramana_counts.get(pramana, 0)
        pct = actual / total * 100 if total else 0.0
        pramana_vs_targets[pramana] = {
            "actual": actual,
            "target": target,
            "pct": round(pct, 1),
            "delta": actual - target,
        }

    warnings = []
    if postulation_count > 0:
        warnings.append(
            f"postulation_derivation records found in training set: {postulation_count}"
        )

    result = {
        "input": str(in_path),
        "total_records": total,
        "postulation_derivation_count": postulation_count,
        "pramana_distribution": dict(pramana_counts),
        "pramana_vs_targets": pramana_vs_targets,
        "verdict_distribution": dict(verdict_counts),
        "source_distribution": dict(source_counts),
        "warnings": warnings,
        "pass": postulation_count == 0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print("Validate Training Dataset: " + ("PASS" if result["pass"] else "FAIL"))
    print(f"Input          : {in_path}")
    print(f"Output         : {out_path}")
    print(f"Total records  : {total:,}")
    print(f"postulation_derivation: {postulation_count}")
    print()
    print("Pramana distribution vs ADR-012 targets:")
    for pramana, info in sorted(pramana_vs_targets.items()):
        delta_str = f"{info['delta']:+d}" if info["delta"] else "  0"
        print(
            f"  {pramana:<22} actual={info['actual']:>5,}  target={info['target']:>5,}  delta={delta_str}"
        )
    print()
    print("Source distribution:")
    for src, count in sorted(source_counts.items()):
        pct = count / total * 100 if total else 0.0
        print(f"  {src:<12} {count:>6,}  ({pct:.1f}%)")
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  ! {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
