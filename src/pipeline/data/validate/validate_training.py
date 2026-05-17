"""Validate training JSONL against ADR-012 Pramana distribution targets."""

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
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        targets = cfg.get("training", {}).get("evidence_type_targets", {})
        if targets:
            return {k: int(v) for k, v in targets.items()}
    except Exception:
        pass
    return dict(_ADR012_TARGETS)


def run(args) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    targets = _load_targets(args.config)
    et_counts: dict[str, int] = defaultdict(int)
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
            for et in record.get("epistemic", {}).get("evidence_types_all", []):
                et_counts[et] += 1
            verdict_counts[record.get("verdict", {}).get("label", "unknown")] += 1
            source_counts[record.get("provenance", {}).get("dataset", "unknown")] += 1

    postulation_count = et_counts.get("postulation_derivation", 0)
    et_vs_targets = {}
    for et, target in targets.items():
        actual = et_counts.get(et, 0)
        pct = actual / total * 100 if total else 0.0
        et_vs_targets[et] = {
            "actual": actual,
            "target": target,
            "pct": round(pct, 1),
            "delta": actual - target,
        }

    warnings = []
    if postulation_count > 0:
        warnings.append(f"postulation_derivation records found: {postulation_count}")

    result = {
        "input": str(in_path),
        "total_records": total,
        "postulation_derivation_count": postulation_count,
        "evidence_type_distribution": dict(et_counts),
        "evidence_type_vs_targets": et_vs_targets,
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
    print("Validate Training: " + ("PASS" if result["pass"] else "FAIL"))
    print(f"Total records: {total:,}  |  postulation_derivation: {postulation_count}")
    print(f"Output: {out_path}")
    print("=" * 60)
    return 0 if result["pass"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate training JSONL vs ADR-012 targets."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", required=True)
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
