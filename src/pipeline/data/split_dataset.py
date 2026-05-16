#!/usr/bin/env python3
"""Train/val/test split for the graph dataset (ADR-009).

AI2THOR: split by context_id (floorplan-based, not random).
AVeriTeC: random stratified split by verdict label.
Outputs deterministic index files to out/splits/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                r["_line_idx"] = i
                records.append(r)
            except json.JSONDecodeError:
                continue
    return records


def _floorplan_split(
    records: list[dict], train_frac: float, val_frac: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Split AI2THOR records by context_id (floorplan) — ADR-009."""
    by_context: dict[str, list[int]] = defaultdict(list)
    for r in records:
        ctx = r.get("provenance", {}).get("context_id", r["_line_idx"])
        by_context[str(ctx)].append(r["_line_idx"])

    contexts = sorted(by_context.keys())
    rng = random.Random(seed)
    rng.shuffle(contexts)

    n = len(contexts)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_ctx = contexts[:n_train]
    val_ctx = contexts[n_train : n_train + n_val]
    test_ctx = contexts[n_train + n_val :]

    def indices(ctx_list: list[str]) -> list[int]:
        out = []
        for c in ctx_list:
            out.extend(by_context[c])
        return sorted(out)

    return indices(train_ctx), indices(val_ctx), indices(test_ctx)


def _stratified_split(
    records: list[dict], train_frac: float, val_frac: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Stratified random split by verdict label — for AVeriTeC records."""
    by_verdict: dict[str, list[int]] = defaultdict(list)
    for r in records:
        verdict = r.get("verdict", {}).get("label", "unknown")
        by_verdict[verdict].append(r["_line_idx"])

    train_idx, val_idx, test_idx = [], [], []
    rng = random.Random(seed)

    for verdict, idxs in by_verdict.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate deterministic train/val/test index files (ADR-009).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path to filtered training JSONL")
    ap.add_argument(
        "--output-dir", default="out/data/splits", help="Directory for index JSON files"
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(in_path)
    ai2thor = [
        r for r in records if r.get("provenance", {}).get("dataset") == "ai2thor"
    ]
    averitec = [
        r for r in records if r.get("provenance", {}).get("dataset") != "ai2thor"
    ]

    ai_train, ai_val, ai_test = _floorplan_split(
        ai2thor, args.train_frac, args.val_frac, args.seed
    )
    av_train, av_val, av_test = _stratified_split(
        averitec, args.train_frac, args.val_frac, args.seed
    )

    splits = {
        "train": sorted(ai_train + av_train),
        "val": sorted(ai_val + av_val),
        "test": sorted(ai_test + av_test),
    }

    meta = {"seed": args.seed, "train_frac": args.train_frac, "val_frac": args.val_frac}
    for split, idxs in splits.items():
        out_path = out_dir / f"{split}_indices.json"
        out_path.write_text(json.dumps({"indices": idxs, "meta": meta}, indent=2))

    total = len(records)
    print("=" * 60)
    print("Split Dataset: SUCCESS")
    print(f"Input     : {in_path}  ({total:,} records)")
    print(f"Output    : {out_dir}/")
    for split, idxs in splits.items():
        print(f"  {split:<6}: {len(idxs):,}  ({len(idxs) / total * 100:.1f}%)")
    print(f"Seed      : {args.seed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
