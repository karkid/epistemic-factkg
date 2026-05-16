#!/usr/bin/env python3
"""Build PyG HeteroData graph dataset from filtered training JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.model.data.dataset import EpistemicFactDataset
from src.model.data.featurizer import Featurizer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert filtered training JSONL to serialised PyG graph dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path to filtered training JSONL")
    ap.add_argument("--output", required=True, help="Path to write graph_dataset.pt")
    ap.add_argument(
        "--embed-cache",
        default="out/graphs/embed_cache.pkl",
        help="Path to sentence-embedding cache .pkl",
    )
    ap.add_argument(
        "--force-rebuild", action="store_true", help="Rebuild even if .pt cache exists"
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force_rebuild:
        print(
            f"Graph dataset already exists at {out_path}. Use --force-rebuild to regenerate."
        )
        sys.exit(0)

    if args.verbose:
        print(f"Building graph dataset from {in_path}...")

    featurizer = Featurizer(cache_path=args.embed_cache)
    dataset = EpistemicFactDataset(
        jsonl_path=in_path,
        pt_cache=out_path,
        featurizer=featurizer,
        force_rebuild=args.force_rebuild,
    )

    print("=" * 60)
    print("Build Graph Dataset: SUCCESS")
    print(f"Input JSONL     : {in_path}")
    print(f"Output .pt      : {out_path}")
    print(f"Total graphs    : {len(dataset):,}")
    class_weights = dataset.get_class_weights()
    print(f"Class weights   : {class_weights.tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
