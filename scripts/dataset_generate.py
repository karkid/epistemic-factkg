#!/usr/bin/env python3
"""
Dataset Generator

Generate datasets from RDF knowledge graphs.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdf.rdf_to_dataset import main as rdf_main


def main():
    parser = argparse.ArgumentParser(
        description="🗃️ Generate Datasets from RDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("ttl_file", type=Path, help="Input TTL file path")
    parser.add_argument("output_file", type=Path, help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--onehop-per-floorplan",
        type=int,
        default=200,
        help="One-hop examples per floorplan",
    )
    parser.add_argument(
        "--neg-pairs-per-floorplan",
        type=int,
        default=60,
        help="Negation pairs per floorplan",
    )
    parser.add_argument(
        "--conj-per-floorplan",
        type=int,
        default=80,
        help="Conjunction examples per floorplan",
    )

    args = parser.parse_args()

    # Override sys.argv to pass to the original main function
    sys.argv = [
        "rdf_to_dataset.py",
        "--ttl",
        str(args.ttl_file),
        "--output",
        str(args.output_file),
        "--seed",
        str(args.seed),
        "--onehop_per_floorplan",
        str(args.onehop_per_floorplan),
        "--neg_pairs_per_floorplan",
        str(args.neg_pairs_per_floorplan),
        "--conj_per_floorplan",
        str(args.conj_per_floorplan),
    ]

    try:
        rdf_main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
