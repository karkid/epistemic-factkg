#!/usr/bin/env python3
"""
CLI for building semantic claims from a knowledge graph TTL file.
"""

import argparse
import sys
from pathlib import Path

from src.pipelines.result import BuildClaimsResult, BuildClaimsResultSummary
from src.pipelines.build_claims import build_claims


def main():
    parser = argparse.ArgumentParser(
        description="Build semantic claims from a knowledge graph TTL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "ttl_path", help="Path to the TTL file containing the knowledge graph"
    )

    parser.add_argument(
        "--output-dir", "-o", default="out", help="Directory to save claim JSONL files"
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Maximum number of contexts to process (default: all)",
    )

    parser.add_argument(
        "--n-claims",
        type=int,
        default=500,
        help="Number of claims to generate per method",
    )

    parser.add_argument(
        "--no-corruption",
        action="store_true",
        help="Disable corrupted claim generation",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate input file
    ttl_path = Path(args.ttl_path)
    if not ttl_path.exists():
        print(f"Error: TTL file not found: {ttl_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result: BuildClaimsResult = build_claims(
            ttl_path=str(ttl_path),
            output_dir=str(output_dir),
            max_contexts=args.max_contexts,
            n_claims=args.n_claims,
            add_corruption=not args.no_corruption,
            verbose=args.verbose,
        )

        summary = BuildClaimsResultSummary()
        summary.add_result(result)
        summary.print_summary(logger_func=print, show_files=args.verbose)
        print("=" * 80)
        print("Build Claims: SUCCESS")
        print("=" * 80)
        print(f"Input TTL      : {result.ttl_path}")
        print(f"Total Triples  : {result.total_triples}")
        print(f"Contexts       : {len(result.contexts)} ({', '.join(result.contexts)})")
        print(f"Total Claims   : {len(result.claim_corpus.claims)}")
        print(f"Output Files   : {len(result.output_files)}")

        for output_file in result.output_files:
            print(f"  - {output_file}")

        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
